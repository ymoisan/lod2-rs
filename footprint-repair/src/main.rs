mod points;
mod quality;
mod territory;

use anyhow::{Context, Result};
use clap::Parser;
use gdal::spatial_ref::SpatialRef;
use gdal::vector::{
    FieldValue, Geometry, LayerAccess, LayerOptions, OGRFieldType,
    OGRwkbGeometryType::{wkbLinearRing, wkbPolygon},
};
use gdal::{Dataset, DriverManager};
use rstar::RTree;
use std::path::{Path, PathBuf};
use tracing::info;

use points::BuildingPointIndex;

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "footprint-repair",
    about = "Detect and repair truncated AutoBuilding footprints using orphaned class-6 points \
             with Voronoi-based assignment"
)]
struct Cli {
    /// Input footprints (AutoBuilding) GeoPackage.
    #[arg(long)]
    footprints: PathBuf,

    /// Classified LAS/LAZ point cloud.
    #[arg(long)]
    pointcloud: PathBuf,

    /// Output repaired GeoPackage.
    #[arg(long)]
    output: PathBuf,

    /// Buffer distance (m) around each footprint bbox to search for orphan points.
    #[arg(long, default_value_t = 5.0)]
    buffer: f64,

    /// Minimum orphan class-6 points to trigger repair.
    #[arg(long, default_value_t = 8)]
    min_orphan_points: usize,

    /// Rectangularity threshold below which a footprint is considered truncated.
    #[arg(long, default_value_t = 0.75)]
    rect_threshold: f64,

    /// EPSG code for output CRS.
    #[arg(long, default_value_t = 2961)]
    epsg: u32,

    /// Reject repair if area(repaired) / area(original) exceeds this (guardrail).
    #[arg(long, default_value_t = 1.5)]
    max_area_ratio: f64,

    /// Optional cap on outward growth: result is intersected with buffer(original, metres).
    /// Secondary to Voronoi clipping; use e.g. 10–25 when repairs still sprawl.
    #[arg(long)]
    max_growth_m: Option<f64>,

    /// Margin (m) around all footprint envelopes when building the global bbox for Voronoi cells.
    #[arg(long, default_value_t = 50.0)]
    bounds_margin: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

struct FootprintData {
    id: i64,
    geom: Geometry,
    centroid: (f64, f64),
}

struct RepairResult {
    id: i64,
    original_geom: Geometry,
    repaired_geom: Geometry,
    repaired: bool,
    /// e.g. `convex_hull_clipped_voronoi`, `rejected_intersection`, `rejected_area_ratio`
    action: String,
    rect_before: f64,
    rect_after: f64,
    orphan_count: usize,
    area_before: f64,
    area_after: f64,
    comments: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// I/O
// ─────────────────────────────────────────────────────────────────────────────

fn read_footprints(path: &Path) -> Result<Vec<FootprintData>> {
    let ds = Dataset::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut layer = ds.layer(0).context("layer 0")?;
    let mut out = Vec::new();
    let mut fid_counter = 0i64;

    for feature in layer.features() {
        let id = feature
            .fid()
            .map(|f| f as i64)
            .unwrap_or_else(|| {
                fid_counter += 1;
                fid_counter
            });

        let geom = match feature.geometry() {
            Some(g) => {
                let wkt = g.wkt().unwrap_or_default();
                Geometry::from_wkt(&wkt).context("re-parse geometry")?
            }
            None => continue,
        };

        if geom.geometry_count() == 0 && geom.point_count() == 0 {
            continue;
        }

        let centroid = polygon_centroid(&geom);
        out.push(FootprintData { id, geom, centroid });
    }
    info!("Read {} footprints from {}", out.len(), path.display());
    Ok(out)
}

fn polygon_centroid(geom: &Geometry) -> (f64, f64) {
    let verts = quality::exterior_vertices(geom);
    if verts.len() < 3 {
        return (0.0, 0.0);
    }
    let n = verts.len();
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    let mut signed_area = 0.0_f64;
    for i in 0..n {
        let j = (i + 1) % n;
        let a = verts[i].0 * verts[j].1 - verts[j].0 * verts[i].1;
        signed_area += a;
        cx += (verts[i].0 + verts[j].0) * a;
        cy += (verts[i].1 + verts[j].1) * a;
    }
    signed_area *= 0.5;
    if signed_area.abs() < 1e-10 {
        let sx: f64 = verts.iter().map(|v| v.0).sum();
        let sy: f64 = verts.iter().map(|v| v.1).sum();
        return (sx / n as f64, sy / n as f64);
    }
    cx /= 6.0 * signed_area;
    cy /= 6.0 * signed_area;
    (cx, cy)
}

// ─────────────────────────────────────────────────────────────────────────────
// Voronoi assignment via centroid R-tree (nearest-centroid = Voronoi cell)
// ─────────────────────────────────────────────────────────────────────────────

struct CentroidIndex {
    tree: RTree<[f64; 3]>, // [x, y, footprint_index_as_f64]
}

impl CentroidIndex {
    fn build(footprints: &[FootprintData]) -> Self {
        let pts: Vec<[f64; 3]> = footprints
            .iter()
            .enumerate()
            .map(|(i, fp)| [fp.centroid.0, fp.centroid.1, i as f64])
            .collect();
        Self {
            tree: RTree::bulk_load(pts),
        }
    }

    fn nearest_footprint_index(&self, x: f64, y: f64) -> Option<usize> {
        self.tree
            .nearest_neighbor(&[x, y, 0.0])
            .map(|pt| pt[2] as usize)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Orphan detection and repair
// ─────────────────────────────────────────────────────────────────────────────

fn detect_and_repair(
    footprints: &[FootprintData],
    bpi: &BuildingPointIndex,
    centroid_idx: &CentroidIndex,
    voronoi_cells: &[Geometry],
    cli: &Cli,
) -> Vec<RepairResult> {
    let mut results = Vec::with_capacity(footprints.len());

    for (fi, fp) in footprints.iter().enumerate() {
        let rect_before = quality::rectangularity(&fp.geom);
        let area_before = fp.geom.area().max(1e-12);
        let verts = quality::exterior_vertices(&fp.geom);
        let (min_x, min_y, max_x, max_y) = points::bbox(&verts);

        let candidates = bpi.points_in_bbox(
            min_x - cli.buffer,
            min_y - cli.buffer,
            max_x + cli.buffer,
            max_y + cli.buffer,
        );

        let mut orphans: Vec<(f64, f64)> = Vec::new();
        for pt in &candidates {
            if !points::point_in_polygon(pt[0], pt[1], &verts) {
                if let Some(nearest) = centroid_idx.nearest_footprint_index(pt[0], pt[1]) {
                    if nearest == fi {
                        orphans.push((pt[0], pt[1]));
                    }
                }
            }
        }

        let orphan_count = orphans.len();
        let needs_repair =
            orphan_count >= cli.min_orphan_points && rect_before < cli.rect_threshold;

        if needs_repair {
            let mut all_pts: Vec<(f64, f64)> = verts.clone();
            all_pts.extend_from_slice(&orphans);

            let hull = quality::convex_hull(&all_pts);
            if hull.len() < 3 {
                results.push(RepairResult {
                    id: fp.id,
                    original_geom: fp.geom.clone(),
                    repaired_geom: fp.geom.clone(),
                    repaired: false,
                    action: "skipped_degenerate_hull".to_string(),
                    rect_before,
                    rect_after: rect_before,
                    orphan_count,
                    area_before,
                    area_after: area_before,
                    comments: format!(
                        "convex hull < 3 vertices, rect={rect_before:.2}, {orphan_count} orphans"
                    ),
                });
                continue;
            }

            let candidate = match corners_to_polygon(&hull) {
                Some(g) => g,
                None => {
                    results.push(RepairResult {
                        id: fp.id,
                        original_geom: fp.geom.clone(),
                        repaired_geom: fp.geom.clone(),
                        repaired: false,
                        action: "skipped_invalid_hull".to_string(),
                        rect_before,
                        rect_after: rect_before,
                        orphan_count,
                        area_before,
                        area_after: area_before,
                        comments: "could not build polygon from hull".to_string(),
                    });
                    continue;
                }
            };

            let cell = &voronoi_cells[fi];
            let mut repaired = match candidate.intersection(cell) {
                Some(g) if g.area() > 1e-9 => g,
                _ => {
                    results.push(RepairResult {
                        id: fp.id,
                        original_geom: fp.geom.clone(),
                        repaired_geom: fp.geom.clone(),
                        repaired: false,
                        action: "rejected_intersection".to_string(),
                        rect_before,
                        rect_after: rect_before,
                        orphan_count,
                        area_before,
                        area_after: area_before,
                        comments:
                            "hull ∩ Voronoi cell empty or degenerate (GEOS off or geometry issue)"
                                .to_string(),
                    });
                    continue;
                }
            };

            if let Some(m) = cli.max_growth_m {
                match fp.geom.buffer(m, 8) {
                    Ok(cap) => {
                        repaired = match repaired.intersection(&cap) {
                            Some(g) if g.area() > 1e-9 => g,
                            _ => {
                                results.push(RepairResult {
                                    id: fp.id,
                                    original_geom: fp.geom.clone(),
                                    repaired_geom: fp.geom.clone(),
                                    repaired: false,
                                    action: "rejected_growth_cap".to_string(),
                                    rect_before,
                                    rect_after: rect_before,
                                    orphan_count,
                                    area_before,
                                    area_after: area_before,
                                    comments: format!(
                                        "hull ∩ Voronoi ∩ buffer({m} m) empty — keeping original"
                                    ),
                                });
                                continue;
                            }
                        };
                    }
                    Err(e) => {
                        results.push(RepairResult {
                            id: fp.id,
                            original_geom: fp.geom.clone(),
                            repaired_geom: fp.geom.clone(),
                            repaired: false,
                            action: "rejected_buffer".to_string(),
                            rect_before,
                            rect_after: rect_before,
                            orphan_count,
                            area_before,
                            area_after: area_before,
                            comments: format!("buffer({m} m) failed: {e}"),
                        });
                        continue;
                    }
                }
            }

            let repaired = territory::keep_largest_polygon(&repaired);
            let area_after = repaired.area().max(0.0);
            let ratio = area_after / area_before;

            if ratio > cli.max_area_ratio {
                results.push(RepairResult {
                    id: fp.id,
                    original_geom: fp.geom.clone(),
                    repaired_geom: fp.geom.clone(),
                    repaired: false,
                    action: "rejected_area_ratio".to_string(),
                    rect_before,
                    rect_after: rect_before,
                    orphan_count,
                    area_before,
                    area_after: area_before,
                    comments: format!(
                        "area ratio {ratio:.2} > max_area_ratio {} — keeping original",
                        cli.max_area_ratio
                    ),
                });
                continue;
            }

            let rect_after = quality::rectangularity(&repaired);
            let n_neighbours = count_nearby_centroids(footprints, fi, cli.buffer * 2.0);

            results.push(RepairResult {
                id: fp.id,
                original_geom: fp.geom.clone(),
                repaired_geom: repaired,
                repaired: true,
                action: "convex_hull_clipped_voronoi".to_string(),
                rect_before,
                rect_after,
                orphan_count,
                area_before,
                area_after,
                comments: format!(
                    "rect {rect_before:.2} -> {rect_after:.2}, hull+Voronoi clip, \
                     {orphan_count} orphans, ratio={ratio:.2}, \
                     {n_neighbours} neighbours within {:.0}m",
                    cli.buffer * 2.0
                ),
            });
        } else {
            let reason = if orphan_count < cli.min_orphan_points && rect_before < cli.rect_threshold
            {
                format!(
                    "truncated (rect={rect_before:.2}) but only {orphan_count} orphans (< {})",
                    cli.min_orphan_points
                )
            } else if orphan_count >= cli.min_orphan_points {
                format!(
                    "well-formed (rect={rect_before:.2}), {orphan_count} nearby orphans ignored"
                )
            } else {
                format!("ok (rect={rect_before:.2}, {orphan_count} orphans)")
            };

            results.push(RepairResult {
                id: fp.id,
                original_geom: fp.geom.clone(),
                repaired_geom: fp.geom.clone(),
                repaired: false,
                action: "unchanged".to_string(),
                rect_before,
                rect_after: rect_before,
                orphan_count,
                area_before,
                area_after: area_before,
                comments: reason,
            });
        }
    }

    results
}

fn count_nearby_centroids(footprints: &[FootprintData], self_idx: usize, radius: f64) -> usize {
    let self_c = footprints[self_idx].centroid;
    footprints
        .iter()
        .enumerate()
        .filter(|(i, fp)| {
            *i != self_idx && {
                let dx = fp.centroid.0 - self_c.0;
                let dy = fp.centroid.1 - self_c.1;
                (dx * dx + dy * dy).sqrt() < radius
            }
        })
        .count()
}

fn corners_to_polygon(corners: &[(f64, f64)]) -> Option<Geometry> {
    if corners.len() < 3 {
        return None;
    }
    let mut ring = Geometry::empty(wkbLinearRing).ok()?;
    for &(x, y) in corners {
        ring.add_point_2d((x, y));
    }
    ring.add_point_2d(corners[0]);
    let mut poly = Geometry::empty(wkbPolygon).ok()?;
    poly.add_geometry(ring).ok()?;
    Some(poly)
}

// ─────────────────────────────────────────────────────────────────────────────
// Class-6 coverage audit
// ─────────────────────────────────────────────────────────────────────────────

fn coverage_audit(footprints: &[FootprintData], bpi: &BuildingPointIndex, label: &str) {
    let total = bpi.total_class6();
    let mut inside = 0usize;
    for fp in footprints {
        inside += bpi.count_in_polygon(&fp.geom);
    }
    let orphaned = total.saturating_sub(inside);
    let pct = if total > 0 {
        100.0 * inside as f64 / total as f64
    } else {
        0.0
    };
    info!(
        "Coverage audit [{label}]: {total} class-6 pts, {inside} inside footprints \
         ({pct:.1}%), {orphaned} orphaned"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Output: repaired GeoPackage
// ─────────────────────────────────────────────────────────────────────────────

fn write_repaired_gpkg(path: &Path, results: &[RepairResult], epsg: u32) -> Result<()> {
    let driver =
        DriverManager::get_driver_by_name("GPKG").context("GDAL GPKG driver not available")?;
    let mut ds = driver
        .create_vector_only(path.to_str().context("non-UTF-8 path")?)
        .context("creating GeoPackage")?;

    let srs = SpatialRef::from_epsg(epsg).context("unknown EPSG")?;
    let opts = LayerOptions {
        name: "footprints",
        srs: Some(&srs),
        ty: wkbPolygon,
        options: None,
    };
    let mut layer = ds.create_layer(opts).context("creating layer")?;
    layer
        .create_defn_fields(&[("fid", OGRFieldType::OFTInteger64 as u32)])
        .context("fid field")?;

    for r in results {
        let wkt = r.repaired_geom.wkt().unwrap_or_default();
        if wkt.is_empty() {
            continue;
        }
        let parsed = Geometry::from_wkt(&wkt).context("re-parse for write")?;
        let geom = territory::keep_largest_polygon(&parsed);
        layer
            .create_feature_fields(
                geom,
                &["fid"],
                &[FieldValue::Integer64Value(r.id)],
            )
            .context("writing feature")?;
    }

    let n = results.len();
    info!("Wrote {n} features to {}", path.display());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Output: audit GeoParquet (or fallback GeoPackage)
// ─────────────────────────────────────────────────────────────────────────────

fn write_audit(path: &Path, results: &[RepairResult], epsg: u32) -> Result<()> {
    let modified: Vec<&RepairResult> = results.iter().filter(|r| r.repaired).collect();
    if modified.is_empty() {
        info!("No footprints were repaired — skipping audit file");
        return Ok(());
    }

    let (driver_name, ext) = if DriverManager::get_driver_by_name("Parquet").is_ok() {
        ("Parquet", "parquet")
    } else {
        info!("GDAL Parquet driver not available, falling back to GeoPackage for audit");
        ("GPKG", "gpkg")
    };

    let audit_path = path.with_extension(ext);
    let driver =
        DriverManager::get_driver_by_name(driver_name).context("driver not available")?;
    let mut ds = driver
        .create_vector_only(audit_path.to_str().context("non-UTF-8 path")?)
        .context("creating audit dataset")?;

    let srs = SpatialRef::from_epsg(epsg).context("unknown EPSG")?;
    let opts = LayerOptions {
        name: "audit",
        srs: Some(&srs),
        ty: wkbPolygon,
        options: None,
    };
    let mut layer = ds.create_layer(opts).context("creating audit layer")?;

    layer
        .create_defn_fields(&[
            ("fid", OGRFieldType::OFTInteger64 as u32),
            ("action", OGRFieldType::OFTString as u32),
            ("original_wkt", OGRFieldType::OFTString as u32),
            ("rect_before", OGRFieldType::OFTReal as u32),
            ("rect_after", OGRFieldType::OFTReal as u32),
            ("orphan_count", OGRFieldType::OFTInteger64 as u32),
            ("area_before_m2", OGRFieldType::OFTReal as u32),
            ("area_after_m2", OGRFieldType::OFTReal as u32),
            ("comments", OGRFieldType::OFTString as u32),
        ])
        .context("defining audit fields")?;

    for r in &modified {
        let wkt = r.repaired_geom.wkt().unwrap_or_default();
        if wkt.is_empty() {
            continue;
        }
        let parsed = Geometry::from_wkt(&wkt).context("re-parse for audit")?;
        let geom = territory::keep_largest_polygon(&parsed);
        let orig_wkt = r.original_geom.wkt().unwrap_or_default();

        layer
            .create_feature_fields(
                geom,
                &[
                    "fid",
                    "action",
                    "original_wkt",
                    "rect_before",
                    "rect_after",
                    "orphan_count",
                    "area_before_m2",
                    "area_after_m2",
                    "comments",
                ],
                &[
                    FieldValue::Integer64Value(r.id),
                    FieldValue::StringValue(r.action.clone()),
                    FieldValue::StringValue(orig_wkt),
                    FieldValue::RealValue(r.rect_before),
                    FieldValue::RealValue(r.rect_after),
                    FieldValue::Integer64Value(r.orphan_count as i64),
                    FieldValue::RealValue(r.area_before),
                    FieldValue::RealValue(r.area_after),
                    FieldValue::StringValue(r.comments.clone()),
                ],
            )
            .context("writing audit feature")?;
    }

    info!(
        "Wrote {} audit features to {}",
        modified.len(),
        audit_path.display()
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    // 1. Load inputs
    let footprints = read_footprints(&cli.footprints)?;
    let bpi = BuildingPointIndex::from_laz(&cli.pointcloud)?;

    // 2. Coverage audit BEFORE repair
    coverage_audit(&footprints, &bpi, "original_ab");

    // 3. Build centroid index (Voronoi equivalent)
    info!("Building centroid index for {} footprints …", footprints.len());
    let centroid_idx = CentroidIndex::build(&footprints);

    let bounds = territory::global_bounds_from_vertices(
        footprints
            .iter()
            .map(|f| quality::exterior_vertices(&f.geom)),
        cli.bounds_margin,
    );
    let centroids: Vec<(f64, f64)> = footprints.iter().map(|f| f.centroid).collect();
    info!("Precomputing Voronoi cells (global bbox margin {} m) …", cli.bounds_margin);
    let mut voronoi_cells: Vec<Geometry> = Vec::with_capacity(footprints.len());
    for fi in 0..footprints.len() {
        let cell = territory::voronoi_cell_for_site(&centroids, fi, bounds)
            .with_context(|| format!("Voronoi cell for footprint index {fi}"))?;
        voronoi_cells.push(cell);
    }

    // 4. Detect orphans and repair
    info!("Detecting orphans and repairing footprints …");
    let results = detect_and_repair(&footprints, &bpi, &centroid_idx, &voronoi_cells, &cli);

    let n_repaired = results.iter().filter(|r| r.repaired).count();
    let n_unchanged = results.len() - n_repaired;
    let total_orphans: usize = results.iter().map(|r| r.orphan_count).sum();
    let repaired_orphans: usize = results
        .iter()
        .filter(|r| r.repaired)
        .map(|r| r.orphan_count)
        .sum();

    info!(
        "Repair complete: {n_repaired} repaired, {n_unchanged} unchanged, \
         {total_orphans} total orphans detected ({repaired_orphans} rescued)"
    );

    // 5. Coverage audit AFTER repair
    let repaired_fps: Vec<FootprintData> = results
        .iter()
        .map(|r| FootprintData {
            id: r.id,
            geom: r.repaired_geom.clone(),
            centroid: polygon_centroid(&r.repaired_geom),
        })
        .collect();
    coverage_audit(&repaired_fps, &bpi, "repaired_ab");

    // 6. Write outputs
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    write_repaired_gpkg(&cli.output, &results, cli.epsg)?;

    let audit_stem = cli.output.with_extension("");
    let audit_base = format!("{}-audit", audit_stem.display());
    write_audit(Path::new(&audit_base), &results, cli.epsg)?;

    info!("Done.");
    Ok(())
}
