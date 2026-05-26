mod points;
mod quality;

use anyhow::{Context, Result};
use clap::Parser;
use gdal::spatial_ref::SpatialRef;
use gdal::vector::{
    FieldValue, Geometry, LayerAccess, LayerOptions,
    OGRFieldType,
    OGRwkbGeometryType::{wkbLinearRing, wkbMultiPolygon, wkbPolygon},
};
use gdal::{Dataset, DriverManager};
use std::path::{Path, PathBuf};
use tracing::info;

use points::BuildingPointIndex;

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "footprint-merge",
    about = "Merge AutoBuilding (reference) and Overture (complement) footprints with \
             offset correction and shape-quality diagnostics"
)]
struct Cli {
    /// Reference footprints (AutoBuilding) GeoPackage.
    #[arg(long)]
    reference: PathBuf,

    /// Complement footprints (Overture) GeoPackage.
    #[arg(long)]
    complement: PathBuf,

    /// Classified LAS/LAZ point cloud (for building-point confirmation).
    #[arg(long)]
    pointcloud: PathBuf,

    /// Output merged GeoPackage (pipeline input).
    #[arg(long)]
    output: PathBuf,

    /// Offset correction mode: "auto" computes median centroid translation,
    /// "none" skips correction.  Use --offset-dx/--offset-dy for manual.
    #[arg(long, default_value = "auto")]
    offset_correction: String,

    /// Manual X offset to apply to complement footprints (metres).
    #[arg(long)]
    offset_dx: Option<f64>,

    /// Manual Y offset to apply to complement footprints (metres).
    #[arg(long)]
    offset_dy: Option<f64>,

    /// Maximum centroid distance (m) to consider two footprints a match.
    #[arg(long, default_value_t = 15.0)]
    match_radius: f64,

    /// Minimum building points for an unmatched complement footprint to be added.
    #[arg(long, default_value_t = 10)]
    min_building_points: usize,

    /// AB rectangularity below which the footprint is considered truncated.
    #[arg(long, default_value_t = 0.7)]
    rect_truncated: f64,

    /// Complement rectangularity above which the footprint is considered regular.
    #[arg(long, default_value_t = 0.8)]
    rect_regular: f64,

    /// Area ratio (AB/Overture) below which AB is considered heavily truncated.
    #[arg(long, default_value_t = 0.5)]
    area_ratio_low: f64,

    /// EPSG code for output CRS.
    #[arg(long, default_value_t = 2961)]
    epsg: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

struct FootprintData {
    id: i64,
    geom: Geometry,
    centroid: (f64, f64),
    area: f64,
}

#[derive(Clone, Copy)]
enum Decision {
    AbOnly,
    Union,
    ReplacedByComplement,
    ComplementAdded,
    ComplementDiscarded,
}

impl Decision {
    fn source_label(self) -> &'static str {
        match self {
            Decision::AbOnly => "ab_only",
            Decision::Union => "union",
            Decision::ReplacedByComplement => "replaced_by_overture",
            Decision::ComplementAdded => "overture_only",
            Decision::ComplementDiscarded => "discarded",
        }
    }
}

struct MergeResult {
    geom: Geometry,
    decision: Decision,
    comments: String,
    ab_rect: Option<f64>,
    comp_rect: Option<f64>,
    area_ratio: Option<f64>,
    building_pt_count: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Geometry helpers
// ─────────────────────────────────────────────────────────────────────────────

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

fn translate_polygon(geom: &Geometry, dx: f64, dy: f64) -> Result<Geometry> {
    let gt = geom.geometry_type();
    if gt == wkbMultiPolygon {
        let mut mp = Geometry::empty(wkbMultiPolygon).context("empty multi")?;
        for i in 0..geom.geometry_count() {
            let sub = geom.get_geometry(i);
            let translated = translate_single_polygon(&sub, dx, dy)?;
            mp.add_geometry(translated).context("add sub-polygon")?;
        }
        return Ok(mp);
    }
    translate_single_polygon(geom, dx, dy)
}

fn translate_single_polygon(geom: &Geometry, dx: f64, dy: f64) -> Result<Geometry> {
    let ring_count = geom.geometry_count();
    let mut new_poly = Geometry::empty(wkbPolygon).context("empty polygon")?;
    for i in 0..ring_count {
        let ring = geom.get_geometry(i);
        let pts = ring.get_point_vec();
        let mut new_ring = Geometry::empty(wkbLinearRing).context("empty ring")?;
        for (x, y, _z) in pts {
            new_ring.add_point_2d((x + dx, y + dy));
        }
        new_poly.add_geometry(new_ring).context("add ring")?;
    }
    Ok(new_poly)
}

// ─────────────────────────────────────────────────────────────────────────────
// I/O: read footprints from a GeoPackage
// ─────────────────────────────────────────────────────────────────────────────

fn read_footprints(path: &Path) -> Result<Vec<FootprintData>> {
    let ds = Dataset::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut layer = ds.layer(0).context("layer 0")?;
    let mut out = Vec::new();
    let mut fid_counter = 0i64;

    for feature in layer.features() {
        let id = feature.fid().map(|f| f as i64).unwrap_or_else(|| {
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
        let area = geom.area();
        out.push(FootprintData { id, geom, centroid, area });
    }
    info!("Read {} footprints from {}", out.len(), path.display());
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Offset computation (Option A: median centroid translation)
// ─────────────────────────────────────────────────────────────────────────────

/// Match footprints by IoU: for each complement footprint, find the reference
/// footprint with highest IoU (geometric overlap).  A centroid-distance pre-filter
/// avoids O(n*m) intersection tests.
fn iou_matches(
    reference: &[FootprintData],
    complement: &[FootprintData],
    match_radius: f64,
    min_iou: f64,
) -> Vec<(usize, usize)> {
    let mut ref_taken = vec![false; reference.len()];
    let mut pairs = Vec::new();

    // Sort complement by area descending so larger buildings match first
    let mut comp_order: Vec<usize> = (0..complement.len()).collect();
    comp_order.sort_by(|&a, &b| complement[b].area.partial_cmp(&complement[a].area).unwrap());

    for &ci in &comp_order {
        let comp = &complement[ci];
        let mut best_iou = 0.0_f64;
        let mut best_ri: Option<usize> = None;

        for (ri, rf) in reference.iter().enumerate() {
            if ref_taken[ri] {
                continue;
            }
            if centroid_dist(rf.centroid, comp.centroid) > match_radius {
                continue;
            }
            let iou = compute_iou(&rf.geom, &comp.geom);
            if iou > best_iou {
                best_iou = iou;
                best_ri = Some(ri);
            }
        }

        if best_iou >= min_iou {
            if let Some(ri) = best_ri {
                ref_taken[ri] = true;
                pairs.push((ri, ci));
            }
        }
    }
    pairs
}

fn compute_iou(a: &Geometry, b: &Geometry) -> f64 {
    let area_a = a.area();
    let area_b = b.area();
    if area_a <= 0.0 || area_b <= 0.0 {
        return 0.0;
    }
    let intersection = match a.intersection(b) {
        Some(g) => g.area(),
        None => return 0.0,
    };
    let union_area = area_a + area_b - intersection;
    if union_area <= 0.0 {
        return 0.0;
    }
    intersection / union_area
}

fn compute_median_offset(
    reference: &[FootprintData],
    complement: &[FootprintData],
    match_radius: f64,
) -> (f64, f64) {
    // For offset computation, use centroid proximity (looser than IoU, since
    // we haven't corrected the offset yet).  Accept any pair within radius
    // where the complement has only one plausible reference within 2x radius.
    let pairs = unambiguous_centroid_matches(reference, complement, match_radius);

    let mut dx_vec: Vec<f64> = Vec::with_capacity(pairs.len());
    let mut dy_vec: Vec<f64> = Vec::with_capacity(pairs.len());

    for &(ri, ci) in &pairs {
        let rf = &reference[ri];
        let comp = &complement[ci];
        dx_vec.push(comp.centroid.0 - rf.centroid.0);
        dy_vec.push(comp.centroid.1 - rf.centroid.1);
    }

    if dx_vec.is_empty() {
        info!("No mutual nearest-neighbor pairs found for offset computation");
        return (0.0, 0.0);
    }

    let median_dx = median(&mut dx_vec);
    let median_dy = median(&mut dy_vec);

    let mean_dx: f64 = dx_vec.iter().sum::<f64>() / dx_vec.len() as f64;
    let mean_dy: f64 = dy_vec.iter().sum::<f64>() / dy_vec.len() as f64;
    let std_dx = std_dev(&dx_vec, mean_dx);
    let std_dy = std_dev(&dy_vec, mean_dy);

    info!(
        "Offset from {} unambiguous centroid pairs: median=({:.3}, {:.3}), mean=({:.3}, {:.3}), std=({:.3}, {:.3})",
        pairs.len(), median_dx, median_dy, mean_dx, mean_dy, std_dx, std_dy
    );

    (median_dx, median_dy)
}

/// For offset computation: match complement→reference by nearest centroid, but
/// only accept pairs where the nearest reference is unambiguously closer than
/// the second-nearest (ratio < 0.7).  This filters out dense clusters where
/// the nearest centroid might be the wrong building.
fn unambiguous_centroid_matches(
    reference: &[FootprintData],
    complement: &[FootprintData],
    match_radius: f64,
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (ci, comp) in complement.iter().enumerate() {
        let mut dists: Vec<(usize, f64)> = reference
            .iter()
            .enumerate()
            .map(|(i, r)| (i, centroid_dist(r.centroid, comp.centroid)))
            .filter(|(_, d)| *d <= match_radius)
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        if dists.is_empty() {
            continue;
        }
        let (best_ri, best_d) = dists[0];
        let unambiguous = if dists.len() >= 2 {
            let second_d = dists[1].1;
            second_d > 0.0 && best_d / second_d < 0.7
        } else {
            true
        };
        if unambiguous {
            pairs.push((best_ri, ci));
        }
    }
    info!("{} unambiguous centroid pairs out of {} complement footprints", pairs.len(), complement.len());
    pairs
}

fn centroid_dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

fn std_dev(v: &[f64], mean: f64) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let var: f64 = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Merge logic
// ─────────────────────────────────────────────────────────────────────────────

fn run_merge(
    reference: &[FootprintData],
    complement: &[FootprintData],
    bpi: &BuildingPointIndex,
    cli: &Cli,
) -> Vec<MergeResult> {
    let pairs = iou_matches(reference, complement, cli.match_radius, 0.05);
    info!("{} IoU-matched pairs found", pairs.len());

    let mut ab_matched = vec![false; reference.len()];
    let mut comp_matched = vec![false; complement.len()];
    let mut results: Vec<MergeResult> = Vec::new();

    for &(ri, ci) in &pairs {
        ab_matched[ri] = true;
        comp_matched[ci] = true;
        let result = merge_pair(&reference[ri], &complement[ci], bpi, cli);
        results.push(result);
    }

    // Unmatched AB: keep as-is
    for (i, rf) in reference.iter().enumerate() {
        if !ab_matched[i] {
            let pt_count = bpi.count_in_polygon(&rf.geom);
            results.push(MergeResult {
                geom: rf.geom.clone(),
                decision: Decision::AbOnly,
                comments: format!(
                    "AB-only (rect={:.2}), no matching Overture footprint",
                    quality::rectangularity(&rf.geom)
                ),
                ab_rect: Some(quality::rectangularity(&rf.geom)),
                comp_rect: None,
                area_ratio: None,
                building_pt_count: pt_count,
            });
        }
    }

    // Unmatched Overture: add if building points confirm, else discard
    for (j, comp) in complement.iter().enumerate() {
        if !comp_matched[j] {
            let pt_count = bpi.count_in_polygon(&comp.geom);
            if pt_count >= cli.min_building_points {
                results.push(MergeResult {
                    geom: comp.geom.clone(),
                    decision: Decision::ComplementAdded,
                    comments: format!(
                        "Overture-only, {} building pts confirmed",
                        pt_count
                    ),
                    ab_rect: None,
                    comp_rect: Some(quality::rectangularity(&comp.geom)),
                    area_ratio: None,
                    building_pt_count: pt_count,
                });
            } else {
                results.push(MergeResult {
                    geom: comp.geom.clone(),
                    decision: Decision::ComplementDiscarded,
                    comments: format!(
                        "Overture-only, discarded — {} building pts",
                        pt_count
                    ),
                    ab_rect: None,
                    comp_rect: Some(quality::rectangularity(&comp.geom)),
                    area_ratio: None,
                    building_pt_count: pt_count,
                });
            }
        }
    }

    results
}

fn merge_pair(
    rf: &FootprintData,
    comp: &FootprintData,
    bpi: &BuildingPointIndex,
    cli: &Cli,
) -> MergeResult {
    let ab_rect = quality::rectangularity(&rf.geom);
    let comp_rect = quality::rectangularity(&comp.geom);
    let area_ratio = if comp.area > 0.0 { rf.area / comp.area } else { 1.0 };

    // Case 2: AB truncated, complement regular
    if ab_rect < cli.rect_truncated && comp_rect >= cli.rect_regular {
        let diff_pts = match rf.geom.difference(&comp.geom) {
            Some(diff) => bpi.count_in_polygon(&diff),
            None => 0,
        };
        let comp_pts = bpi.count_in_polygon(&comp.geom);

        if comp_pts >= cli.min_building_points {
            return MergeResult {
                geom: comp.geom.clone(),
                decision: Decision::ReplacedByComplement,
                comments: format!(
                    "AB truncated (rect={ab_rect:.2} vs Overture rect={comp_rect:.2}), \
                     replaced — {comp_pts} class-6 pts confirm Overture extent"
                ),
                ab_rect: Some(ab_rect),
                comp_rect: Some(comp_rect),
                area_ratio: Some(area_ratio),
                building_pt_count: comp_pts,
            };
        } else {
            return MergeResult {
                geom: rf.geom.clone(),
                decision: Decision::AbOnly,
                comments: format!(
                    "AB truncated (rect={ab_rect:.2}) but Overture not confirmed \
                     ({comp_pts} pts, diff_pts={diff_pts}), keeping AB"
                ),
                ab_rect: Some(ab_rect),
                comp_rect: Some(comp_rect),
                area_ratio: Some(area_ratio),
                building_pt_count: bpi.count_in_polygon(&rf.geom),
            };
        }
    }

    // Case 3: AB much smaller
    if area_ratio < cli.area_ratio_low {
        let comp_pts = bpi.count_in_polygon(&comp.geom);
        if comp_pts >= cli.min_building_points {
            let union_geom = rf.geom.union(&comp.geom).unwrap_or_else(|| rf.geom.clone());
            let union_pts = bpi.count_in_polygon(&union_geom);
            return MergeResult {
                geom: union_geom,
                decision: Decision::Union,
                comments: format!(
                    "AB heavily truncated (area_ratio={area_ratio:.2}), \
                     union with Overture — {union_pts} building pts"
                ),
                ab_rect: Some(ab_rect),
                comp_rect: Some(comp_rect),
                area_ratio: Some(area_ratio),
                building_pt_count: union_pts,
            };
        }
    }

    // Case 1 / Default: union
    let union_geom = rf.geom.union(&comp.geom).unwrap_or_else(|| rf.geom.clone());
    let union_pts = bpi.count_in_polygon(&union_geom);
    let case = if ab_rect >= 0.8 && area_ratio >= 0.7 {
        format!("AB well-formed (rect={ab_rect:.2}), minor union with Overture (rect={comp_rect:.2})")
    } else {
        format!("Default union (AB rect={ab_rect:.2}, Overture rect={comp_rect:.2}, area_ratio={area_ratio:.2})")
    };
    MergeResult {
        geom: union_geom,
        decision: Decision::Union,
        comments: case,
        ab_rect: Some(ab_rect),
        comp_rect: Some(comp_rect),
        area_ratio: Some(area_ratio),
        building_pt_count: union_pts,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Output: GeoPackage (pipeline) + audit (GeoParquet or fallback GeoPackage)
// ─────────────────────────────────────────────────────────────────────────────

fn write_pipeline_gpkg(path: &Path, results: &[MergeResult], epsg: u32) -> Result<()> {
    let active: Vec<&MergeResult> = results
        .iter()
        .filter(|r| !matches!(r.decision, Decision::ComplementDiscarded))
        .collect();

    let driver = DriverManager::get_driver_by_name("GPKG")
        .context("GDAL GPKG driver not available")?;
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

    for (i, r) in active.iter().enumerate() {
        let wkt = r.geom.wkt().unwrap_or_default();
        if wkt.is_empty() {
            continue;
        }
        let geom = Geometry::from_wkt(&wkt).context("re-parse for write")?;
        layer
            .create_feature_fields(
                geom,
                &["fid"],
                &[FieldValue::Integer64Value(i as i64)],
            )
            .context("writing feature")?;
    }

    info!("Wrote {} features to {}", active.len(), path.display());
    Ok(())
}

fn write_corrected_complement(
    path: &Path,
    complement: &[FootprintData],
    epsg: u32,
) -> Result<()> {
    let driver = DriverManager::get_driver_by_name("GPKG")
        .context("GDAL GPKG driver not available")?;
    let mut ds = driver
        .create_vector_only(path.to_str().context("non-UTF-8 path")?)
        .context("creating corrected GeoPackage")?;

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

    for fp in complement {
        let wkt = fp.geom.wkt().unwrap_or_default();
        if wkt.is_empty() {
            continue;
        }
        let geom = Geometry::from_wkt(&wkt).context("re-parse")?;
        layer
            .create_feature_fields(
                geom,
                &["fid"],
                &[FieldValue::Integer64Value(fp.id)],
            )
            .context("writing feature")?;
    }

    info!(
        "Wrote {} corrected complement footprints to {}",
        complement.len(),
        path.display()
    );
    Ok(())
}

fn write_audit(path: &Path, results: &[MergeResult], epsg: u32) -> Result<()> {
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
            ("source", OGRFieldType::OFTString as u32),
            ("comments", OGRFieldType::OFTString as u32),
            ("ab_rect", OGRFieldType::OFTReal as u32),
            ("overture_rect", OGRFieldType::OFTReal as u32),
            ("area_ratio", OGRFieldType::OFTReal as u32),
            ("building_point_count", OGRFieldType::OFTInteger64 as u32),
        ])
        .context("defining audit fields")?;

    for (i, r) in results.iter().enumerate() {
        let wkt = r.geom.wkt().unwrap_or_default();
        if wkt.is_empty() {
            continue;
        }
        let geom = Geometry::from_wkt(&wkt).context("re-parse for audit")?;

        let ab_r = r.ab_rect.unwrap_or(f64::NAN);
        let comp_r = r.comp_rect.unwrap_or(f64::NAN);
        let ar = r.area_ratio.unwrap_or(f64::NAN);

        layer
            .create_feature_fields(
                geom,
                &[
                    "fid",
                    "source",
                    "comments",
                    "ab_rect",
                    "overture_rect",
                    "area_ratio",
                    "building_point_count",
                ],
                &[
                    FieldValue::Integer64Value(i as i64),
                    FieldValue::StringValue(r.decision.source_label().to_string()),
                    FieldValue::StringValue(r.comments.clone()),
                    FieldValue::RealValue(ab_r),
                    FieldValue::RealValue(comp_r),
                    FieldValue::RealValue(ar),
                    FieldValue::Integer64Value(r.building_pt_count as i64),
                ],
            )
            .context("writing audit feature")?;
    }

    info!(
        "Wrote {} audit features to {}",
        results.len(),
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

    // 1. Read inputs
    let reference = read_footprints(&cli.reference)?;
    let mut complement = read_footprints(&cli.complement)?;
    let bpi = BuildingPointIndex::from_laz(&cli.pointcloud)?;

    // 2. Compute and apply offset correction
    let (dx, dy) = if let (Some(dx), Some(dy)) = (cli.offset_dx, cli.offset_dy) {
        info!("Using manual offset: ({dx}, {dy})");
        (dx, dy)
    } else if cli.offset_correction == "auto" {
        let (dx, dy) = compute_median_offset(&reference, &complement, cli.match_radius);
        info!("Auto offset correction: ({dx:.3}, {dy:.3})");
        (dx, dy)
    } else {
        info!("Offset correction disabled");
        (0.0, 0.0)
    };

    if dx.abs() > 1e-6 || dy.abs() > 1e-6 {
        info!("Translating complement footprints by ({:.3}, {:.3}) …", -dx, -dy);
        for fp in &mut complement {
            fp.geom = translate_polygon(&fp.geom, -dx, -dy)?;
            fp.centroid = (fp.centroid.0 - dx, fp.centroid.1 - dy);
        }
    }

    // 3. Write corrected complement for Overture-only runs
    let corrected_path = cli
        .output
        .parent()
        .unwrap_or(Path::new("."))
        .join("overture-corrected.gpkg");
    write_corrected_complement(&corrected_path, &complement, cli.epsg)?;

    // 4. Run merge
    info!("Running merge ({} AB, {} complement) …", reference.len(), complement.len());
    let results = run_merge(&reference, &complement, &bpi, &cli);

    let n_active = results
        .iter()
        .filter(|r| !matches!(r.decision, Decision::ComplementDiscarded))
        .count();
    let n_ab_only = results.iter().filter(|r| matches!(r.decision, Decision::AbOnly)).count();
    let n_union = results.iter().filter(|r| matches!(r.decision, Decision::Union)).count();
    let n_replaced = results
        .iter()
        .filter(|r| matches!(r.decision, Decision::ReplacedByComplement))
        .count();
    let n_added = results
        .iter()
        .filter(|r| matches!(r.decision, Decision::ComplementAdded))
        .count();
    let n_discarded = results
        .iter()
        .filter(|r| matches!(r.decision, Decision::ComplementDiscarded))
        .count();

    info!("Merge results: {n_active} active footprints");
    info!("  AB-only: {n_ab_only}");
    info!("  Union: {n_union}");
    info!("  Replaced by Overture: {n_replaced}");
    info!("  Overture added: {n_added}");
    info!("  Overture discarded: {n_discarded}");

    // 5. Write outputs
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    write_pipeline_gpkg(&cli.output, &results, cli.epsg)?;

    let audit_stem = cli.output.with_extension("");
    let audit_base = format!("{}-audit", audit_stem.display());
    write_audit(Path::new(&audit_base), &results, cli.epsg)?;

    info!("Done.");
    Ok(())
}
