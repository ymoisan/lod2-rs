/// Extract squarish building footprints from a classified LAS/LAZ point cloud.
///
/// Pipeline
/// ────────
///   1. Read class-6 (building) and class-2 (ground) points; rasterise both onto
///      a shared regular grid.
///   2. Morphologically close the building layer (dilate by `--gap-fill` cells,
///      then erode back) so that nearby building sections of the same structure
///      are merged into one component.
///   3. BFS connected components on the closed grid; discard small ones.
///   4. For each component:
///        a. Minimum-area bounding rectangle (rotating calipers) → outer shape.
///        b. Inside the component's axis-aligned bounding box, find cells that
///           have ground points but no building points.
///        c. BFS connected components of those ground-only cells.
///        d. Each large ground cluster is subtracted from the outer polygon using
///           GDAL geometry difference (punching out parking lots / courtyards).
///   5. Write every resulting polygon / multi-polygon to a GeoPackage.
use anyhow::{Context, Result, bail};
use clap::Parser;
use gdal::DriverManager;
use gdal::spatial_ref::SpatialRef;
use gdal::vector::OGRFieldType;
use gdal::vector::OGRwkbGeometryType::{wkbLinearRing, wkbMultiPolygon, wkbPolygon};
use gdal::vector::{FieldValue, Geometry, LayerAccess, LayerOptions};
use las::Reader;
use nalgebra::Point2;
use std::collections::VecDeque;
use std::path::PathBuf;
use tracing::info;

const CLASS_BUILDING: u8 = 6;
const CLASS_GROUND: u8 = 2;

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "footprint-extract",
    about = "Derive squarish building footprints from a classified LAS/LAZ point cloud"
)]
struct Cli {
    /// Input LAS/LAZ file.
    input: PathBuf,

    /// Output directory (defaults to the input file's directory).
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Output GeoPackage filename stem (no extension).
    #[arg(long, default_value = "footprints")]
    output_name: String,

    /// Grid cell size in metres.
    #[arg(long, default_value_t = 0.5)]
    resolution: f64,

    /// Minimum occupied cells for a component to become a footprint.
    #[arg(long, default_value_t = 20)]
    min_cells: usize,

    /// Morphological gap-fill radius in grid cells.
    /// Closing bridges gaps up to 2×radius×resolution metres between building
    /// sections that belong to the same structure.
    #[arg(long, default_value_t = 4)]
    gap_fill: usize,

    /// Minimum ground-only cells for a region inside a footprint to be
    /// subtracted as a hole (parking lot, courtyard, …).
    #[arg(long, default_value_t = 40)]
    hole_min_cells: usize,

    /// EPSG code.  Read from the LAS header when omitted.
    #[arg(long)]
    epsg: Option<u32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Dual-class occupancy grid
// ─────────────────────────────────────────────────────────────────────────────

struct Grid {
    build: Vec<bool>,
    ground: Vec<bool>,
    cols: usize,
    rows: usize,
    origin_x: f64,
    origin_y: f64,
    res: f64,
}

impl Grid {
    fn new(
        build_pts: &[(f64, f64)],
        ground_pts: &[(f64, f64)],
        res: f64,
    ) -> Self {
        let all = build_pts.iter().chain(ground_pts.iter());
        let min_x = all.clone().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let min_y = all.clone().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let max_x = all.clone().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let max_y = all.map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

        let cols = ((max_x - min_x) / res).ceil() as usize + 1;
        let rows = ((max_y - min_y) / res).ceil() as usize + 1;
        let mut build = vec![false; cols * rows];
        let mut ground = vec![false; cols * rows];

        for &(x, y) in build_pts {
            let c = ((x - min_x) / res) as usize;
            let r = ((y - min_y) / res) as usize;
            build[r * cols + c] = true;
        }
        for &(x, y) in ground_pts {
            let c = ((x - min_x) / res) as usize;
            let r = ((y - min_y) / res) as usize;
            ground[r * cols + c] = true;
        }

        Grid { build, ground, cols, rows, origin_x: min_x, origin_y: min_y, res }
    }

    fn cell_centre(&self, col: usize, row: usize) -> (f64, f64) {
        (
            self.origin_x + (col as f64 + 0.5) * self.res,
            self.origin_y + (row as f64 + 0.5) * self.res,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Morphological operations on flat bool slices
// ─────────────────────────────────────────────────────────────────────────────

fn dilate(occ: &[bool], cols: usize, rows: usize, radius: usize) -> Vec<bool> {
    if radius == 0 {
        return occ.to_vec();
    }
    let mut out = vec![false; cols * rows];
    for r in 0..rows {
        for c in 0..cols {
            if !occ[r * cols + c] {
                continue;
            }
            let r0 = r.saturating_sub(radius);
            let r1 = (r + radius + 1).min(rows);
            let c0 = c.saturating_sub(radius);
            let c1 = (c + radius + 1).min(cols);
            for nr in r0..r1 {
                for nc in c0..c1 {
                    out[nr * cols + nc] = true;
                }
            }
        }
    }
    out
}

fn erode(occ: &[bool], cols: usize, rows: usize, radius: usize) -> Vec<bool> {
    if radius == 0 {
        return occ.to_vec();
    }
    // erode = complement of dilate(complement)
    let comp: Vec<bool> = occ.iter().map(|&b| !b).collect();
    let d = dilate(&comp, cols, rows, radius);
    d.iter().map(|&b| !b).collect()
}

/// Morphological closing: bridges gaps up to 2×radius cells wide.
fn morph_close(occ: &[bool], cols: usize, rows: usize, radius: usize) -> Vec<bool> {
    let d = dilate(occ, cols, rows, radius);
    erode(&d, cols, rows, radius)
}

// ─────────────────────────────────────────────────────────────────────────────
// BFS connected components
// ─────────────────────────────────────────────────────────────────────────────

fn connected_components(
    occ: &[bool],
    cols: usize,
    rows: usize,
) -> Vec<Vec<(usize, usize)>> {
    let mut visited = vec![false; cols * rows];
    let mut components = Vec::new();

    for start_r in 0..rows {
        for start_c in 0..cols {
            let idx = start_r * cols + start_c;
            if !occ[idx] || visited[idx] {
                continue;
            }
            let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
            queue.push_back((start_c, start_r));
            visited[idx] = true;
            let mut comp = Vec::new();

            while let Some((c, r)) = queue.pop_front() {
                comp.push((c, r));
                for (dc, dr) in [
                    (usize::MAX, 0usize),
                    (1, 0),
                    (0, usize::MAX),
                    (0, 1),
                ] {
                    let nc = c.wrapping_add(dc);
                    let nr = r.wrapping_add(dr);
                    if nc < cols && nr < rows {
                        let ni = nr * cols + nc;
                        if occ[ni] && !visited[ni] {
                            visited[ni] = true;
                            queue.push_back((nc, nr));
                        }
                    }
                }
            }
            components.push(comp);
        }
    }
    components
}

// ─────────────────────────────────────────────────────────────────────────────
// Convex hull (Andrew's monotone chain)
// ─────────────────────────────────────────────────────────────────────────────

fn convex_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64)> = points.to_vec();
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(a.1.partial_cmp(&b.1).unwrap())
    });
    pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-10 && (a.1 - b.1).abs() < 1e-10);

    if pts.len() < 2 {
        return pts;
    }

    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| -> f64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };

    let mut lower: Vec<(f64, f64)> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2
            && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0
        {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<(f64, f64)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2
            && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0
        {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimum-area bounding rectangle (rotating calipers on convex hull)
// ─────────────────────────────────────────────────────────────────────────────

/// Returns (corners[4], axis_u, axis_v) where u/v are the rectangle's unit axes.
fn min_area_bounding_rect(points: &[(f64, f64)]) -> [Point2<f64>; 4] {
    let hull = convex_hull(points);

    if hull.is_empty() {
        return [Point2::new(0.0, 0.0); 4];
    }
    if hull.len() == 1 {
        let p = hull[0];
        return [Point2::new(p.0, p.1); 4];
    }
    if hull.len() == 2 {
        let (ax, ay) = hull[0];
        let (bx, by) = hull[1];
        return [
            Point2::new(ax, ay),
            Point2::new(bx, by),
            Point2::new(bx, by),
            Point2::new(ax, ay),
        ];
    }

    let n = hull.len();
    let mut best_area = f64::INFINITY;
    let mut best_corners = [Point2::new(0.0, 0.0); 4];

    for i in 0..n {
        let j = (i + 1) % n;
        let ex = hull[j].0 - hull[i].0;
        let ey = hull[j].1 - hull[i].1;
        let len = (ex * ex + ey * ey).sqrt();
        if len < 1e-10 {
            continue;
        }
        let ux = ex / len;
        let uy = ey / len;
        let vx = -uy;
        let vy = ux;

        let mut min_u = f64::INFINITY;
        let mut max_u = f64::NEG_INFINITY;
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for &(x, y) in &hull {
            let u = x * ux + y * uy;
            let v = x * vx + y * vy;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }

        let area = (max_u - min_u) * (max_v - min_v);
        if area < best_area {
            best_area = area;
            let corner =
                |pu: f64, pv: f64| Point2::new(pu * ux + pv * vx, pu * uy + pv * vy);
            best_corners = [
                corner(min_u, min_v),
                corner(max_u, min_v),
                corner(max_u, max_v),
                corner(min_u, max_v),
            ];
        }
    }

    best_corners
}

// ─────────────────────────────────────────────────────────────────────────────
// GDAL geometry helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_rect_geom(corners: &[Point2<f64>; 4]) -> Result<Geometry> {
    let mut ring = Geometry::empty(wkbLinearRing).context("ring")?;
    for p in corners {
        ring.add_point_2d((p.x, p.y));
    }
    ring.add_point_2d((corners[0].x, corners[0].y)); // close
    let mut poly = Geometry::empty(wkbPolygon).context("poly")?;
    poly.add_geometry(ring).context("add ring")?;
    Ok(poly)
}

/// Flatten a GDAL geometry into a list of Polygon WKTs, regardless of whether
/// the input is a Polygon, a Polygon-with-holes, or a MultiPolygon.
fn collect_polygon_wkts(geom: &Geometry) -> Vec<String> {
    let gt = geom.geometry_type();
    if gt == wkbPolygon {
        return vec![geom.wkt().unwrap_or_default()];
    }
    if gt == wkbMultiPolygon {
        let n = geom.geometry_count();
        return (0..n)
            .map(|i| geom.get_geometry(i).wkt().unwrap_or_default())
            .collect();
    }
    // Anything else (GeometryCollection, etc.): try sub-geometries recursively.
    let n = geom.geometry_count();
    (0..n)
        .flat_map(|i| collect_polygon_wkts(&geom.get_geometry(i)))
        .collect()
}

fn geom_from_wkt(wkt: &str) -> Result<Geometry> {
    Geometry::from_wkt(wkt).context("geometry from WKT")
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-component processing
// ─────────────────────────────────────────────────────────────────────────────

fn process_component(
    comp: &[(usize, usize)],
    grid: &Grid,
    hole_min_cells: usize,
) -> Result<Vec<String>> {
    let world_pts: Vec<(f64, f64)> =
        comp.iter().map(|&(c, r)| grid.cell_centre(c, r)).collect();

    if world_pts.len() < 3 {
        return Ok(vec![]);
    }

    let corners = min_area_bounding_rect(&world_pts);
    let mut outer = make_rect_geom(&corners)?;

    // Axis-aligned bounding box of the component (in grid coordinates).
    let min_c = comp.iter().map(|&(c, _)| c).min().unwrap();
    let max_c = comp.iter().map(|&(c, _)| c).max().unwrap();
    let min_r = comp.iter().map(|&(_, r)| r).min().unwrap();
    let max_r = comp.iter().map(|&(_, r)| r).max().unwrap();

    // Build a fast lookup set of the original building cells in this component.
    let comp_set: std::collections::HashSet<(usize, usize)> = comp.iter().cloned().collect();

    // Sub-grid dimensions (inclusive → add 1).
    let sub_cols = max_c - min_c + 1;
    let sub_rows = max_r - min_r + 1;

    // Ground-only cells: have ground points, no building points, and are not
    // one of this component's cells.
    let mut ground_only = vec![false; sub_cols * sub_rows];
    for r in min_r..=max_r {
        for c in min_c..=max_c {
            let idx = r * grid.cols + c;
            if grid.ground[idx] && !grid.build[idx] && !comp_set.contains(&(c, r)) {
                ground_only[(r - min_r) * sub_cols + (c - min_c)] = true;
            }
        }
    }

    let hole_comps = connected_components(&ground_only, sub_cols, sub_rows);

    for hole_comp in hole_comps {
        if hole_comp.len() < hole_min_cells {
            continue;
        }

        // Map sub-grid coords back to world coords.
        let hole_world: Vec<(f64, f64)> = hole_comp
            .iter()
            .map(|&(sc, sr)| grid.cell_centre(sc + min_c, sr + min_r))
            .collect();

        let hole_corners = min_area_bounding_rect(&hole_world);
        let hole_geom = make_rect_geom(&hole_corners)?;

        // Subtract the hole from the outer polygon.  GDAL may return a
        // Polygon (with interior ring) or a MultiPolygon.
        match outer.difference(&hole_geom) {
            Some(result) => outer = result,
            None => tracing::warn!("geometry difference returned None; skipping hole"),
        }
    }

    Ok(collect_polygon_wkts(&outer))
}

// ─────────────────────────────────────────────────────────────────────────────
// GeoPackage writer
// ─────────────────────────────────────────────────────────────────────────────

fn write_gpkg(path: &std::path::Path, wkts: &[String], epsg: u32) -> Result<()> {
    let driver =
        DriverManager::get_driver_by_name("GPKG").context("GDAL GPKG driver not available")?;

    let mut ds = driver
        .create_vector_only(path.to_str().context("non-UTF-8 path")?)
        .context("creating GeoPackage")?;

    let srs = SpatialRef::from_epsg(epsg).context("unknown EPSG code")?;

    let layer_opts = LayerOptions {
        name: "footprints",
        srs: Some(&srs),
        ty: wkbPolygon,
        options: None,
    };
    let mut layer = ds.create_layer(layer_opts).context("creating layer")?;
    layer
        .create_defn_fields(&[("fid", OGRFieldType::OFTInteger64 as u32)])
        .context("adding fid field")?;

    for (i, wkt) in wkts.iter().enumerate() {
        if wkt.is_empty() {
            continue;
        }
        let geom = geom_from_wkt(wkt)?;
        layer
            .create_feature_fields(
                geom,
                &["fid"],
                &[FieldValue::Integer64Value(i as i64)],
            )
            .context("writing feature")?;
    }

    info!("Wrote {} footprint(s) to {}", wkts.len(), path.display());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// CRS from LAS header
// ─────────────────────────────────────────────────────────────────────────────

fn epsg_from_las(path: &std::path::Path) -> Option<u32> {
    let reader = Reader::from_path(path).ok()?;
    let header = reader.header();

    if let Ok(Some(geotiff)) = header.get_geotiff_crs() {
        for entry in &geotiff.entries {
            if entry.id == 2048 || entry.id == 3072 {
                if let las::crs::GeoTiffData::U16(code) = &entry.data {
                    return Some(*code as u32);
                }
            }
        }
    }

    if let Some(wkt_bytes) = header.get_wkt_crs_bytes() {
        let wkt = String::from_utf8_lossy(wkt_bytes);
        let marker = "AUTHORITY[\"EPSG\",\"";
        if let Some(start) = wkt.find(marker) {
            let val_start = start + marker.len();
            if let Some(end_off) = wkt[val_start..].find('"') {
                let code = &wkt[val_start..val_start + end_off];
                if let Ok(n) = code.parse::<u32>() {
                    return Some(n);
                }
            }
        }
    }

    None
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    let output_dir = match &cli.output_dir {
        Some(d) => d.clone(),
        None => cli
            .input
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from(".")),
    };
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("creating {}", output_dir.display()))?;

    let gpkg_path = output_dir.join(format!("{}.gpkg", cli.output_name));

    let epsg = match cli.epsg {
        Some(e) => e,
        None => match epsg_from_las(&cli.input) {
            Some(e) => {
                info!("CRS from LAS header: EPSG:{e}");
                e
            }
            None => bail!(
                "No CRS in LAS header and --epsg not provided. \
                 Please supply --epsg <code>."
            ),
        },
    };

    // ── 1. Read points ────────────────────────────────────────────────────────
    info!("Reading {} …", cli.input.display());
    let mut reader = Reader::from_path(&cli.input)
        .with_context(|| format!("opening {}", cli.input.display()))?;

    let mut build_pts: Vec<(f64, f64)> = Vec::new();
    let mut ground_pts: Vec<(f64, f64)> = Vec::new();
    let mut total = 0usize;

    let build_class = las::point::Classification::new(CLASS_BUILDING)
        .unwrap_or_default();
    let ground_class = las::point::Classification::new(CLASS_GROUND)
        .unwrap_or_default();

    for wrapped in reader.points() {
        let pt = wrapped?;
        total += 1;
        if pt.classification == build_class {
            build_pts.push((pt.x, pt.y));
        } else if pt.classification == ground_class {
            ground_pts.push((pt.x, pt.y));
        }
    }
    info!(
        "{total} points total — {} building, {} ground",
        build_pts.len(),
        ground_pts.len()
    );

    if build_pts.is_empty() {
        bail!("No class-{CLASS_BUILDING} (Building) points in {}", cli.input.display());
    }

    // ── 2. Build dual-class occupancy grid ───────────────────────────────────
    info!("Building {}m grid …", cli.resolution);
    let grid = Grid::new(&build_pts, &ground_pts, cli.resolution);
    info!("Grid: {} × {} cells", grid.cols, grid.rows);

    // ── 3. Morphological closing on building layer ───────────────────────────
    let closed = if cli.gap_fill > 0 {
        info!("Morphological closing (radius {} cells = {}m) …",
              cli.gap_fill, cli.gap_fill as f64 * cli.resolution);
        morph_close(&grid.build, grid.cols, grid.rows, cli.gap_fill)
    } else {
        grid.build.clone()
    };

    // ── 4. Connected components on closed grid ────────────────────────────────
    let all_comps = connected_components(&closed, grid.cols, grid.rows);
    info!("{} raw components", all_comps.len());

    let large: Vec<_> = all_comps
        .into_iter()
        .filter(|c| c.len() >= cli.min_cells)
        .collect();
    info!("{} components with ≥ {} cells", large.len(), cli.min_cells);

    if large.is_empty() {
        bail!("No components survived --min-cells filter; try lowering --min-cells");
    }

    // ── 5. Per-component: MABR + ground-hole subtraction ─────────────────────
    let mut all_wkts: Vec<String> = Vec::new();
    for comp in &large {
        let wkts = process_component(comp, &grid, cli.hole_min_cells)?;
        all_wkts.extend(wkts);
    }
    info!("{} polygon(s) after hole subtraction", all_wkts.len());

    // ── 6. Write GeoPackage ───────────────────────────────────────────────────
    write_gpkg(&gpkg_path, &all_wkts, epsg)?;

    Ok(())
}
