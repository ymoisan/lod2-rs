use crate::cityjson::{CityJsonTransform, CityJsonWriter};
use crate::hints::BuildingHint;
#[cfg(feature = "io")]
use crate::las_reader::LasReader;
use crate::mesh::BuildingGeometry;
use crate::plane::Plane;
use crate::point_cloud::PointCloud;
use crate::polygon::Footprint;
#[cfg(feature = "io")]
use crate::vector_reader::VectorReader;
use clap::Parser;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Parser, Debug, Clone)]
pub struct PipelineArgs {
    #[arg(long)]
    pub footprints: PathBuf,

    /// LAS/LAZ point cloud. Must contain classified points:
    /// class 6 (Building) for roof reconstruction, class 2 (Ground) for ground
    /// height estimation. Other classes are ignored. Classification accuracy
    /// directly determines reconstruction quality — garbage in, garbage out.
    #[arg(long)]
    pub pointcloud: PathBuf,

    #[arg(long)]
    pub output: PathBuf,

    /// NDVI threshold for spectral vegetation filtering.
    /// Class-6 points with NDVI above this value are excluded from roof
    /// reconstruction (they are likely tree canopy misclassified as building).
    /// Requires input in LAS format 7/8 with RGB+NIR. Typical value: 0.3.
    /// Disabled when not specified.
    #[arg(long)]
    pub ndvi_threshold: Option<f64>,

    /// Optional comma-separated list of footprint fids to keep (others discarded
    /// before reconstruction). Useful for benchmarking a fixed bench set.
    /// Example: --fids 3579,10201,4091
    #[arg(long, value_delimiter = ',')]
    pub fids: Vec<String>,
}

impl PipelineArgs {
    /// Filter `footprints` in place to keep only fids listed in `--fids`.
    /// No-op when `--fids` is empty.
    pub fn apply_fid_filter(&self, footprints: &mut Vec<Footprint>) {
        if self.fids.is_empty() {
            return;
        }
        let keep: std::collections::HashSet<&str> =
            self.fids.iter().map(|s| s.as_str()).collect();
        let before = footprints.len();
        footprints.retain(|fp| keep.contains(fp.id.as_str()));
        tracing::info!(
            "--fids filter: kept {}/{} footprints ({} ids requested)",
            footprints.len(),
            before,
            self.fids.len()
        );
    }
}

pub trait Reconstructor: Send + Sync {
    fn name(&self) -> &str;
    fn reconstruct(
        &self,
        footprint: &Footprint,
        points: &PointCloud,
        h_ground: f64,
    ) -> BuildingGeometry;
}

/// Crop points within a footprint's bounding box + polygon test.
/// When a Z ceiling is provided (from building height hint), points above it are
/// discarded to eliminate vegetation/noise before plane detection.
pub fn crop_points(pc: &PointCloud, footprint: &Footprint, z_ceiling: Option<f64>) -> PointCloud {
    let bbox = footprint.polygon.bbox_2d();
    let margin = 2.0;
    let mut cropped = PointCloud::new();
    for i in 0..pc.len() {
        let p = &pc.positions[i];
        if p.x >= bbox[0] - margin
            && p.x <= bbox[2] + margin
            && p.y >= bbox[1] - margin
            && p.y <= bbox[3] + margin
            && footprint.contains_2d(p.x, p.y)
        {
            if let Some(ceil) = z_ceiling {
                if p.z > ceil {
                    continue;
                }
            }
            if let Some(ref ndvi) = pc.ndvi {
                cropped.push_with_ndvi(*p, pc.classifications[i], ndvi[i]);
            } else {
                cropped.push(*p);
            }
        }
    }
    cropped
}

/// Crop points by classification: returns (building_points, ground_points).
/// - building_points: class 6, inside footprint polygon
/// - ground_points: class 2, inside footprint bbox + ground_buffer (may be just outside footprint)
pub fn crop_points_classified(
    pc: &PointCloud,
    footprint: &Footprint,
    ground_buffer: f64,
) -> (PointCloud, PointCloud) {
    let bbox = footprint.polygon.bbox_2d();
    let margin = 2.0;
    let mut building = PointCloud::new();
    let mut ground = PointCloud::new();

    for i in 0..pc.len() {
        let p = &pc.positions[i];
        let class = pc.classifications[i];

        match class {
            6 => {
                // Building points: inside footprint polygon
                if p.x >= bbox[0] - margin
                    && p.x <= bbox[2] + margin
                    && p.y >= bbox[1] - margin
                    && p.y <= bbox[3] + margin
                    && footprint.contains_2d(p.x, p.y)
                {
                    if let Some(ref ndvi) = pc.ndvi {
                        building.push_with_ndvi(*p, class, ndvi[i]);
                    } else {
                        building.push_classified(*p, class);
                    }
                }
            }
            2 => {
                // Ground points: wider buffer around footprint bbox
                if p.x >= bbox[0] - ground_buffer
                    && p.x <= bbox[2] + ground_buffer
                    && p.y >= bbox[1] - ground_buffer
                    && p.y <= bbox[3] + ground_buffer
                {
                    ground.push_classified(*p, class);
                }
            }
            _ => {} // All other classes ignored
        }
    }

    (building, ground)
}

/// Detect courtyards from point cloud density.
///
/// Rasterizes class-6 (building) points to a grid, finds connected empty
/// regions inside the footprint, and returns them as interior ring polygons.
/// This detects real courtyards even when the source footprint has incorrect
/// or missing interior rings.
pub fn detect_courtyards(
    footprint: &Footprint,
    building_pts: &PointCloud,
    cell_size: f64,
    min_area: f64,
) -> Vec<crate::polygon::LinearRing> {
    use crate::polygon::Polygon3D;
    use std::collections::VecDeque;

    let bbox = footprint.polygon.bbox_2d();
    let xmin = bbox[0];
    let ymin = bbox[1];
    let xmax = bbox[2];
    let ymax = bbox[3];
    let nx = ((xmax - xmin) / cell_size).ceil() as usize + 1;
    let ny = ((ymax - ymin) / cell_size).ceil() as usize + 1;

    if nx < 3 || ny < 3 {
        return Vec::new();
    }

    // Rasterize class-6 points to grid
    let mut grid = vec![0u16; ny * nx];
    for p in &building_pts.positions {
        let gx = ((p.x - xmin) / cell_size) as usize;
        let gy = ((p.y - ymin) / cell_size) as usize;
        if gx < nx && gy < ny {
            let idx = gy * nx + gx;
            grid[idx] = grid[idx].saturating_add(1);
        }
    }

    // Build inside mask: which cells are inside the exterior ring
    // (ignoring existing interior rings — we're replacing them)
    let ext_only = Polygon3D::new(footprint.polygon.exterior.clone());
    let mut inside_mask = vec![false; ny * nx];
    for gy in 0..ny {
        let cy = ymin + (gy as f64 + 0.5) * cell_size;
        for gx in 0..nx {
            let cx = xmin + (gx as f64 + 0.5) * cell_size;
            if ext_only.contains_2d(cx, cy) {
                inside_mask[gy * nx + gx] = true;
            }
        }
    }

    // BFS flood fill to find connected empty regions inside footprint
    let mut labeled = vec![0u32; ny * nx];
    let mut label_id = 0u32;
    let mut regions: Vec<Vec<(usize, usize)>> = Vec::new();

    for start in 0..(ny * nx) {
        if !inside_mask[start] || grid[start] > 0 || labeled[start] > 0 {
            continue;
        }
        label_id += 1;
        let mut cells = Vec::new();
        let mut queue = VecDeque::new();
        labeled[start] = label_id;
        queue.push_back(start);

        while let Some(idx) = queue.pop_front() {
            let gy = idx / nx;
            let gx = idx % nx;
            cells.push((gx, gy));
            for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                let nx2 = gx as i32 + dx;
                let ny2 = gy as i32 + dy;
                if nx2 >= 0 && ny2 >= 0 {
                    let nx2 = nx2 as usize;
                    let ny2 = ny2 as usize;
                    if nx2 < nx && ny2 < ny {
                        let nidx = ny2 * nx + nx2;
                        if inside_mask[nidx] && grid[nidx] == 0 && labeled[nidx] == 0 {
                            labeled[nidx] = label_id;
                            queue.push_back(nidx);
                        }
                    }
                }
            }
        }

        let area = cells.len() as f64 * cell_size * cell_size;
        if area >= min_area {
            regions.push(cells);
        }
    }

    // Convert each region to a boundary polygon ring
    let mut rings = Vec::new();
    for cells in &regions {
        if let Some(ring) = grid_region_to_ring(cells, xmin, ymin, cell_size, nx, ny) {
            rings.push(ring);
        }
    }
    rings
}

/// Convert a set of grid cells to a boundary polygon ring.
/// Walks the boundary edges of the connected region to produce a closed ring.
fn grid_region_to_ring(
    cells: &[(usize, usize)],
    xmin: f64,
    ymin: f64,
    cell_size: f64,
    _nx: usize,
    _ny: usize,
) -> Option<crate::polygon::LinearRing> {
    use std::collections::{HashMap, HashSet};

    let cell_set: HashSet<(usize, usize)> = cells.iter().copied().collect();

    // Collect directed boundary edges.
    // For each cell, check 4 sides. If the neighbor is NOT in the set, that side
    // is a boundary edge. Edges are stored as directed segments (CCW around the region).
    // Cell (gx, gy) has corners:
    //   bottom-left: (gx, gy), bottom-right: (gx+1, gy)
    //   top-left: (gx, gy+1), top-right: (gx+1, gy+1)
    let mut edges: Vec<((i64, i64), (i64, i64))> = Vec::new();

    for &(gx, gy) in cells {
        let gx = gx as i64;
        let gy = gy as i64;
        // Bottom edge: neighbor at (gx, gy-1)
        if !cell_set.contains(&(gx as usize, (gy - 1) as usize)) || gy == 0 {
            edges.push(((gx, gy), (gx + 1, gy))); // left to right
        }
        // Top edge: neighbor at (gx, gy+1)
        if !cell_set.contains(&(gx as usize, (gy + 1) as usize)) {
            edges.push(((gx + 1, gy + 1), (gx, gy + 1))); // right to left
        }
        // Left edge: neighbor at (gx-1, gy)
        if !cell_set.contains(&(((gx - 1) as usize), gy as usize)) || gx == 0 {
            edges.push(((gx, gy + 1), (gx, gy))); // top to bottom
        }
        // Right edge: neighbor at (gx+1, gy)
        if !cell_set.contains(&((gx + 1) as usize, gy as usize)) {
            edges.push(((gx + 1, gy), (gx + 1, gy + 1))); // bottom to top
        }
    }

    if edges.is_empty() {
        return None;
    }

    // Build outgoing-edge map
    let mut outgoing: HashMap<(i64, i64), Vec<(i64, i64)>> = HashMap::new();
    for &(a, b) in &edges {
        outgoing.entry(a).or_default().push(b);
    }

    // Walk the longest cycle (outer boundary)
    let mut used: HashSet<((i64, i64), (i64, i64))> = HashSet::new();
    let mut best_ring: Vec<(i64, i64)> = Vec::new();

    for &(start, _) in &edges {
        if used.contains(&(start, outgoing[&start][0])) {
            continue;
        }
        let mut ring = vec![start];
        let mut curr = start;
        loop {
            let nexts = match outgoing.get(&curr) {
                Some(n) => n,
                None => break,
            };
            let next = nexts.iter().find(|&&n| !used.contains(&(curr, n)));
            let next = match next {
                Some(&n) => n,
                None => break,
            };
            used.insert((curr, next));
            if next == start {
                break;
            }
            ring.push(next);
            curr = next;
        }
        if ring.len() > best_ring.len() {
            best_ring = ring;
        }
    }

    if best_ring.len() < 3 {
        return None;
    }

    // Convert grid coordinates to world coordinates
    let vertices: Vec<nalgebra::Point3<f64>> = best_ring
        .iter()
        .map(|&(gx, gy)| {
            nalgebra::Point3::new(
                xmin + gx as f64 * cell_size,
                ymin + gy as f64 * cell_size,
                0.0,
            )
        })
        .collect();

    // Close the ring (WKB convention)
    let mut verts = vertices;
    if let (Some(first), Some(last)) = (verts.first(), verts.last()) {
        if (first.x - last.x).abs() > 1e-10 || (first.y - last.y).abs() > 1e-10 {
            verts.push(*first);
        }
    }

    Some(crate::polygon::LinearRing::from_vertices(verts))
}
#[cfg(feature = "io")]
pub fn run_pipeline(args: &PipelineArgs, reconstructor: &dyn Reconstructor) -> anyhow::Result<()> {
    tracing::info!("Reading footprints from {}", args.footprints.display());
    let footprints = VectorReader::read_footprints(&args.footprints)?;
    let crs = VectorReader::read_crs(&args.footprints)?;
    tracing::info!("Found CRS: {:?}", crs);
    tracing::info!("Read {} footprints", footprints.len());

    tracing::info!("Reading point cloud from {}", args.pointcloud.display());
    let pc = LasReader::read_file(&args.pointcloud)?;

    write_results(&args.output, &footprints, &pc, crs.as_deref(), reconstructor)
}

/// Reconstruct buildings and write CityJSONL output.
///
/// This is the I/O-independent core: it takes already-loaded footprints and
/// point cloud data, runs the reconstruction, and writes the results.
pub fn write_results(
    output_dir: &std::path::Path,
    footprints: &[Footprint],
    pc: &PointCloud,
    crs: Option<&str>,
    reconstructor: &dyn Reconstructor,
) -> anyhow::Result<()> {
    let (cx, cy) = compute_centroid(footprints);
    let transform = CityJsonTransform {
        scale: [0.001, 0.001, 0.001],
        translate: [cx, cy, 0.0],
    };

    let output_path = output_dir.join("output.city.jsonl");
    let writer = Mutex::new(
        CityJsonWriter::new(&output_path, transform)?
            .with_reference_system(crs.unwrap_or("2154")),
    );

    {
        let mut w = writer.lock().unwrap();
        w.write_header()?;
    }

    tracing::info!("Reconstructing buildings...");
    crate::plane::reset_detection_stats();
    std::panic::set_hook(Box::new(|_| {}));
    let results: Vec<BuildingGeometry> = footprints
        .par_iter()
        .map(|fp| {
            let hint = BuildingHint::from_footprint(fp);

            // First pass without Z ceiling to determine h_ground
            let raw = crop_points(pc, fp, None);
            let raw_stats = raw.compute_statistics();
            let h_ground = raw_stats.z_min;

            // Second pass: apply Z ceiling from height hint to strip vegetation/noise
            let cropped = if hint.z_ceiling(h_ground).is_some() {
                crop_points(pc, fp, hint.z_ceiling(h_ground))
            } else {
                raw
            };
            let stats = if hint.z_ceiling(h_ground).is_some() {
                cropped.compute_statistics()
            } else {
                raw_stats
            };

            // Catch panics from CDT or other geometry issues
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                reconstructor.reconstruct(fp, &cropped, h_ground)
            }));

            match result {
                Ok(geom) => geom,
                Err(_) => {
                    tracing::warn!("Reconstruction panicked for {}, using flat roof fallback", fp.id);
                    let mut geom = BuildingGeometry::new(&fp.id);
                    geom.attributes = fp.attributes.clone();
                    geom.h_ground = h_ground;
                    let h_roof = hint.best_roof_height(stats.z_70p, h_ground);
                    geom.lod22 = build_flat_roof(&fp, h_ground, h_roof);
                    geom.roof_reason = Some(crate::mesh::RoofReason::FallbackCdtPanic);
                    geom
                }
            }
        })
        .collect();

    let mut n_reconstructed = 0usize;
    let mut n_reconstructed_mixture = 0usize;
    let mut n_flat_attr = 0usize;
    let mut n_sloped = 0usize;
    let mut n_fallback_planes = 0usize;
    let mut n_fallback_panic = 0usize;
    let mut n_unknown = 0usize;
    {
        let mut w = writer.lock().unwrap();
        for geom in &results {
            match &geom.roof_reason {
                Some(crate::mesh::RoofReason::Reconstructed) => n_reconstructed += 1,
                Some(crate::mesh::RoofReason::ReconstructedMixture) => n_reconstructed_mixture += 1,
                Some(crate::mesh::RoofReason::FlatByAttribute) => n_flat_attr += 1,
                Some(crate::mesh::RoofReason::SlopedExtrusion) => n_sloped += 1,
                Some(crate::mesh::RoofReason::FallbackNoPlanes) => n_fallback_planes += 1,
                Some(crate::mesh::RoofReason::FallbackCdtPanic) => n_fallback_panic += 1,
                None => n_unknown += 1,
            }
            w.write_feature(geom)?;
        }
    }

    writer.into_inner().unwrap().finish()?;
    let total = results.len();
    tracing::info!(
        "Reconstruction complete: {total} buildings — \
         {n_reconstructed} reconstructed, {n_reconstructed_mixture} reconstructed_mixture, \
         {n_flat_attr} flat_by_attribute, {n_sloped} sloped_extrusion, \
         {n_fallback_planes} fallback_no_planes, {n_fallback_panic} fallback_cdt_panic, \
         {n_unknown} untagged"
    );
    let (rg, ransac) = crate::plane::detection_stats();
    if rg + ransac > 0 {
        let rg_pct = 100.0 * rg as f64 / (rg + ransac) as f64;
        tracing::info!(
            "Plane detection: {rg} region growing ({rg_pct:.1}%), {ransac} RANSAC fallback"
        );
    }
    // Aggregate point-cloud validation metrics
    let mut rmse_sum = 0.0_f64;
    let mut coverage_sum = 0.0_f64;
    let mut n_validated = 0usize;
    for geom in &results {
        if let Some(crate::polygon::AttributeValue::Float(rmse)) = geom.attributes.0.get("rf_rmse_lod22") {
            if let Some(crate::polygon::AttributeValue::Float(cov)) = geom.attributes.0.get("rf_coverage") {
                rmse_sum += rmse;
                coverage_sum += cov;
                n_validated += 1;
            }
        }
    }
    if n_validated > 0 {
        let avg_rmse = rmse_sum / n_validated as f64;
        let avg_coverage = 100.0 * coverage_sum / n_validated as f64;
        tracing::info!(
            "Quality ({n_validated} buildings with planes): avg RMSE {avg_rmse:.3}m, avg coverage {avg_coverage:.1}%"
        );
    }

    Ok(())
}

/// Classification-aware variant of `write_results`.
/// Uses class 6 (building) for reconstruction and class 2 (ground) for h_ground.
pub fn write_results_classified(
    output_dir: &std::path::Path,
    footprints: &[Footprint],
    pc: &PointCloud,
    crs: Option<&str>,
    reconstructor: &dyn Reconstructor,
    ndvi_threshold: Option<f64>,
) -> anyhow::Result<()> {
    let (cx, cy) = compute_centroid(footprints);
    let transform = CityJsonTransform {
        scale: [0.001, 0.001, 0.001],
        translate: [cx, cy, 0.0],
    };

    let output_path = output_dir.join("output.city.jsonl");
    let writer = Mutex::new(
        CityJsonWriter::new(&output_path, transform)?
            .with_reference_system(crs.unwrap_or("2154")),
    );

    {
        let mut w = writer.lock().unwrap();
        w.write_header()?;
    }

    tracing::info!("Reconstructing buildings (classification-aware)...");
    crate::plane::reset_detection_stats();
    // Suppress panic output — panics in kiddo/spade are caught and handled gracefully.
    std::panic::set_hook(Box::new(|_| {}));
    let progress_counter = AtomicUsize::new(0);
    let total_buildings = footprints.len();
    let start_time = Instant::now();
    let last_log_secs = AtomicUsize::new(0);
    let results: Vec<BuildingGeometry> = footprints
        .par_iter()
        .map(|fp| {
            let done = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
            let elapsed_secs = start_time.elapsed().as_secs() as usize;
            let prev = last_log_secs.load(Ordering::Relaxed);
            if done == total_buildings || (elapsed_secs >= prev + 30 && last_log_secs.compare_exchange(prev, elapsed_secs, Ordering::Relaxed, Ordering::Relaxed).is_ok()) {
                tracing::info!("  {}/{} buildings processed ({:.0}s)", done, total_buildings, start_time.elapsed().as_secs_f64());
            }
            let hint = BuildingHint::from_footprint(fp);

            // Classification-aware crop: class 6 for building, class 2 for ground
            let (building_pts_raw, ground_pts) = crop_points_classified(pc, fp, 3.0);

            // Spectral pre-filtering (Level 1): remove vegetation points from building set
            let (building_pts, n_spectral_removed) = if let Some(threshold) = ndvi_threshold {
                if building_pts_raw.has_spectral() {
                    building_pts_raw.filter_vegetation(threshold as f32)
                } else {
                    (building_pts_raw, 0)
                }
            } else {
                (building_pts_raw, 0)
            };

            // Trust the GPKG footprint as-is (matches Roofer behavior).
            // Inferring courtyards from LiDAR density gaps is unsafe when
            // coverage is sparse: large empty regions get stamped as phantom
            // holes that subdivide the arrangement and corrupt reconstruction.
            let fp = fp.clone();

            // Footprint agreement: IoU between supplied footprint and lidar-derived footprint.
            // Only computed when derived area >= supplied (enough lidar coverage).
            let footprint_iou = compute_footprint_iou(&fp, &building_pts, 0.5);

            // Derived footprint for LoD0 visualization
            let derived_fp = compute_derived_footprint(&building_pts, 1.0);

            // h_ground from class 2 (ground) 10th percentile — matches roofer-c behavior
            // Using a low percentile avoids bias from ground points at varying terrain levels
            let h_ground = if !ground_pts.is_empty() {
                let stats = ground_pts.compute_statistics();
                stats.z_10p
            } else if !building_pts.is_empty() {
                building_pts.compute_statistics().z_min
            } else {
                0.0
            };

            let stats = building_pts.compute_statistics();

            // Catch panics from CDT or other geometry issues
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                reconstructor.reconstruct(&fp, &building_pts, h_ground)
            }));

            let mut geom = match result {
                Ok(geom) => geom,
                Err(_) => {
                    tracing::warn!("Reconstruction panicked for {}, using flat roof fallback", fp.id);
                    let mut geom = BuildingGeometry::new(&fp.id);
                    geom.attributes = fp.attributes.clone();
                    geom.h_ground = h_ground;
                    let h_roof = hint.best_roof_height(stats.z_70p, h_ground);
                    geom.lod22 = build_flat_roof(&fp, h_ground, h_roof);
                    geom.roof_reason = Some(crate::mesh::RoofReason::FallbackCdtPanic);
                    geom
                }
            };

            // Footprint agreement attribute
            if let Some(iou) = footprint_iou {
                geom.attributes.insert_float("footprint_iou", (iou * 1000.0).round() / 1000.0);
            }

            // Spectral filtering diagnostics
            if ndvi_threshold.is_some() && n_spectral_removed > 0 {
                geom.attributes.insert_int("n_spectral_removed", n_spectral_removed as i64);
            }
            if let Some(ref ndvi_vec) = building_pts.ndvi {
                let valid_ndvi: Vec<f32> = ndvi_vec.iter().copied().filter(|v| v.is_finite()).collect();
                if !valid_ndvi.is_empty() {
                    let mean_ndvi: f32 = valid_ndvi.iter().sum::<f32>() / valid_ndvi.len() as f32;
                    geom.attributes.insert_float("mean_ndvi", (mean_ndvi as f64 * 1000.0).round() / 1000.0);
                }
            }

            // LoD0: derived footprint from lidar points
            if let Some(ref dfp) = derived_fp {
                geom.lod0 = build_lod0(dfp, h_ground);
            }

            // Roof offset diagnostics: compute per-point dz and dist_to_edge,
            // aggregate into summary attributes for roof typology inference.
            let z_70p = stats.z_70p;
            let n_pts = building_pts.len();
            geom.attributes.insert_float("h_roof_z70p", z_70p);
            geom.attributes.insert_int("n_points", n_pts as i64);

            if n_pts >= 3 {
                let mut dz_sum = 0.0_f64;
                let mut dz_min = f64::MAX;
                let mut dz_max = f64::MIN;
                let mut de_sum = 0.0_f64;
                let mut dz_de_sum = 0.0_f64;
                let mut dz_sq_sum = 0.0_f64;
                let mut de_sq_sum = 0.0_f64;

                for p in &building_pts.positions {
                    let dz = p.z - z_70p;
                    let de = fp.min_distance_to_boundary(p.x, p.y);
                    dz_sum += dz;
                    dz_sq_sum += dz * dz;
                    de_sum += de;
                    de_sq_sum += de * de;
                    dz_de_sum += dz * de;
                    if dz < dz_min { dz_min = dz; }
                    if dz > dz_max { dz_max = dz; }
                }

                let n = n_pts as f64;
                let dz_mean = dz_sum / n;
                let dz_var = (dz_sq_sum / n) - (dz_mean * dz_mean);
                let dz_std = if dz_var > 0.0 { dz_var.sqrt() } else { 0.0 };

                // Pearson correlation between dist_to_edge and dz
                let de_mean = de_sum / n;
                let de_var = (de_sq_sum / n) - (de_mean * de_mean);
                let cov = (dz_de_sum / n) - (dz_mean * de_mean);
                let denom = (dz_var * de_var).sqrt();
                let correlation = if denom > 1e-12 { cov / denom } else { 0.0 };

                geom.attributes.insert_float("dz_mean", (dz_mean * 100.0).round() / 100.0);
                geom.attributes.insert_float("dz_std", (dz_std * 100.0).round() / 100.0);
                geom.attributes.insert_float("dz_min", (dz_min * 100.0).round() / 100.0);
                geom.attributes.insert_float("dz_max", (dz_max * 100.0).round() / 100.0);
                geom.attributes.insert_float("dz_edge_correlation", (correlation * 1000.0).round() / 1000.0);

                let inferred = if dz_std < 1.0 {
                    "flat"
                } else if dz_std < 2.5 {
                    "slanted"
                } else {
                    "complex"
                };
                geom.attributes.insert_string("inferred_roof_class", inferred);
            } else {
                geom.attributes.insert_string("inferred_roof_class", "unknown");
            }

            geom
        })
        .collect();

    let mut n_reconstructed = 0usize;
    let mut n_reconstructed_mixture = 0usize;
    let mut n_flat_attr = 0usize;
    let mut n_sloped = 0usize;
    let mut n_fallback_planes = 0usize;
    let mut n_fallback_panic = 0usize;
    let mut n_unknown = 0usize;
    let mut n_with_enough_points = 0usize;
    tracing::info!("Writing CityJSON output ({} buildings)...", results.len());
    {
        let mut w = writer.lock().unwrap();
        for geom in &results {
            match &geom.roof_reason {
                Some(crate::mesh::RoofReason::Reconstructed) => n_reconstructed += 1,
                Some(crate::mesh::RoofReason::ReconstructedMixture) => n_reconstructed_mixture += 1,
                Some(crate::mesh::RoofReason::FlatByAttribute) => n_flat_attr += 1,
                Some(crate::mesh::RoofReason::SlopedExtrusion) => n_sloped += 1,
                Some(crate::mesh::RoofReason::FallbackNoPlanes) => n_fallback_planes += 1,
                Some(crate::mesh::RoofReason::FallbackCdtPanic) => n_fallback_panic += 1,
                None => n_unknown += 1,
            }
            if let Some(crate::polygon::AttributeValue::Int(np)) = geom.attributes.0.get("n_points") {
                if *np >= 15 { n_with_enough_points += 1; }
            }
            w.write_feature(geom)?;
        }
    }

    writer.into_inner().unwrap().finish()?;
    let total = results.len();

    // Aggregate point-cloud validation metrics
    let mut rmse_sum = 0.0_f64;
    let mut coverage_sum = 0.0_f64;
    let mut n_validated = 0usize;
    for geom in &results {
        if let Some(crate::polygon::AttributeValue::Float(rmse)) = geom.attributes.0.get("rf_rmse_lod22") {
            if let Some(crate::polygon::AttributeValue::Float(cov)) = geom.attributes.0.get("rf_coverage") {
                rmse_sum += rmse;
                coverage_sum += cov;
                n_validated += 1;
            }
        }
    }

    // Clear summary telling the real story
    let n_no_points = total - n_with_enough_points;
    let n_planes_but_no_mesh = if n_validated > n_reconstructed { n_validated - n_reconstructed } else { 0 };
    let n_points_but_no_planes = if n_with_enough_points > n_validated { n_with_enough_points - n_validated } else { 0 };

    tracing::info!("─── Reconstruction Summary ───");
    tracing::info!("  Total footprints: {total}  (all get 3D geometry in output)");
    tracing::info!("  ┌ {n_no_points} footprints had <15 building points → flat roof extrusion");
    tracing::info!("  ├ {n_with_enough_points} footprints had ≥15 building points → plane detection attempted");
    if n_points_but_no_planes > 0 {
        tracing::info!("  │  └ {n_points_but_no_planes} had no planes detected → flat roof extrusion");
    }
    tracing::info!("  ├ {n_validated} had roof planes detected");
    if n_planes_but_no_mesh > 0 {
        tracing::info!("  │  └ {n_planes_but_no_mesh} had planes but meshing failed → flat roof extrusion");
    }
    tracing::info!("  └ {n_reconstructed} successfully reconstructed with multi-plane roof");
    if n_validated > 0 {
        let avg_rmse = rmse_sum / n_validated as f64;
        let avg_coverage = 100.0 * coverage_sum / n_validated as f64;
        let success_rate = 100.0 * n_reconstructed as f64 / n_validated as f64;
        tracing::info!(
            "  Meshing success rate: {n_reconstructed}/{n_validated} = {success_rate:.0}%"
        );
        tracing::info!(
            "  Point-cloud fit: avg RMSE {avg_rmse:.3}m, avg coverage {avg_coverage:.1}%"
        );
    }
    if n_reconstructed_mixture > 0 || n_sloped > 0 || n_fallback_panic > 0 || n_unknown > 0 {
        tracing::info!(
            "  Other: {n_reconstructed_mixture} mixture, {n_sloped} sloped, \
             {n_fallback_panic} panic fallback, {n_unknown} untagged"
        );
    }
    let (rg, ransac) = crate::plane::detection_stats();
    if rg + ransac > 0 {
        let rg_pct = 100.0 * rg as f64 / (rg + ransac) as f64;
        tracing::info!(
            "  Plane detection: {rg} region growing ({rg_pct:.1}%), {ransac} RANSAC fallback"
        );
    }
    tracing::info!("──────────────────────────────");

    Ok(())
}

pub fn compute_centroid(footprints: &[Footprint]) -> (f64, f64) {
    if footprints.is_empty() {
        return (0.0, 0.0);
    }
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut n = 0.0;
    for fp in footprints {
        let (x, y) = fp.polygon.centroid_2d();
        cx += x;
        cy += y;
        n += 1.0;
    }
    (cx / n, cy / n)
}

/// Build a mesh by extruding the footprint with variable-height roof vertices.
/// The `roof_z` closure maps each footprint vertex (x, y) to a roof Z value.
/// Watertight by construction: ground (CW), walls (CCW quads), roof (CCW).
/// Supports interior rings (courtyards).
pub fn build_variable_roof(
    footprint: &Footprint,
    h_ground: f64,
    roof_z: impl Fn(f64, f64) -> f64,
) -> Option<crate::mesh::Mesh> {
    use crate::mesh::{Face, Mesh, SemanticSurface};

    let exterior = &footprint.polygon.exterior;
    let n = exterior.len().saturating_sub(1);
    if n < 3 {
        return None;
    }

    let mut mesh = Mesh::new();
    let ground_idx = mesh.add_semantic(SemanticSurface::ground());
    let wall_idx = mesh.add_semantic(SemanticSurface::wall(true));
    let roof_idx = mesh.add_semantic(SemanticSurface::roof());

    // Exterior ring vertices
    let mut bottom = Vec::with_capacity(n);
    let mut top = Vec::with_capacity(n);
    for i in 0..n {
        let v = &exterior.vertices[i];
        bottom.push(mesh.add_vertex(nalgebra::Point3::new(v.x, v.y, h_ground)));
        top.push(mesh.add_vertex(nalgebra::Point3::new(v.x, v.y, roof_z(v.x, v.y))));
    }

    // Ground face with courtyard holes
    let mut ground_face = Face::new(bottom.iter().rev().copied().collect()).with_semantic(ground_idx);
    // Roof face with courtyard holes
    let mut roof_face = Face::new(top.clone()).with_semantic(roof_idx);

    // Interior rings (courtyards)
    for hole in &footprint.polygon.interiors {
        let hn = hole.len().saturating_sub(1);
        if hn < 3 { continue; }
        let mut hole_bottom = Vec::with_capacity(hn);
        let mut hole_top = Vec::with_capacity(hn);
        for i in 0..hn {
            let v = &hole.vertices[i];
            hole_bottom.push(mesh.add_vertex(nalgebra::Point3::new(v.x, v.y, h_ground)));
            hole_top.push(mesh.add_vertex(nalgebra::Point3::new(v.x, v.y, roof_z(v.x, v.y))));
        }
        ground_face.holes.push(hole_bottom.clone());
        roof_face.holes.push(hole_top.clone());
        // Courtyard walls (face inward — reversed winding)
        for i in 0..hn {
            let j = (i + 1) % hn;
            mesh.add_face(
                Face::new(vec![hole_bottom[j], hole_bottom[i], hole_top[i], hole_top[j]])
                    .with_semantic(wall_idx),
            );
        }
    }

    mesh.add_face(ground_face);
    mesh.add_face(roof_face);

    // Exterior walls
    for i in 0..n {
        let j = (i + 1) % n;
        mesh.add_face(
            Face::new(vec![bottom[i], bottom[j], top[j], top[i]]).with_semantic(wall_idx),
        );
    }

    Some(mesh)
}

/// Helper to build a flat-roof mesh (constant height).
pub fn build_flat_roof(
    footprint: &Footprint,
    h_ground: f64,
    h_roof: f64,
) -> Option<crate::mesh::Mesh> {
    build_variable_roof(footprint, h_ground, |_, _| h_roof)
}

/// Build a sloped-roof mesh using detected plane(s).
/// For 1 plane: z_roof = plane.eval_z(x, y).
/// For 2 planes (gable): z_roof = max(plane0, plane1) — upper envelope.
pub fn build_sloped_roof(
    footprint: &Footprint,
    planes: &[Plane],
    h_ground: f64,
    z_95p: f64,
) -> Option<crate::mesh::Mesh> {
    if planes.is_empty() {
        return None;
    }
    let z_min_wall = h_ground + 1.0;
    let z_max = z_95p;
    let fallback = h_ground + 3.0;
    build_variable_roof(footprint, h_ground, |x, y| {
        let z = if planes.len() == 1 {
            planes[0].eval_z(x, y).unwrap_or(fallback)
        } else {
            let z0 = planes[0].eval_z(x, y).unwrap_or(fallback);
            let z1 = planes[1].eval_z(x, y).unwrap_or(fallback);
            z0.max(z1)
        };
        z.clamp(z_min_wall, z_max)
    })
}

/// Compute IoU (Intersection over Union) between the supplied footprint polygon
/// and the "derived" footprint from rasterized building points.
/// Returns None if the derived area is smaller than the supplied area
/// (meaning sparse lidar coverage — comparison would be meaningless).
pub fn compute_footprint_iou(
    footprint: &Footprint,
    building_pts: &PointCloud,
    cell_size: f64,
) -> Option<f64> {
    if building_pts.len() < 3 {
        return None;
    }

    let fp_bbox = footprint.polygon.bbox_2d();
    // Combined bbox of footprint + building points
    let mut xmin = fp_bbox[0];
    let mut ymin = fp_bbox[1];
    let mut xmax = fp_bbox[2];
    let mut ymax = fp_bbox[3];
    for p in &building_pts.positions {
        xmin = xmin.min(p.x);
        ymin = ymin.min(p.y);
        xmax = xmax.max(p.x);
        ymax = ymax.max(p.y);
    }

    let nx = ((xmax - xmin) / cell_size).ceil() as usize + 1;
    let ny = ((ymax - ymin) / cell_size).ceil() as usize + 1;
    if nx < 2 || ny < 2 {
        return None;
    }

    // Grid B (derived): occupied cells from building points
    let mut derived = vec![false; ny * nx];
    for p in &building_pts.positions {
        let gx = ((p.x - xmin) / cell_size) as usize;
        let gy = ((p.y - ymin) / cell_size) as usize;
        if gx < nx && gy < ny {
            derived[gy * nx + gx] = true;
        }
    }

    // Grid A (supplied): footprint polygon rasterized via contains_2d
    let mut supplied = vec![false; ny * nx];
    for gy in 0..ny {
        let cy = ymin + (gy as f64 + 0.5) * cell_size;
        for gx in 0..nx {
            let cx = xmin + (gx as f64 + 0.5) * cell_size;
            if footprint.contains_2d(cx, cy) {
                supplied[gy * nx + gx] = true;
            }
        }
    }

    let n_supplied = supplied.iter().filter(|&&v| v).count();
    let n_derived = derived.iter().filter(|&&v| v).count();

    // Skip if derived area < supplied area (sparse lidar)
    if n_derived < n_supplied {
        return None;
    }

    let mut n_intersection = 0usize;
    let mut n_union = 0usize;
    for i in 0..(ny * nx) {
        let a = supplied[i];
        let b = derived[i];
        if a || b {
            n_union += 1;
        }
        if a && b {
            n_intersection += 1;
        }
    }

    if n_union == 0 {
        return None;
    }

    Some(n_intersection as f64 / n_union as f64)
}

/// Compute the derived footprint boundary from rasterized building points.
/// Returns the boundary ring of the largest connected occupied region.
pub fn compute_derived_footprint(
    building_pts: &PointCloud,
    cell_size: f64,
) -> Option<crate::polygon::Polygon3D> {
    use std::collections::{HashSet, VecDeque};

    if building_pts.len() < 3 {
        return None;
    }

    let mut xmin = f64::MAX;
    let mut ymin = f64::MAX;
    let mut xmax = f64::MIN;
    let mut ymax = f64::MIN;
    for p in &building_pts.positions {
        xmin = xmin.min(p.x);
        ymin = ymin.min(p.y);
        xmax = xmax.max(p.x);
        ymax = ymax.max(p.y);
    }

    let nx = ((xmax - xmin) / cell_size).ceil() as usize + 1;
    let ny = ((ymax - ymin) / cell_size).ceil() as usize + 1;
    if nx < 2 || ny < 2 {
        return None;
    }

    // Rasterize building points
    let mut grid = vec![false; ny * nx];
    for p in &building_pts.positions {
        let gx = ((p.x - xmin) / cell_size) as usize;
        let gy = ((p.y - ymin) / cell_size) as usize;
        if gx < nx && gy < ny {
            grid[gy * nx + gx] = true;
        }
    }

    // BFS to find the largest connected occupied region
    let mut labeled = vec![0u32; ny * nx];
    let mut label_id = 0u32;
    let mut best_cells: Vec<(usize, usize)> = Vec::new();

    for start in 0..(ny * nx) {
        if !grid[start] || labeled[start] > 0 {
            continue;
        }
        label_id += 1;
        let mut cells = Vec::new();
        let mut queue = VecDeque::new();
        labeled[start] = label_id;
        queue.push_back(start);

        while let Some(idx) = queue.pop_front() {
            let gy = idx / nx;
            let gx = idx % nx;
            cells.push((gx, gy));
            for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                let nx2 = gx as i32 + dx;
                let ny2 = gy as i32 + dy;
                if nx2 >= 0 && ny2 >= 0 {
                    let nx2 = nx2 as usize;
                    let ny2 = ny2 as usize;
                    if nx2 < nx && ny2 < ny {
                        let nidx = ny2 * nx + nx2;
                        if grid[nidx] && labeled[nidx] == 0 {
                            labeled[nidx] = label_id;
                            queue.push_back(nidx);
                        }
                    }
                }
            }
        }

        if cells.len() > best_cells.len() {
            best_cells = cells;
        }
    }

    if best_cells.len() < 3 {
        return None;
    }

    let ring = grid_region_to_ring(&best_cells, xmin, ymin, cell_size, nx, ny)?;
    Some(crate::polygon::Polygon3D::new(ring))
}

/// Build a LoD0 mesh: flat polygon at h_ground from a derived footprint.
pub fn build_lod0(
    derived: &crate::polygon::Polygon3D,
    h_ground: f64,
) -> Option<crate::mesh::Mesh> {
    use crate::mesh::{Face, Mesh, SemanticSurface};

    let n = derived.exterior.len();
    if n < 3 {
        return None;
    }

    let mut mesh = Mesh::new();
    let ground_idx = mesh.add_semantic(SemanticSurface::ground());

    let mut indices = Vec::with_capacity(n);
    for v in &derived.exterior.vertices {
        indices.push(mesh.add_vertex(nalgebra::Point3::new(v.x, v.y, h_ground)));
    }

    mesh.add_face(Face::new(indices).with_semantic(ground_idx));
    Some(mesh)
}


pub fn compute_ridge_segments(
    footprint: &Footprint,
    planes: &[Plane],
) -> Vec<([f64; 2], [f64; 2], usize, usize)> {
    use crate::plane::{clip_line_to_bbox, intersect_planes, split_segments_at_intersections};

    let bbox = footprint.polygon.bbox_2d();

    let mut segments: Vec<([f64; 2], [f64; 2], usize, usize)> = Vec::new();
    for i in 0..planes.len() {
        for j in (i + 1)..planes.len() {
            if let Some((origin, dir)) = intersect_planes(&planes[i], &planes[j]) {
                if let Some((p1, p2)) = clip_line_to_bbox(&origin, &dir, &bbox) {
                    segments.push((p1, p2, i, j));
                }
            }
        }
    }

    let fp_edges = footprint_boundary_edges(footprint);
    let ridge_with_meta: Vec<_> = segments
        .iter()
        .map(|&(p1, p2, i, j)| (p1, p2, (i, j)))
        .collect();
    let split = split_segments_at_intersections(&fp_edges, &ridge_with_meta, 1e-3);
    // Keep only segments whose endpoints are strictly inside the footprint
    // with some margin, to avoid T-junctions with wall quads.
    let margin = 0.1; // 10cm margin from boundary
    split
        .into_iter()
        .map(|(p1, p2, (i, j))| (p1, p2, i, j))
        .filter(|(p1, p2, _, _)| {
            let d1 = footprint.min_distance_to_boundary(p1[0], p1[1]);
            let d2 = footprint.min_distance_to_boundary(p2[0], p2[1]);
            d1 > margin && d2 > margin
                && footprint.contains_2d(p1[0], p1[1])
                && footprint.contains_2d(p2[0], p2[1])
        })
        .collect()
}

fn footprint_boundary_edges(footprint: &Footprint) -> Vec<([f64; 2], [f64; 2])> {
    let ext = &footprint.polygon.exterior;
    let n = ext.len().saturating_sub(1);
    let mut edges = Vec::with_capacity(n);
    for i in 0..n {
        let a = &ext.vertices[i];
        let b = &ext.vertices[(i + 1) % n];
        edges.push(([a.x, a.y], [b.x, b.y]));
    }
    for hole in &footprint.polygon.interiors {
        let hn = hole.len().saturating_sub(1);
        for i in 0..hn {
            let a = &hole.vertices[i];
            let b = &hole.vertices[(i + 1) % hn];
            edges.push(([a.x, a.y], [b.x, b.y]));
        }
    }
    edges
}

/// Build a roof mesh using CDT-based arrangement: partition footprint into triangles
/// along detected plane intersection lines, assign each triangle to its best-fit plane,
/// and extrude walls from the footprint boundary.
#[cfg(feature = "pipeline")]
pub fn build_arrangement_roof(
    footprint: &Footprint,
    planes: &[Plane],
    segments: &[([f64; 2], [f64; 2], usize, usize)],
    points: &[nalgebra::Point3<f64>],
    h_ground: f64,
    z_95p: f64,
) -> Option<crate::mesh::Mesh> {
    use crate::mesh::{Face, Mesh, SemanticSurface};
    use spade::{ConstrainedDelaunayTriangulation, Point2 as SpadePoint2, Triangulation};

    let exterior = &footprint.polygon.exterior;
    let n = exterior.len().saturating_sub(1);
    if n < 3 {
        return None;
    }

    let mut cdt: ConstrainedDelaunayTriangulation<SpadePoint2<f64>> =
        ConstrainedDelaunayTriangulation::new();

    // Insert footprint boundary as constraints
    let mut boundary_handles = Vec::new();
    for i in 0..n {
        let v = &exterior.vertices[i];
        match cdt.insert(SpadePoint2::new(v.x, v.y)) {
            Ok(h) => boundary_handles.push(h),
            Err(_) => return None,
        }
    }
    for i in 0..boundary_handles.len() {
        let j = (i + 1) % boundary_handles.len();
        let _ = cdt.add_constraint(boundary_handles[i], boundary_handles[j]);
    }

    // Insert interior rings (courtyards) as constraints
    for hole in &footprint.polygon.interiors {
        let hn = hole.len().saturating_sub(1);
        if hn < 3 {
            continue;
        }
        let mut hole_handles = Vec::new();
        for i in 0..hn {
            let v = &hole.vertices[i];
            match cdt.insert(SpadePoint2::new(v.x, v.y)) {
                Ok(h) => hole_handles.push(h),
                Err(_) => continue,
            }
        }
        for i in 0..hole_handles.len() {
            let j = (i + 1) % hole_handles.len();
            let _ = cdt.add_constraint(hole_handles[i], hole_handles[j]);
        }
    }

    // Insert ridge segments as constraints.
    // spade panics if a constraint intersects an existing one; after a panic the
    // CDT may be inconsistent, so stop inserting further constraints.
    for (p1, p2, _, _) in segments {
        let h1 = cdt.insert(SpadePoint2::new(p1[0], p1[1]));
        let h2 = cdt.insert(SpadePoint2::new(p2[0], p2[1]));
        if let (Ok(h1), Ok(h2)) = (h1, h2) {
            if h1 != h2 {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    cdt.add_constraint(h1, h2);
                }));
                if result.is_err() {
                    break;
                }
            }
        }
    }

    let mut mesh = Mesh::new();
    let ground_idx = mesh.add_semantic(SemanticSurface::ground());
    let wall_idx = mesh.add_semantic(SemanticSurface::wall(true));

    let roof_semantics: Vec<usize> = planes
        .iter()
        .map(|p| {
            mesh.add_semantic(SemanticSurface::roof_with_stats(
                p.slope_degrees(),
                p.azimuth_degrees(),
            ))
        })
        .collect();

    // Build vertex dedup map: quantized (x,y) → mesh vertex index.
    // CDT vertices at the same 2D position get the same mesh index.
    use std::collections::HashMap;
    let mut pos_to_idx: HashMap<[i64; 2], u32> = HashMap::new();

    // Data-driven fallback height: median Z of all plane inlier points.
    let fallback_h = {
        let mut zs: Vec<f64> = planes.iter()
            .flat_map(|p| p.inliers.iter().map(|&i| points[i].z))
            .collect();
        if zs.is_empty() {
            h_ground + 3.0
        } else {
            zs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            zs[zs.len() / 2]
        }
    };

    let mut get_roof_vertex = |mesh: &mut Mesh,
                                pos_map: &mut HashMap<[i64; 2], u32>,
                                x: f64,
                                y: f64,
                                z: f64|
     -> u32 {
        let key = [(x * 1000.0) as i64, (y * 1000.0) as i64];
        if let Some(&idx) = pos_map.get(&key) {
            idx
        } else {
            let idx = mesh.add_vertex(nalgebra::Point3::new(x, y, z));
            pos_map.insert(key, idx);
            idx
        }
    };

    // Classify each CDT face: find best-fit plane and assign roof height
    for face_handle in cdt.inner_faces() {
        let [v0, v1, v2] = face_handle.vertices();
        let p0 = v0.position();
        let p1 = v1.position();
        let p2 = v2.position();

        let cx = (p0.x + p1.x + p2.x) / 3.0;
        let cy = (p0.y + p1.y + p2.y) / 3.0;

        if !footprint.contains_2d(cx, cy) {
            continue;
        }
        if footprint.polygon.is_in_hole_2d(cx, cy) {
            continue;
        }

        let best_plane = find_best_plane(planes, points, cx, cy, footprint);

        let (z0, z1, z2, sem_idx) = if let Some(pi) = best_plane {
            let plane = &planes[pi];
            if plane.slope_degrees() < 2.0 {
                // Near-horizontal plane: use median Z of nearby points
                // to avoid slight tilts creating visible slant across flat sections.
                let r2 = 3.0 * 3.0;
                let mut zs: Vec<f64> = points
                    .iter()
                    .filter(|p| {
                        let dx = p.x - cx;
                        let dy = p.y - cy;
                        dx * dx + dy * dy < r2 && footprint.contains_2d(p.x, p.y)
                    })
                    .map(|p| p.z)
                    .collect();
                let h = if zs.len() >= 3 {
                    zs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    zs[zs.len() / 2]
                } else {
                    plane.eval_z(cx, cy).unwrap_or(fallback_h)
                };
                (h, h, h, roof_semantics[pi])
            } else {
                (
                    plane.eval_z(p0.x, p0.y).unwrap_or(fallback_h),
                    plane.eval_z(p1.x, p1.y).unwrap_or(fallback_h),
                    plane.eval_z(p2.x, p2.y).unwrap_or(fallback_h),
                    roof_semantics[pi],
                )
            }
        } else {
            let h = fallback_h;
            (h, h, h, roof_semantics[0])
        };

        let z_min_roof = h_ground + 0.5;
        let z_max_roof = z_95p + 1.0;
        let z0 = z0.clamp(z_min_roof, z_max_roof);
        let z1 = z1.clamp(z_min_roof, z_max_roof);
        let z2 = z2.clamp(z_min_roof, z_max_roof);

        let i0 = get_roof_vertex(&mut mesh, &mut pos_to_idx, p0.x, p0.y, z0);
        let i1 = get_roof_vertex(&mut mesh, &mut pos_to_idx, p1.x, p1.y, z1);
        let i2 = get_roof_vertex(&mut mesh, &mut pos_to_idx, p2.x, p2.y, z2);
        mesh.add_face(Face::new(vec![i0, i1, i2]).with_semantic(sem_idx));
    }

    if mesh.faces.is_empty() {
        return None;
    }

    // Build walls and ground from the FOOTPRINT BOUNDARY directly.
    // This guarantees complete, watertight geometry regardless of which CDT
    // triangles received roof faces.

    let z_min_roof = h_ground + 0.5;
    let z_max_roof = z_95p + 1.0;

    // Evaluate roof height at a 2D position: check existing CDT vertices first,
    // then fall back to plane evaluation.
    fn roof_z_at(
        x: f64, y: f64,
        pos_to_idx: &HashMap<[i64; 2], u32>,
        vertices: &[nalgebra::Point3<f64>],
        planes: &[Plane],
        points: &[nalgebra::Point3<f64>],
        footprint: &Footprint,
        fallback_h: f64,
        z_min: f64,
        z_max: f64,
    ) -> f64 {
        let key = [(x * 1000.0) as i64, (y * 1000.0) as i64];
        if let Some(&vi) = pos_to_idx.get(&key) {
            return vertices[vi as usize].z;
        }
        let best = find_best_plane(planes, points, x, y, footprint);
        let z = match best {
            Some(pi) => planes[pi].eval_z(x, y).unwrap_or(fallback_h),
            None => fallback_h,
        };
        z.clamp(z_min, z_max)
    }

    // Pre-compute roof heights for all footprint boundary vertices
    let ext = &footprint.polygon.exterior;
    let ext_n = ext.len().saturating_sub(1);
    if ext_n < 3 {
        return None;
    }

    let ext_heights: Vec<f64> = (0..ext_n).map(|i| {
        let v = &ext.vertices[i];
        roof_z_at(v.x, v.y, &pos_to_idx, &mesh.vertices, planes, points, footprint, fallback_h, z_min_roof, z_max_roof)
    }).collect();

    let mut hole_heights: Vec<Vec<f64>> = Vec::new();
    for hole in &footprint.polygon.interiors {
        let hn = hole.len().saturating_sub(1);
        if hn < 3 {
            hole_heights.push(Vec::new());
            continue;
        }
        let hh: Vec<f64> = (0..hn).map(|i| {
            let v = &hole.vertices[i];
            roof_z_at(v.x, v.y, &pos_to_idx, &mesh.vertices, planes, points, footprint, fallback_h, z_min_roof, z_max_roof)
        }).collect();
        hole_heights.push(hh);
    }

    // Now build walls and ground — only mutating mesh from here on.
    let mut ext_roof_verts: Vec<u32> = Vec::with_capacity(ext_n);
    let mut ext_ground_verts: Vec<u32> = Vec::with_capacity(ext_n);
    for i in 0..ext_n {
        let v = &ext.vertices[i];
        let z = ext_heights[i];
        let ri = mesh.add_vertex(nalgebra::Point3::new(v.x, v.y, z));
        let gi = mesh.add_vertex(nalgebra::Point3::new(v.x, v.y, h_ground));
        ext_roof_verts.push(ri);
        ext_ground_verts.push(gi);
    }

    // Build exterior wall quads
    for i in 0..ext_n {
        let j = (i + 1) % ext_n;
        let ri = ext_roof_verts[i];
        let rj = ext_roof_verts[j];
        let gi = ext_ground_verts[i];
        let gj = ext_ground_verts[j];
        mesh.add_face(Face::new(vec![gi, gj, rj, ri]).with_semantic(wall_idx));
    }

    // Hole rings: walls + ground vertices
    let mut hole_ground_rings: Vec<Vec<u32>> = Vec::new();
    for (hi, hole) in footprint.polygon.interiors.iter().enumerate() {
        let hn = hole.len().saturating_sub(1);
        if hn < 3 || hole_heights[hi].is_empty() {
            continue;
        }
        let mut hole_roof_verts: Vec<u32> = Vec::with_capacity(hn);
        let mut hole_ground_verts: Vec<u32> = Vec::with_capacity(hn);
        for i in 0..hn {
            let v = &hole.vertices[i];
            let z = hole_heights[hi][i];
            let ri = mesh.add_vertex(nalgebra::Point3::new(v.x, v.y, z));
            let gi = mesh.add_vertex(nalgebra::Point3::new(v.x, v.y, h_ground));
            hole_roof_verts.push(ri);
            hole_ground_verts.push(gi);
        }
        // Hole walls (reversed winding for inward-facing)
        for i in 0..hn {
            let j = (i + 1) % hn;
            let ri = hole_roof_verts[i];
            let rj = hole_roof_verts[j];
            let gi = hole_ground_verts[i];
            let gj = hole_ground_verts[j];
            mesh.add_face(Face::new(vec![gj, gi, ri, rj]).with_semantic(wall_idx));
        }
        hole_ground_rings.push(hole_ground_verts);
    }

    // Ground face from footprint boundary (CW = downward normal)
    let mut ground_face =
        Face::new(ext_ground_verts.iter().rev().copied().collect()).with_semantic(ground_idx);
    for hole_ring in &hole_ground_rings {
        ground_face.holes.push(hole_ring.clone());
    }
    mesh.add_face(ground_face);

    if mesh.faces.len() < 3 {
        return None;
    }

    Some(mesh)
}

fn find_best_plane(
    planes: &[Plane],
    points: &[nalgebra::Point3<f64>],
    cx: f64,
    cy: f64,
    footprint: &Footprint,
) -> Option<usize> {
    let radius = 3.0;
    let r2 = radius * radius;

    // Collect all points within radius of the CDT face centroid
    let nearby: Vec<&nalgebra::Point3<f64>> = points
        .iter()
        .filter(|p| {
            let dx = p.x - cx;
            let dy = p.y - cy;
            dx * dx + dy * dy < r2 && footprint.contains_2d(p.x, p.y)
        })
        .collect();

    if nearby.len() < 2 {
        // Fallback: pick plane closest to centroid
        let mut best_idx = None;
        let mut min_dist = f64::MAX;
        for (pi, plane) in planes.iter().enumerate() {
            if let Some(z) = plane.eval_z(cx, cy) {
                let dist = plane.distance_to(&nalgebra::Point3::new(cx, cy, z));
                if dist < min_dist {
                    min_dist = dist;
                    best_idx = Some(pi);
                }
            }
        }
        return best_idx;
    }

    // For each candidate plane, compute RMSE of nearby points to the plane.
    // The plane that best fits the local point cloud wins.
    let mut best_idx = None;
    let mut best_rmse = f64::MAX;

    for (pi, plane) in planes.iter().enumerate() {
        let mut sum_sq = 0.0;
        for p in &nearby {
            let d = plane.distance_to(p);
            sum_sq += d * d;
        }
        let rmse = (sum_sq / nearby.len() as f64).sqrt();
        if rmse < best_rmse {
            best_rmse = rmse;
            best_idx = Some(pi);
        }
    }

    best_idx
}

/// Validate how well detected planes explain the building point cloud.
///
/// Returns `(rmse, coverage, n_planes)`:
/// - `rmse`: root mean square distance of inlier points to their assigned plane
/// - `coverage`: fraction of total building points within `epsilon` of any plane
/// - `n_planes`: number of planes used
pub fn validate_reconstruction(
    planes: &[Plane],
    points: &[nalgebra::Point3<f64>],
    epsilon: f64,
) -> (f64, f64, usize) {
    if planes.is_empty() || points.is_empty() {
        return (0.0, 0.0, 0);
    }

    let mut sum_sq = 0.0_f64;
    let mut n_inliers = 0usize;

    for (idx, p) in points.iter().enumerate() {
        // Find the closest plane for this point
        let mut min_dist = f64::MAX;
        for plane in planes {
            let d = plane.distance_to(p);
            if d < min_dist {
                min_dist = d;
            }
        }
        if min_dist <= epsilon {
            sum_sq += min_dist * min_dist;
            n_inliers += 1;
        }
    }

    let n_total = points.len() as f64;
    let rmse = if n_inliers > 0 {
        (sum_sq / n_inliers as f64).sqrt()
    } else {
        0.0
    };
    let coverage = n_inliers as f64 / n_total;

    (rmse, coverage, planes.len())
}

pub fn eval_height_at(
    planes: &[Plane],
    x: f64,
    y: f64,
    points: &[nalgebra::Point3<f64>],
    footprint: &Footprint,
) -> Option<f64> {
    let best = find_best_plane(planes, points, x, y, footprint)?;
    planes[best].eval_z(x, y)
}
