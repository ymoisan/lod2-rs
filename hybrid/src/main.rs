use clap::Parser;
use lod2_common::hints::BuildingHint;
use lod2_common::mesh::{BuildingGeometry, RoofReason};
use lod2_common::pipeline::{self, build_flat_roof, PipelineArgs, Reconstructor};
use lod2_common::plane::{self, PlaneDetector, PlaneDetectorConfig};
use lod2_common::point_cloud::{self, PointCloud, PointCloudStats};
use lod2_common::polygon::Footprint;
use las::Reader;
use nalgebra::Point3;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod copc_reader;
mod gpkg_reader;
mod methods;
mod scoring;

struct HybridReconstructor {
    ndvi_threshold: Option<f64>,
}

impl HybridReconstructor {
    fn new(ndvi_threshold: Option<f64>) -> Self {
        Self { ndvi_threshold }
    }

    fn make_config(_hint: &BuildingHint, _stats: &PointCloudStats) -> PlaneDetectorConfig {
        PlaneDetectorConfig {
            epsilon: 0.2,
            min_points: 15,
            wall_angle_threshold: 15.0,
            merge_angle_degrees: 10.0,
            merge_distance: 0.4,
            // Disable centroid-distance gate on plane merging.  Two
            // coplanar detections (same normal, same offset) ARE the
            // same plane regardless of where their inlier clusters
            // happen to be in (x,y) — the angle + distance checks
            // already enforce true coplanarity.  Keeping a small
            // centroid gate fragments large hip/shed roofs into many
            // near-duplicate planes (e.g. 94257 had six ~33° planes
            // for a single slope) which in turn shatter the
            // arrangement into hundreds of perimeter slivers.
            merge_centroid_2d_distance: f64::INFINITY,
            ..PlaneDetectorConfig::default()
        }
    }
}

impl Reconstructor for HybridReconstructor {
    fn name(&self) -> &str {
        "hybrid"
    }

    fn reconstruct(
        &self,
        footprint: &Footprint,
        points: &PointCloud,
        h_ground: f64,
    ) -> BuildingGeometry {
        let mut geom = BuildingGeometry::new(&footprint.id);
        geom.attributes = footprint.attributes.clone();
        geom.h_ground = h_ground;
        let hint = BuildingHint::from_footprint(footprint);
        let stats = points.compute_statistics();
        let h_roof = hint.best_roof_height(stats.z_70p, h_ground);

        // Always attempt plane detection if we have enough points
        if points.len() >= 15 {
            let config = Self::make_config(&hint, &stats);
            let detector = PlaneDetector::new(config);
            let all_planes = detector.detect_multiple(&points.positions, usize::MAX);

            // Spectral plane validation — reject canopy-dominated planes
            let all_planes = if let (Some(threshold), Some(ref ndvi)) = (self.ndvi_threshold, &points.ndvi) {
                plane::filter_vegetation_planes(all_planes, ndvi, threshold as f32, 0.6).0
            } else {
                all_planes
            };

            // Filter out near-vertical planes (walls)
            let mut planes: Vec<_> = all_planes.into_iter()
                .filter(|p| p.slope_degrees() < 60.0)
                .collect();

            // Drop rooftop attachments (HVAC/chimney/parapet bumps) whose
            // 2D footprint sits inside a larger plane's footprint and
            // higher in z. Geometric criterion, no tunable threshold.
            planes = plane::filter_rooftop_attachments(planes, &points.positions);

            // Phase 4: Single residual pass — only if coverage is low and
            // building has reasonable point count (avoid expensive re-detection
            // on very large point clouds).
            if planes.len() < 15 && points.len() < 2000 {
                let epsilon = 0.3;
                let min_pts = 15usize;
                // Find uncovered points
                let mut residual_indices: Vec<usize> = Vec::new();
                for (i, pt) in points.positions.iter().enumerate() {
                    let covered = planes.iter().any(|p| p.distance_to(pt).abs() < epsilon);
                    if !covered {
                        residual_indices.push(i);
                    }
                }
                if residual_indices.len() >= min_pts {
                    let residual_pts: Vec<_> = residual_indices.iter().map(|&i| points.positions[i]).collect();
                    let new_planes = detector.detect_multiple(&residual_pts, 10);
                    for mut np in new_planes {
                        if np.slope_degrees() >= 60.0 { continue; }
                        np.inliers = np.inliers.iter().map(|&i| residual_indices[i]).collect();
                        planes.push(np);
                    }
                }
            }

            // Compute point-cloud validation metrics
            let (rmse, coverage, n_planes) = pipeline::validate_reconstruction(
                &planes, &points.positions, 0.3,
            );
            geom.attributes.insert_float("rf_rmse_lod22", rmse);
            geom.attributes.insert_float("rf_coverage", coverage);
            geom.attributes.insert_int("rf_n_planes", n_planes as i64);

            if !planes.is_empty() {
                // Cap planes to avoid alpha-expansion blowup on complex buildings
                if planes.len() > 20 {
                    // Keep the 20 largest planes by inlier count
                    planes.sort_by(|a, b| b.inliers.len().cmp(&a.inliers.len()));
                    planes.truncate(20);
                }
                // Use graph-cut optimized reconstruction (catch panics from spade CDT)
                let gc_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    methods::build_graphcut_lod22(
                        footprint, &planes, &points.positions, h_ground, &hint,
                    )
                }));
                if let Ok(Some(mesh)) = gc_result {
                    geom.lod22 = Some(mesh);
                    geom.roof_reason = Some(RoofReason::Reconstructed);
                    return geom;
                }
            }
        }

        // No planes detected or not enough points: flat roof
        geom.lod22 = build_flat_roof(footprint, h_ground, h_roof);
        geom.roof_reason = Some(RoofReason::FlatByAttribute);
        geom
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("info".parse()?),
        )
        .init();

    let args = PipelineArgs::parse();
    let reconstructor = HybridReconstructor::new(args.ndvi_threshold);

    // Read footprints via pure-Rust GeoPackage reader (no GDAL).
    tracing::info!("Reading footprints from {}", args.footprints.display());
    let mut footprints = gpkg_reader::read_footprints(&args.footprints)?;
    let crs = gpkg_reader::read_crs(&args.footprints)?;
    tracing::info!("Found CRS: {:?}", crs);
    tracing::info!("Read {} footprints", footprints.len());
    args.apply_fid_filter(&mut footprints);

    // Read point cloud with classification from LAS/LAZ.
    tracing::info!("Reading point cloud from {}", args.pointcloud.display());
    let is_copc = args.pointcloud.to_string_lossy().ends_with(".copc.laz");
    let pc = if is_copc {
        tracing::info!("Detected COPC format, using COPC reader");
        let mut reader = copc_reader::CopcReader::from_path(&args.pointcloud)?;
        let mut pc = PointCloud::with_capacity(0);
        for (pt, _level) in reader.points(copc_reader::LodSelection::All, copc_reader::BoundsSelection::All)? {
            let class = u8::from(pt.classification);
            match (pt.color, pt.nir) {
                (Some(c), Some(nir)) => {
                    let ndvi = point_cloud::compute_ndvi(c.red, nir);
                    pc.push_with_ndvi(Point3::new(pt.x, pt.y, pt.z), class, ndvi);
                }
                _ => pc.push_classified(Point3::new(pt.x, pt.y, pt.z), class),
            }
        }
        pc
    } else {
        let mut reader = Reader::from_path(&args.pointcloud)?;
        let n_points = reader.header().number_of_points() as usize;
        let mut pc = PointCloud::with_capacity(n_points);
        for wrapped in reader.points() {
            let pt = wrapped?;
            let class = u8::from(pt.classification);
            match (pt.color, pt.nir) {
                (Some(c), Some(nir)) => {
                    let ndvi = point_cloud::compute_ndvi(c.red, nir);
                    pc.push_with_ndvi(Point3::new(pt.x, pt.y, pt.z), class, ndvi);
                }
                _ => pc.push_classified(Point3::new(pt.x, pt.y, pt.z), class),
            }
        }
        pc
    };

    // Log classification breakdown
    let mut class_counts: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
    for &c in &pc.classifications {
        *class_counts.entry(c).or_insert(0) += 1;
    }
    let mut sorted_classes: Vec<_> = class_counts.iter().collect();
    sorted_classes.sort_by_key(|(k, _)| *k);
    for (class, count) in &sorted_classes {
        tracing::info!("  Class {}: {} points", class, count);
    }
    let n_building = *class_counts.get(&6).unwrap_or(&0);
    let n_ground = *class_counts.get(&2).unwrap_or(&0);
    let n_other = pc.len() - n_building - n_ground;
    tracing::info!(
        "Point cloud: {} building (class 6), {} ground (class 2), {} other (ignored)",
        n_building, n_ground, n_other
    );
    if n_building == 0 {
        tracing::warn!(
            "No class 6 (Building) points found — reconstruction will produce flat roofs only. \
             Check point cloud classification."
        );
    }

    if pc.has_spectral() {
        tracing::info!("Spectral data (NDVI) available from RGB+NIR");
        if let Some(t) = args.ndvi_threshold {
            tracing::info!("NDVI vegetation filter active: threshold = {:.2}", t);
        }
    } else if args.ndvi_threshold.is_some() {
        tracing::warn!(
            "--ndvi-threshold specified but point cloud has no spectral data (needs LAS format 7/8 with RGB+NIR)"
        );
    }

    // Run reconstruction with classification-aware point cropping.
    pipeline::write_results_classified(&args.output, &footprints, &pc, crs.as_deref(), &reconstructor, args.ndvi_threshold)
}
