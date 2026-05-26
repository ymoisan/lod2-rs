use clap::Parser;
use lod2_common::hints::{BuildingHint, RoofShape};
use lod2_common::mesh::{BuildingGeometry, Face, Mesh, SemanticSurface};
use lod2_common::pipeline::{self, PipelineArgs, Reconstructor, build_flat_roof};
use lod2_common::plane::{intersect_planes, clip_line_to_bbox, split_segments_at_intersections, Plane, PlaneDetector, PlaneDetectorConfig};
use lod2_common::point_cloud::PointCloud;
use lod2_common::polygon::Footprint;
use nalgebra::Point3;
use spade::{ConstrainedDelaunayTriangulation, Point2 as SpadePoint2, Triangulation};

struct ArrangementReconstructor;

impl ArrangementReconstructor {
    fn new() -> Self {
        Self
    }

    fn make_config(hint: &BuildingHint) -> PlaneDetectorConfig {
        let max_planes = hint
            .roof_shape
            .as_ref()
            .map(|s| s.suggested_max_planes())
            .unwrap_or(12);
        PlaneDetectorConfig {
            epsilon: 0.2,
            min_points: 10,
            max_planes,
            wall_angle_threshold: 15.0,
            merge_angle_degrees: 10.0,
            merge_distance: 0.4,
            merge_centroid_2d_distance: 5.0,
            ..PlaneDetectorConfig::default()
        }
    }

    fn build_lod22(
        &self,
        footprint: &Footprint,
        points: &PointCloud,
        h_ground: f64,
        hint: &BuildingHint,
    ) -> Option<Mesh> {
        let config = Self::make_config(hint);
        let detector = PlaneDetector::new(config);
        let planes = detector.detect_multiple(&points.positions, 50);
        if planes.is_empty() {
            return None;
        }

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

        if let Some(bearing) = hint.roof_direction {
            let allowed = allowed_bearings(hint);
            segments.retain(|&(p1, p2, _, _)| {
                let seg_bearing = segment_bearing(&p1, &p2);
                allowed.iter().any(|&b| bearing_within(seg_bearing, b, 25.0))
            });
            let _ = bearing; // suppress unused warning
        }

        // Pre-split ridgelines at mutual intersections and footprint boundary crossings
        let fp_edges = footprint_boundary_edges(footprint);
        let ridge_with_meta: Vec<_> = segments.iter().map(|&(p1, p2, i, j)| (p1, p2, (i, j))).collect();
        let split = split_segments_at_intersections(&fp_edges, &ridge_with_meta, 1e-3);
        let segments: Vec<_> = split.into_iter().map(|(p1, p2, (i, j))| (p1, p2, i, j)).collect();

        self.build_arrangement_mesh(footprint, &planes, &segments, &points.positions, h_ground)
    }

    fn build_arrangement_mesh(
        &self,
        footprint: &Footprint,
        planes: &[Plane],
        segments: &[([f64; 2], [f64; 2], usize, usize)],
        points: &[Point3<f64>],
        h_ground: f64,
    ) -> Option<Mesh> {
        let exterior = &footprint.polygon.exterior;
        let n = exterior.len().saturating_sub(1);
        if n < 3 {
            return None;
        }

        let mut cdt: ConstrainedDelaunayTriangulation<SpadePoint2<f64>> =
            ConstrainedDelaunayTriangulation::new();

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

        for (p1, p2, _, _) in segments {
            let h1 = cdt.insert(SpadePoint2::new(p1[0], p1[1]));
            let h2 = cdt.insert(SpadePoint2::new(p2[0], p2[1]));
            if let (Ok(h1), Ok(h2)) = (h1, h2) {
                if h1 != h2 {
                    let _ = cdt.add_constraint(h1, h2);
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

            let best_plane = self.find_best_plane(planes, points, cx, cy, footprint);

            let (z0, z1, z2, sem_idx) = if let Some(pi) = best_plane {
                let plane = &planes[pi];
                (
                    plane.eval_z(p0.x, p0.y).unwrap_or(h_ground),
                    plane.eval_z(p1.x, p1.y).unwrap_or(h_ground),
                    plane.eval_z(p2.x, p2.y).unwrap_or(h_ground),
                    roof_semantics[pi],
                )
            } else {
                let stats = PointCloud { positions: points.to_vec(), classifications: vec![6; points.len()], ndvi: None }.compute_statistics();
                let h = stats.z_70p;
                (h, h, h, roof_semantics[0])
            };

            let i0 = mesh.add_vertex(Point3::new(p0.x, p0.y, z0));
            let i1 = mesh.add_vertex(Point3::new(p1.x, p1.y, z1));
            let i2 = mesh.add_vertex(Point3::new(p2.x, p2.y, z2));
            mesh.add_face(Face::new(vec![i0, i1, i2]).with_semantic(sem_idx));
        }

        let mut bottom = Vec::with_capacity(n);
        for i in 0..n {
            let v = &exterior.vertices[i];
            bottom.push(mesh.add_vertex(Point3::new(v.x, v.y, h_ground)));
        }
        mesh.add_face(
            Face::new(bottom.iter().rev().copied().collect()).with_semantic(ground_idx),
        );

        for i in 0..n {
            let j = (i + 1) % n;
            let vi = &exterior.vertices[i];
            let vj = &exterior.vertices[j];

            let zi = self
                .eval_height_at(planes, vi.x, vi.y, points, footprint)
                .unwrap_or(h_ground + 3.0);
            let zj = self
                .eval_height_at(planes, vj.x, vj.y, points, footprint)
                .unwrap_or(h_ground + 3.0);

            let ti = mesh.add_vertex(Point3::new(vi.x, vi.y, zi));
            let tj = mesh.add_vertex(Point3::new(vj.x, vj.y, zj));
            mesh.add_face(
                Face::new(vec![bottom[i], bottom[j], tj, ti]).with_semantic(wall_idx),
            );
        }

        if mesh.faces.len() < 3 {
            return None;
        }

        Some(mesh)
    }

    fn find_best_plane(
        &self,
        planes: &[Plane],
        points: &[Point3<f64>],
        cx: f64,
        cy: f64,
        footprint: &Footprint,
    ) -> Option<usize> {
        let radius = 3.0;
        let mut best_idx = None;
        let mut best_score = 0usize;

        for (pi, plane) in planes.iter().enumerate() {
            let count = plane
                .inliers
                .iter()
                .filter(|&&i| {
                    let p = &points[i];
                    let dx = p.x - cx;
                    let dy = p.y - cy;
                    dx * dx + dy * dy < radius * radius && footprint.contains_2d(p.x, p.y)
                })
                .count();
            if count > best_score {
                best_score = count;
                best_idx = Some(pi);
            }
        }

        if best_score < 2 {
            let mut min_dist = f64::MAX;
            for (pi, plane) in planes.iter().enumerate() {
                if let Some(z) = plane.eval_z(cx, cy) {
                    let dist = plane.distance_to(&Point3::new(cx, cy, z));
                    if dist < min_dist {
                        min_dist = dist;
                        best_idx = Some(pi);
                    }
                }
            }
        }

        best_idx
    }

    fn eval_height_at(
        &self,
        planes: &[Plane],
        x: f64,
        y: f64,
        points: &[Point3<f64>],
        footprint: &Footprint,
    ) -> Option<f64> {
        let best = self.find_best_plane(planes, points, x, y, footprint)?;
        planes[best].eval_z(x, y)
    }
}

impl Reconstructor for ArrangementReconstructor {
    fn name(&self) -> &str {
        "arrangement"
    }

    fn reconstruct(
        &self,
        footprint: &Footprint,
        points: &PointCloud,
        h_ground: f64,
    ) -> BuildingGeometry {
        use lod2_common::mesh::RoofReason;

        let mut geom = BuildingGeometry::new(&footprint.id);
        geom.attributes = footprint.attributes.clone();
        geom.h_ground = h_ground;
        let hint = BuildingHint::from_footprint(footprint);

        if hint.is_flat() {
            let stats = points.compute_statistics();
            let h_roof = hint.best_roof_height(stats.z_70p, h_ground);
            geom.lod22 = build_flat_roof(footprint, h_ground, h_roof);
            geom.roof_reason = Some(RoofReason::FlatByAttribute);
            return geom;
        }

        if points.len() < 5 {
            let stats = points.compute_statistics();
            let h_roof = hint.best_roof_height(stats.z_70p, h_ground);
            geom.lod22 = build_flat_roof(footprint, h_ground, h_roof);
            geom.roof_reason = Some(RoofReason::FallbackNoPlanes);
            return geom;
        }

        geom.lod22 = self.build_lod22(footprint, points, h_ground, &hint);

        if geom.lod22.is_none() {
            let stats = points.compute_statistics();
            let h_roof = hint.best_roof_height(stats.z_70p, h_ground);
            geom.lod22 = build_flat_roof(footprint, h_ground, h_roof);
            geom.roof_reason = Some(RoofReason::FallbackNoPlanes);
        } else {
            geom.roof_reason = Some(RoofReason::Reconstructed);
        }

        geom
    }
}

/// Compute the bearing (0..360 degrees, north=0, clockwise) of a 2D segment.
fn segment_bearing(p1: &[f64; 2], p2: &[f64; 2]) -> f64 {
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    let mut b = dy.atan2(dx).to_degrees();
    if b < 0.0 {
        b += 360.0;
    }
    b
}

/// Check if two bearings are within `tolerance` degrees of each other,
/// accounting for the 180-degree ambiguity of undirected lines.
fn bearing_within(a: f64, b: f64, tolerance: f64) -> bool {
    let mut diff = (a - b).abs() % 360.0;
    if diff > 180.0 {
        diff = 360.0 - diff;
    }
    diff <= tolerance || (180.0 - diff).abs() <= tolerance
}

/// Return the set of bearings to allow based on roof shape and direction hint.
fn allowed_bearings(hint: &BuildingHint) -> Vec<f64> {
    let bearing = match hint.roof_direction {
        Some(b) => b,
        None => return vec![],
    };
    match hint.roof_shape {
        Some(RoofShape::Gabled) | Some(RoofShape::Skillion) => vec![bearing],
        Some(RoofShape::Hipped) | Some(RoofShape::Pyramidal) => vec![bearing, (bearing + 90.0) % 360.0],
        _ => vec![bearing, (bearing + 90.0) % 360.0],
    }
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

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let args = PipelineArgs::parse();
    let reconstructor = ArrangementReconstructor::new();
    pipeline::run_pipeline(&args, &reconstructor)?;
    Ok(())
}
