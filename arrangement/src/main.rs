use clap::Parser;
use lod2_common::mesh::{BuildingGeometry, Face, Mesh, SemanticSurface};
use lod2_common::pipeline::{self, PipelineArgs, Reconstructor, build_flat_roof};
use lod2_common::plane::{intersect_planes, clip_line_to_bbox, Plane, PlaneDetector, RansacConfig};
use lod2_common::point_cloud::PointCloud;
use lod2_common::polygon::Footprint;
use nalgebra::Point3;
use spade::{ConstrainedDelaunayTriangulation, Point2 as SpadePoint2, Triangulation};

/// City3D-style: plane arrangement + face selection optimization.
struct ArrangementReconstructor {
    detector: PlaneDetector,
}

impl ArrangementReconstructor {
    fn new() -> Self {
        Self {
            detector: PlaneDetector::new(RansacConfig {
                epsilon: 0.25,
                max_iterations: 2000,
                min_points: 10,
                max_planes: 25,
                wall_angle_threshold: 70.0,
                merge_angle_degrees: 10.0,
                merge_distance: 0.4,
            }),
        }
    }

    fn build_lod22(
        &self,
        footprint: &Footprint,
        points: &PointCloud,
        h_ground: f64,
    ) -> Option<Mesh> {
        let planes = self.detector.detect_multiple(&points.positions, 20);
        if planes.is_empty() {
            let stats = points.compute_statistics();
            return build_flat_roof(footprint, h_ground, stats.z_70p);
        }

        let bbox = footprint.polygon.bbox_2d();

        // Compute ridgelines (pairwise plane intersections clipped to bbox)
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

        // Build CDT with footprint boundary + ridgeline segments
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

        // Insert footprint boundary vertices
        let mut boundary_handles = Vec::new();
        for i in 0..n {
            let v = &exterior.vertices[i];
            match cdt.insert(SpadePoint2::new(v.x, v.y)) {
                Ok(h) => boundary_handles.push(h),
                Err(_) => return None,
            }
        }

        // Add footprint boundary as constraints
        for i in 0..boundary_handles.len() {
            let j = (i + 1) % boundary_handles.len();
            let _ = cdt.add_constraint(boundary_handles[i], boundary_handles[j]);
        }

        // Insert ridgeline segments as constraints
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

        // Per-plane roof semantic indices
        let roof_semantics: Vec<usize> = planes
            .iter()
            .map(|p| {
                mesh.add_semantic(SemanticSurface::roof_with_stats(
                    p.slope_degrees(),
                    p.azimuth_degrees(),
                ))
            })
            .collect();

        // Classify each triangle face: assign to best-fitting plane
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

            // Score each plane: how many inlier points are near this triangle?
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
                let stats = PointCloud { positions: points.to_vec() }.compute_statistics();
                let h = stats.z_70p;
                (h, h, h, roof_semantics[0])
            };

            let i0 = mesh.add_vertex(Point3::new(p0.x, p0.y, z0));
            let i1 = mesh.add_vertex(Point3::new(p1.x, p1.y, z1));
            let i2 = mesh.add_vertex(Point3::new(p2.x, p2.y, z2));
            mesh.add_face(Face::new(vec![i0, i1, i2]).with_semantic(sem_idx));
        }

        // Add ground and walls
        let mut bottom = Vec::with_capacity(n);
        for i in 0..n {
            let v = &exterior.vertices[i];
            bottom.push(mesh.add_vertex(Point3::new(v.x, v.y, h_ground)));
        }
        mesh.add_face(
            Face::new(bottom.iter().rev().copied().collect()).with_semantic(ground_idx),
        );

        // Outer walls: for each footprint edge, create a wall quad
        for i in 0..n {
            let j = (i + 1) % n;
            let vi = &exterior.vertices[i];
            let vj = &exterior.vertices[j];

            // Find roof height at these boundary points (from nearest plane)
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
            // Fall back to closest plane by distance
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
        let mut geom = BuildingGeometry::new(&footprint.id);
        geom.attributes = footprint.attributes.clone();
        geom.h_ground = h_ground;

        if points.len() < 5 {
            let stats = points.compute_statistics();
            let h_roof = if stats.count > 0 { stats.z_70p } else { h_ground + 5.0 };
            geom.lod22 = build_flat_roof(footprint, h_ground, h_roof);
            return geom;
        }

        geom.lod22 = self.build_lod22(footprint, points, h_ground);

        if geom.lod22.is_none() {
            let stats = points.compute_statistics();
            geom.lod22 = build_flat_roof(footprint, h_ground, stats.z_70p);
        }

        geom
    }
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
