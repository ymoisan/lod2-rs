use clap::Parser;
use lod2_common::mesh::{BuildingGeometry, Face, Mesh, SemanticSurface};
use lod2_common::pipeline::{self, PipelineArgs, Reconstructor, build_flat_roof};
use lod2_common::plane::{Plane, PlaneDetector, RansacConfig};
use lod2_common::point_cloud::PointCloud;
use lod2_common::polygon::Footprint;
use nalgebra::Point3;

/// TerraScan-style: detect planes via RANSAC, extrude each independently.
struct PlaneExtrudeReconstructor {
    detector: PlaneDetector,
}

impl PlaneExtrudeReconstructor {
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

    /// Build LoD 2.2 mesh by detecting planes and extruding each roof plane
    /// with walls down to ground level.
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

        let mut mesh = Mesh::new();
        let ground_idx = mesh.add_semantic(SemanticSurface::ground());

        for plane in &planes {
            self.add_plane_to_mesh(&mut mesh, plane, footprint, &points.positions, h_ground);
        }

        // Ground face from footprint
        let exterior = &footprint.polygon.exterior;
        let n = exterior.len().saturating_sub(1);
        if n >= 3 {
            let mut ground_verts = Vec::with_capacity(n);
            for i in 0..n {
                let v = &exterior.vertices[i];
                ground_verts.push(mesh.add_vertex(Point3::new(v.x, v.y, h_ground)));
            }
            mesh.add_face(
                Face::new(ground_verts.iter().rev().copied().collect()).with_semantic(ground_idx),
            );
        }

        if mesh.faces.len() < 2 {
            let stats = points.compute_statistics();
            return build_flat_roof(footprint, h_ground, stats.z_70p);
        }

        Some(mesh)
    }

    /// Add a single detected plane as roof + walls to the mesh.
    fn add_plane_to_mesh(
        &self,
        mesh: &mut Mesh,
        plane: &Plane,
        footprint: &Footprint,
        points: &[Point3<f64>],
        h_ground: f64,
    ) {
        let roof_idx = mesh.add_semantic(SemanticSurface::roof_with_stats(
            plane.slope_degrees(),
            plane.azimuth_degrees(),
        ));
        let wall_idx = mesh.add_semantic(SemanticSurface::wall(true));

        // Get 2D convex hull of inlier points projected onto footprint
        let mut pts_2d: Vec<(f64, f64)> = plane
            .inliers
            .iter()
            .filter_map(|&i| {
                let p = &points[i];
                if footprint.contains_2d(p.x, p.y) {
                    Some((p.x, p.y))
                } else {
                    None
                }
            })
            .collect();

        if pts_2d.len() < 3 {
            return;
        }

        let hull = convex_hull_2d(&mut pts_2d);
        if hull.len() < 3 {
            return;
        }

        // Clip hull to footprint exterior (simplified: keep points inside)
        let clipped: Vec<(f64, f64)> = hull
            .iter()
            .filter(|(x, y)| footprint.contains_2d(*x, *y))
            .copied()
            .collect();

        let roof_pts = if clipped.len() >= 3 { &clipped } else { &hull };
        if roof_pts.len() < 3 {
            return;
        }

        // Create roof face with Z from plane equation
        let mut roof_verts = Vec::new();
        for &(x, y) in roof_pts {
            let z = plane.eval_z(x, y).unwrap_or(h_ground + 5.0);
            roof_verts.push(mesh.add_vertex(Point3::new(x, y, z)));
        }
        mesh.add_face(Face::new(roof_verts.clone()).with_semantic(roof_idx));

        // Create wall faces extruding down to ground
        let n = roof_verts.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let vi_top = roof_verts[i];
            let vj_top = roof_verts[j];
            let pi_pos = mesh.vertices[vi_top as usize];
            let pj_pos = mesh.vertices[vj_top as usize];
            let vi_bot = mesh.add_vertex(Point3::new(pi_pos.x, pi_pos.y, h_ground));
            let vj_bot = mesh.add_vertex(Point3::new(pj_pos.x, pj_pos.y, h_ground));
            mesh.add_face(
                Face::new(vec![vi_top, vj_top, vj_bot, vi_bot]).with_semantic(wall_idx),
            );
        }
    }
}

impl Reconstructor for PlaneExtrudeReconstructor {
    fn name(&self) -> &str {
        "plane-extrude"
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

/// Simple 2D convex hull (Andrew's monotone chain).
fn convex_hull_2d(pts: &mut Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
    pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-10 && (a.1 - b.1).abs() < 1e-10);
    let n = pts.len();
    if n < 3 {
        return pts.to_vec();
    }

    let mut hull = Vec::with_capacity(2 * n);

    for &p in pts.iter() {
        while hull.len() >= 2 && cross(&hull[hull.len() - 2], &hull[hull.len() - 1], &p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }

    let lower_len = hull.len() + 1;
    for &p in pts.iter().rev().skip(1) {
        while hull.len() >= lower_len
            && cross(&hull[hull.len() - 2], &hull[hull.len() - 1], &p) <= 0.0
        {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

fn cross(o: &(f64, f64), a: &(f64, f64), b: &(f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let args = PipelineArgs::parse();
    let reconstructor = PlaneExtrudeReconstructor::new();
    pipeline::run_pipeline(&args, &reconstructor)?;
    Ok(())
}
