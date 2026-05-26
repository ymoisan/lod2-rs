use clap::Parser;
use lod2_common::hints::BuildingHint;
use lod2_common::mesh::{BuildingGeometry, Face, Mesh, SemanticSurface};
use lod2_common::pipeline::{self, PipelineArgs, Reconstructor, build_flat_roof};
use lod2_common::plane::{Plane, PlaneDetector, PlaneDetectorConfig};
use lod2_common::point_cloud::PointCloud;
use lod2_common::polygon::Footprint;
use nalgebra::Point3;

struct PlaneExtrudeReconstructor;

impl PlaneExtrudeReconstructor {
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

        let z_max = hint
            .estimated_height()
            .map(|h| h_ground + h);

        let mut mesh = Mesh::new();
        let ground_idx = mesh.add_semantic(SemanticSurface::ground());

        for plane in &planes {
            self.add_plane_to_mesh(&mut mesh, plane, footprint, &points.positions, h_ground, z_max);
        }

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
            return None;
        }

        Some(mesh)
    }

    fn add_plane_to_mesh(
        &self,
        mesh: &mut Mesh,
        plane: &Plane,
        footprint: &Footprint,
        points: &[Point3<f64>],
        h_ground: f64,
        z_max: Option<f64>,
    ) {
        let roof_idx = mesh.add_semantic(SemanticSurface::roof_with_stats(
            plane.slope_degrees(),
            plane.azimuth_degrees(),
        ));
        let wall_idx = mesh.add_semantic(SemanticSurface::wall(true));

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

        let clipped: Vec<(f64, f64)> = hull
            .iter()
            .filter(|(x, y)| footprint.contains_2d(*x, *y))
            .copied()
            .collect();

        let roof_pts = if clipped.len() >= 3 { &clipped } else { &hull };
        if roof_pts.len() < 3 {
            return;
        }

        let mut roof_verts = Vec::new();
        for &(x, y) in roof_pts {
            let mut z = plane.eval_z(x, y).unwrap_or(h_ground + 5.0);
            z = z.max(h_ground);
            if let Some(ceil) = z_max {
                z = z.min(ceil);
            }
            roof_verts.push(mesh.add_vertex(Point3::new(x, y, z)));
        }
        mesh.add_face(Face::new(roof_verts.clone()).with_semantic(roof_idx));

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
