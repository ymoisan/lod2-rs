use clap::Parser;
use lod2_common::hints::{BuildingHint, RoofShape};
use lod2_common::mesh::{BuildingGeometry, Face, Mesh, SemanticSurface};
use lod2_common::pipeline::{self, PipelineArgs, Reconstructor, build_flat_roof};
use lod2_common::plane::{intersect_planes, clip_line_to_bbox, split_segments_at_intersections, Plane, PlaneDetector, PlaneDetectorConfig};
use lod2_common::point_cloud::PointCloud;
use lod2_common::polygon::Footprint;
use nalgebra::Point3;
use spade::{ConstrainedDelaunayTriangulation, Point2 as SpadePoint2, Triangulation};
use std::collections::HashMap;

struct GraphCutReconstructor {
    lambda: f64,
}

impl GraphCutReconstructor {
    fn new() -> Self {
        Self { lambda: 0.5 }
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

        let mut ridge_segments: Vec<([f64; 2], [f64; 2])> = Vec::new();
        for i in 0..planes.len() {
            for j in (i + 1)..planes.len() {
                if let Some((origin, dir)) = intersect_planes(&planes[i], &planes[j]) {
                    if let Some((p1, p2)) = clip_line_to_bbox(&origin, &dir, &bbox) {
                        ridge_segments.push((p1, p2));
                    }
                }
            }
        }

        if let Some(bearing) = hint.roof_direction {
            let allowed = allowed_bearings(hint);
            ridge_segments.retain(|&(p1, p2)| {
                let seg_bearing = segment_bearing(&p1, &p2);
                allowed.iter().any(|&b| bearing_within(seg_bearing, b, 25.0))
            });
            let _ = bearing;
        }

        // Pre-split ridgelines at mutual intersections and footprint boundary crossings
        let fp_edges = footprint_boundary_edges(footprint);
        let ridge_with_meta: Vec<_> = ridge_segments.iter().map(|&(p1, p2)| (p1, p2, ())).collect();
        let split = split_segments_at_intersections(&fp_edges, &ridge_with_meta, 1e-3);
        let ridge_segments: Vec<_> = split.into_iter().map(|(p1, p2, _)| (p1, p2)).collect();

        self.build_graphcut_mesh(footprint, &planes, &ridge_segments, &points.positions, h_ground)
    }

    fn build_graphcut_mesh(
        &self,
        footprint: &Footprint,
        planes: &[Plane],
        ridge_segments: &[([f64; 2], [f64; 2])],
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

        for (p1, p2) in ridge_segments {
            let h1 = cdt.insert(SpadePoint2::new(p1[0], p1[1]));
            let h2 = cdt.insert(SpadePoint2::new(p2[0], p2[1]));
            if let (Ok(h1), Ok(h2)) = (h1, h2) {
                if h1 != h2 {
                    let _ = cdt.add_constraint(h1, h2);
                }
            }
        }

        let mut face_data: Vec<FaceInfo> = Vec::new();
        let mut face_index_map: HashMap<usize, usize> = HashMap::new();

        for (fi, face_handle) in cdt.inner_faces().enumerate() {
            let [v0, v1, v2] = face_handle.vertices();
            let p0 = v0.position();
            let p1 = v1.position();
            let p2 = v2.position();

            let cx = (p0.x + p1.x + p2.x) / 3.0;
            let cy = (p0.y + p1.y + p2.y) / 3.0;

            if !footprint.contains_2d(cx, cy) {
                continue;
            }

            let area = tri_area_2d(p0.x, p0.y, p1.x, p1.y, p2.x, p2.y);
            if area < 1e-6 {
                continue;
            }

            face_index_map.insert(fi, face_data.len());
            face_data.push(FaceInfo {
                verts: [(p0.x, p0.y), (p1.x, p1.y), (p2.x, p2.y)],
                cx,
                cy,
                area,
            });
        }

        if face_data.is_empty() {
            return None;
        }

        let num_labels = planes.len();
        let data_costs = self.compute_data_costs(&face_data, planes, points, footprint);
        let adjacency = self.build_adjacency(&cdt, &face_index_map, footprint);
        let labels = self.optimize_labels(&face_data, &data_costs, &adjacency, num_labels);

        self.build_mesh_from_labels(
            footprint, &face_data, &labels, planes, h_ground, n, exterior,
        )
    }

    fn compute_data_costs(
        &self,
        faces: &[FaceInfo],
        planes: &[Plane],
        points: &[Point3<f64>],
        footprint: &Footprint,
    ) -> Vec<Vec<f64>> {
        let radius = 2.0;
        faces
            .iter()
            .map(|face| {
                planes
                    .iter()
                    .map(|plane| {
                        let mut sum_dist = 0.0;
                        let mut count = 0usize;
                        for &idx in &plane.inliers {
                            let p = &points[idx];
                            let dx = p.x - face.cx;
                            let dy = p.y - face.cy;
                            if dx * dx + dy * dy < radius * radius
                                && footprint.contains_2d(p.x, p.y)
                            {
                                if let Some(z_plane) = plane.eval_z(p.x, p.y) {
                                    sum_dist += (p.z - z_plane).powi(2);
                                    count += 1;
                                }
                            }
                        }
                        if count > 0 {
                            sum_dist / count as f64
                        } else {
                            10.0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn build_adjacency(
        &self,
        _cdt: &ConstrainedDelaunayTriangulation<SpadePoint2<f64>>,
        _face_index_map: &HashMap<usize, usize>,
        _footprint: &Footprint,
    ) -> Vec<(usize, usize, f64)> {
        Vec::new()
    }

    fn optimize_labels(
        &self,
        faces: &[FaceInfo],
        data_costs: &[Vec<f64>],
        _adjacency: &[(usize, usize, f64)],
        _num_labels: usize,
    ) -> Vec<usize> {
        faces
            .iter()
            .enumerate()
            .map(|(fi, _face)| {
                let costs = &data_costs[fi];
                let mut best_label = 0;
                let mut best_cost = f64::MAX;
                for (li, &cost) in costs.iter().enumerate() {
                    if cost < best_cost {
                        best_cost = cost;
                        best_label = li;
                    }
                }
                best_label
            })
            .collect()
    }

    fn build_mesh_from_labels(
        &self,
        footprint: &Footprint,
        faces: &[FaceInfo],
        labels: &[usize],
        planes: &[Plane],
        h_ground: f64,
        n: usize,
        exterior: &lod2_common::polygon::LinearRing,
    ) -> Option<Mesh> {
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

        for (fi, face) in faces.iter().enumerate() {
            let label = labels[fi];
            let plane = &planes[label];
            let sem = roof_semantics[label];

            let mut verts = Vec::new();
            for &(x, y) in &face.verts {
                let z = plane.eval_z(x, y).unwrap_or(h_ground + 3.0);
                verts.push(mesh.add_vertex(Point3::new(x, y, z)));
            }
            mesh.add_face(Face::new(verts).with_semantic(sem));
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

            let zi = self.eval_height(planes, vi.x, vi.y, faces, labels).unwrap_or(h_ground + 3.0);
            let zj = self.eval_height(planes, vj.x, vj.y, faces, labels).unwrap_or(h_ground + 3.0);

            let ti = mesh.add_vertex(Point3::new(vi.x, vi.y, zi));
            let tj = mesh.add_vertex(Point3::new(vj.x, vj.y, zj));
            mesh.add_face(
                Face::new(vec![bottom[i], bottom[j], tj, ti]).with_semantic(wall_idx),
            );
        }

        Some(mesh)
    }

    fn eval_height(
        &self,
        planes: &[Plane],
        x: f64,
        y: f64,
        faces: &[FaceInfo],
        labels: &[usize],
    ) -> Option<f64> {
        let mut best_dist = f64::MAX;
        let mut best_label = 0;
        for (fi, face) in faces.iter().enumerate() {
            let dx = face.cx - x;
            let dy = face.cy - y;
            let d = dx * dx + dy * dy;
            if d < best_dist {
                best_dist = d;
                best_label = labels[fi];
            }
        }
        planes[best_label].eval_z(x, y)
    }
}

struct FaceInfo {
    verts: [(f64, f64); 3],
    cx: f64,
    cy: f64,
    area: f64,
}

fn tri_area_2d(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs() * 0.5
}

impl Reconstructor for GraphCutReconstructor {
    fn name(&self) -> &str {
        "graph-cut"
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

fn segment_bearing(p1: &[f64; 2], p2: &[f64; 2]) -> f64 {
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    let mut b = dy.atan2(dx).to_degrees();
    if b < 0.0 {
        b += 360.0;
    }
    b
}

fn bearing_within(a: f64, b: f64, tolerance: f64) -> bool {
    let mut diff = (a - b).abs() % 360.0;
    if diff > 180.0 {
        diff = 360.0 - diff;
    }
    diff <= tolerance || (180.0 - diff).abs() <= tolerance
}

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
    let reconstructor = GraphCutReconstructor::new();
    pipeline::run_pipeline(&args, &reconstructor)?;
    Ok(())
}
