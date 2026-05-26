use lod2_common::mesh::{Mesh, SurfaceType};
use nalgebra::Point3;
use rand::Rng;

/// Bidirectional symmetric mean distance between the roof point cloud and
/// the mesh roof surfaces. Lower is better. Returns `None` if the mesh
/// fails the validity gate (non-positive volume, no roof faces, no roof points).
pub fn score_mesh(mesh: &Mesh, points: &[Point3<f64>], h_ground: f64) -> Option<f64> {
    if mesh.compute_volume() <= 0.0 {
        return None;
    }

    let roof_faces = roof_face_indices(mesh);
    if roof_faces.is_empty() {
        return None;
    }

    let p_roof: Vec<&Point3<f64>> = points.iter().filter(|p| p.z > h_ground + 0.5).collect();
    if p_roof.is_empty() {
        return None;
    }

    let c2m: f64 = p_roof
        .iter()
        .map(|p| min_distance_to_roof_faces(p, mesh, &roof_faces))
        .sum::<f64>()
        / p_roof.len() as f64;

    let samples = sample_roof_faces(mesh, &roof_faces, 500);
    if samples.is_empty() {
        return Some(c2m);
    }

    let m2c: f64 = samples
        .iter()
        .map(|s| {
            p_roof
                .iter()
                .map(|p| (s - *p).norm())
                .fold(f64::MAX, f64::min)
        })
        .sum::<f64>()
        / samples.len() as f64;

    Some((c2m + m2c) / 2.0)
}

fn roof_face_indices(mesh: &Mesh) -> Vec<usize> {
    mesh.faces
        .iter()
        .enumerate()
        .filter_map(|(i, face)| {
            let si = face.semantic_index?;
            if mesh.semantics[si].surface_type == SurfaceType::RoofSurface {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

fn min_distance_to_roof_faces(point: &Point3<f64>, mesh: &Mesh, roof_faces: &[usize]) -> f64 {
    let mut min_dist = f64::MAX;
    for &fi in roof_faces {
        let face = &mesh.faces[fi];
        if face.indices.len() < 3 {
            continue;
        }
        let p0 = mesh.vertices[face.indices[0] as usize];
        for i in 1..(face.indices.len() - 1) {
            let p1 = mesh.vertices[face.indices[i] as usize];
            let p2 = mesh.vertices[face.indices[i + 1] as usize];
            let d = point_to_triangle_distance(point, &p0, &p1, &p2);
            if d < min_dist {
                min_dist = d;
            }
        }
    }
    min_dist
}

/// Closest-point-on-triangle distance (Ericson, Real-Time Collision Detection).
fn point_to_triangle_distance(
    p: &Point3<f64>,
    a: &Point3<f64>,
    b: &Point3<f64>,
    c: &Point3<f64>,
) -> f64 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (p - a).norm();
    }

    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (p - b).norm();
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let closest = Point3::from(a.coords + ab * v);
        return (p - closest).norm();
    }

    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (p - c).norm();
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let closest = Point3::from(a.coords + ac * w);
        return (p - closest).norm();
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let closest = Point3::from(b.coords + (c - b) * w);
        return (p - closest).norm();
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let closest = Point3::from(a.coords + ab * v + ac * w);
    (p - closest).norm()
}

fn sample_roof_faces(mesh: &Mesh, roof_faces: &[usize], n_samples: usize) -> Vec<Point3<f64>> {
    let mut rng = rand::thread_rng();
    let mut areas = Vec::with_capacity(roof_faces.len());
    let mut total_area = 0.0;

    for &fi in roof_faces {
        let face = &mesh.faces[fi];
        let area = triangle_fan_area(mesh, &face.indices);
        areas.push(area);
        total_area += area;
    }

    if total_area < 1e-12 {
        return Vec::new();
    }

    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let target = rng.gen::<f64>() * total_area;
        let mut cumulative = 0.0;
        let mut face_local_idx = 0;
        for (i, &area) in areas.iter().enumerate() {
            cumulative += area;
            if cumulative >= target {
                face_local_idx = i;
                break;
            }
        }

        let fi = roof_faces[face_local_idx];
        let face = &mesh.faces[fi];
        if face.indices.len() < 3 {
            continue;
        }

        let p0 = mesh.vertices[face.indices[0] as usize];
        let tri_idx = if face.indices.len() > 3 {
            rng.gen_range(1..face.indices.len() - 1)
        } else {
            1
        };
        let p1 = mesh.vertices[face.indices[tri_idx] as usize];
        let p2 = mesh.vertices[face.indices[tri_idx + 1] as usize];

        let mut u: f64 = rng.gen();
        let mut v: f64 = rng.gen();
        if u + v > 1.0 {
            u = 1.0 - u;
            v = 1.0 - v;
        }

        samples.push(Point3::new(
            p0.x * (1.0 - u - v) + p1.x * u + p2.x * v,
            p0.y * (1.0 - u - v) + p1.y * u + p2.y * v,
            p0.z * (1.0 - u - v) + p1.z * u + p2.z * v,
        ));
    }
    samples
}

fn triangle_fan_area(mesh: &Mesh, indices: &[u32]) -> f64 {
    if indices.len() < 3 {
        return 0.0;
    }
    let p0 = &mesh.vertices[indices[0] as usize];
    let mut area = 0.0;
    for i in 1..(indices.len() - 1) {
        let p1 = &mesh.vertices[indices[i] as usize];
        let p2 = &mesh.vertices[indices[i + 1] as usize];
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        area += e1.cross(&e2).norm() * 0.5;
    }
    area
}
