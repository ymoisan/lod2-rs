use lod2_common::mesh::Mesh;
use nalgebra::Point3;
use rand::Rng;

/// Sample points uniformly on the surface of a mesh by distributing
/// samples across faces proportional to face area.
fn sample_surface(mesh: &Mesh, n_samples: usize, rng: &mut impl Rng) -> Vec<Point3<f64>> {
    let mut face_areas: Vec<f64> = Vec::with_capacity(mesh.faces.len());
    let mut total_area = 0.0;

    for face in &mesh.faces {
        let area = triangle_fan_area(mesh, &face.indices);
        face_areas.push(area);
        total_area += area;
    }

    if total_area < 1e-12 || mesh.faces.is_empty() {
        return Vec::new();
    }

    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let target = rng.gen::<f64>() * total_area;
        let mut cumulative = 0.0;
        let mut face_idx = 0;
        for (i, &area) in face_areas.iter().enumerate() {
            cumulative += area;
            if cumulative >= target {
                face_idx = i;
                break;
            }
        }

        let face = &mesh.faces[face_idx];
        if face.indices.len() < 3 {
            continue;
        }
        let p0 = mesh.vertices[face.indices[0] as usize];
        let tri_idx = rng.gen_range(0..face.indices.len().saturating_sub(2)) + 1;
        let p1 = mesh.vertices[face.indices[tri_idx] as usize];
        let p2 = mesh.vertices[face.indices[tri_idx + 1] as usize];

        let mut u: f64 = rng.gen();
        let mut v: f64 = rng.gen();
        if u + v > 1.0 {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        let pt = Point3::new(
            p0.x * (1.0 - u - v) + p1.x * u + p2.x * v,
            p0.y * (1.0 - u - v) + p1.y * u + p2.y * v,
            p0.z * (1.0 - u - v) + p1.z * u + p2.z * v,
        );
        samples.push(pt);
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

/// One-directional Hausdorff: max over points in `from` of the min distance to `to`.
fn directed_hausdorff(from: &[Point3<f64>], to: &[Point3<f64>]) -> f64 {
    if from.is_empty() || to.is_empty() {
        return f64::MAX;
    }
    let mut max_min = 0.0f64;
    for p in from {
        let min_dist = to
            .iter()
            .map(|q| (p - q).norm())
            .fold(f64::MAX, f64::min);
        max_min = max_min.max(min_dist);
    }
    max_min
}

/// Symmetric sampled Hausdorff distance between two meshes.
///
/// Samples `n_samples` points on each surface and computes the max of both
/// directed distances.
pub fn hausdorff_distance(
    mesh_a: &Mesh,
    mesh_b: &Mesh,
    n_samples: usize,
) -> Option<f64> {
    let mut rng = rand::thread_rng();
    let sa = sample_surface(mesh_a, n_samples, &mut rng);
    let sb = sample_surface(mesh_b, n_samples, &mut rng);

    if sa.is_empty() || sb.is_empty() {
        return None;
    }

    let d_ab = directed_hausdorff(&sa, &sb);
    let d_ba = directed_hausdorff(&sb, &sa);
    Some(d_ab.max(d_ba))
}

/// Mean distance (Chamfer-like) between two sampled surfaces.
pub fn mean_surface_distance(
    mesh_a: &Mesh,
    mesh_b: &Mesh,
    n_samples: usize,
) -> Option<f64> {
    let mut rng = rand::thread_rng();
    let sa = sample_surface(mesh_a, n_samples, &mut rng);
    let sb = sample_surface(mesh_b, n_samples, &mut rng);

    if sa.is_empty() || sb.is_empty() {
        return None;
    }

    let sum_ab: f64 = sa
        .iter()
        .map(|p| sb.iter().map(|q| (p - q).norm()).fold(f64::MAX, f64::min))
        .sum();
    let sum_ba: f64 = sb
        .iter()
        .map(|p| sa.iter().map(|q| (p - q).norm()).fold(f64::MAX, f64::min))
        .sum();

    Some((sum_ab / sa.len() as f64 + sum_ba / sb.len() as f64) / 2.0)
}
