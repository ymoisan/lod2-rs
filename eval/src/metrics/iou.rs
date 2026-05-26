use lod2_common::mesh::{Mesh, SurfaceType};

/// Extract the 2D ground-face polygon vertices from a mesh.
fn ground_face_polygon(mesh: &Mesh) -> Vec<(f64, f64)> {
    let mut polygon = Vec::new();
    for face in &mesh.faces {
        let is_ground = face
            .semantic_index
            .and_then(|idx| mesh.semantics.get(idx))
            .map(|s| s.surface_type == SurfaceType::GroundSurface)
            .unwrap_or(false);
        if is_ground {
            for &idx in &face.indices {
                let v = &mesh.vertices[idx as usize];
                polygon.push((v.x, v.y));
            }
        }
    }
    polygon
}

/// Compute the 2D axis-aligned bounding box of a point set.
fn bbox(pts: &[(f64, f64)]) -> Option<(f64, f64, f64, f64)> {
    if pts.is_empty() {
        return None;
    }
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;
    for &(x, y) in pts {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    Some((min_x, min_y, max_x, max_y))
}

/// Shoelace formula for the signed area of a polygon given as ordered vertices.
fn polygon_area(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += pts[i].0 * pts[j].1;
        area -= pts[j].0 * pts[i].1;
    }
    (area * 0.5).abs()
}

/// Point-in-polygon via ray casting.
fn point_in_polygon(px: f64, py: f64, poly: &[(f64, f64)]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Rasterised IoU: sample a fine grid over the union bbox and count cells
/// inside each polygon separately to approximate intersection and union areas.
///
/// Resolution in CRS units (e.g., 0.5 m for UTM).
pub fn footprint_iou(mesh_a: &Mesh, mesh_b: &Mesh, resolution: f64) -> Option<f64> {
    let poly_a = ground_face_polygon(mesh_a);
    let poly_b = ground_face_polygon(mesh_b);

    if poly_a.is_empty() && poly_b.is_empty() {
        return None;
    }

    let all_pts: Vec<(f64, f64)> = poly_a.iter().chain(poly_b.iter()).copied().collect();
    let (min_x, min_y, max_x, max_y) = bbox(&all_pts)?;

    let nx = ((max_x - min_x) / resolution).ceil() as usize + 1;
    let ny = ((max_y - min_y) / resolution).ceil() as usize + 1;

    if nx == 0 || ny == 0 || nx * ny > 10_000_000 {
        let area_a = polygon_area(&poly_a);
        let area_b = polygon_area(&poly_b);
        return if area_a + area_b > 0.0 {
            Some(area_a.min(area_b) / area_a.max(area_b))
        } else {
            None
        };
    }

    let mut count_a = 0u64;
    let mut count_b = 0u64;
    let mut count_both = 0u64;

    for ix in 0..nx {
        let px = min_x + (ix as f64 + 0.5) * resolution;
        for iy in 0..ny {
            let py = min_y + (iy as f64 + 0.5) * resolution;
            let in_a = point_in_polygon(px, py, &poly_a);
            let in_b = point_in_polygon(px, py, &poly_b);
            if in_a {
                count_a += 1;
            }
            if in_b {
                count_b += 1;
            }
            if in_a && in_b {
                count_both += 1;
            }
        }
    }

    let union = count_a + count_b - count_both;
    if union == 0 {
        None
    } else {
        Some(count_both as f64 / union as f64)
    }
}
