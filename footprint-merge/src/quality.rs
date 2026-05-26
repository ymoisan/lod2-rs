use gdal::vector::Geometry;

/// Rectangularity = polygon_area / MABR_area.  1.0 for a perfect rectangle,
/// ~0.5 for a triangle, 0.6-0.75 for a truncated building.
pub fn rectangularity(geom: &Geometry) -> f64 {
    let area = geom.area();
    if area <= 0.0 {
        return 0.0;
    }
    let verts = exterior_vertices(geom);
    if verts.len() < 3 {
        return 0.0;
    }
    let mabr_area = min_area_bounding_rect_area(&verts);
    if mabr_area <= 0.0 {
        return 0.0;
    }
    (area / mabr_area).min(1.0)
}

/// Extract exterior ring vertices from a Polygon or the first sub-polygon of a MultiPolygon.
pub fn exterior_vertices(geom: &Geometry) -> Vec<(f64, f64)> {
    use gdal::vector::OGRwkbGeometryType::*;
    let gt = geom.geometry_type();
    match gt {
        wkbMultiPolygon | wkbMultiPolygon25D => {
            if geom.geometry_count() == 0 {
                return Vec::new();
            }
            let poly = geom.get_geometry(0);
            if poly.geometry_count() == 0 {
                return Vec::new();
            }
            let ring = poly.get_geometry(0);
            ring.get_point_vec().into_iter().map(|(x, y, _)| (x, y)).collect()
        }
        _ => {
            if geom.geometry_count() == 0 {
                return Vec::new();
            }
            let ring = geom.get_geometry(0);
            ring.get_point_vec().into_iter().map(|(x, y, _)| (x, y)).collect()
        }
    }
}

fn min_area_bounding_rect_area(points: &[(f64, f64)]) -> f64 {
    let hull = convex_hull(points);
    if hull.len() < 3 {
        return 0.0;
    }

    let n = hull.len();
    let mut best_area = f64::INFINITY;

    for i in 0..n {
        let j = (i + 1) % n;
        let ex = hull[j].0 - hull[i].0;
        let ey = hull[j].1 - hull[i].1;
        let len = (ex * ex + ey * ey).sqrt();
        if len < 1e-10 {
            continue;
        }
        let ux = ex / len;
        let uy = ey / len;
        let vx = -uy;
        let vy = ux;

        let mut min_u = f64::INFINITY;
        let mut max_u = f64::NEG_INFINITY;
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for &(x, y) in &hull {
            let u = x * ux + y * uy;
            let v = x * vx + y * vy;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }

        let area = (max_u - min_u) * (max_v - min_v);
        if area < best_area {
            best_area = area;
        }
    }

    best_area
}

fn convex_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64)> = points.to_vec();
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(a.1.partial_cmp(&b.1).unwrap())
    });
    pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-10 && (a.1 - b.1).abs() < 1e-10);

    if pts.len() < 2 {
        return pts;
    }

    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| -> f64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };

    let mut lower: Vec<(f64, f64)> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2
            && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0
        {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<(f64, f64)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2
            && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0
        {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}
