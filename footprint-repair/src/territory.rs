//! Per-site Voronoi cells as intersection of closed half-planes (perpendicular bisectors)
//! with the global axis-aligned bbox. Half-planes are applied via exact convex polygon clipping
//! (no giant GDAL triangles). `intersection` for the final repair step still uses GEOS.

use anyhow::{Context, Result};
use gdal::vector::Geometry;
use gdal::vector::OGRwkbGeometryType::{wkbLinearRing, wkbMultiPolygon, wkbMultiPolygon25D, wkbPolygon, wkbPolygon25D};

use crate::points;

/// Axis-aligned bounds, expanded by `margin` (metres), from exterior vertices.
pub fn global_bounds_from_vertices<I>(footprints: I, margin: f64) -> (f64, f64, f64, f64)
where
    I: Iterator<Item = Vec<(f64, f64)>>,
{
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for verts in footprints {
        if verts.is_empty() {
            continue;
        }
        let (a, b, c, d) = points::bbox(&verts);
        min_x = min_x.min(a);
        min_y = min_y.min(b);
        max_x = max_x.max(c);
        max_y = max_y.max(d);
    }
    if !min_x.is_finite() {
        return (-1.0, -1.0, 1.0, 1.0);
    }
    (
        min_x - margin,
        min_y - margin,
        max_x + margin,
        max_y + margin,
    )
}

/// Bounding rectangle as a polygon geometry (CCW ring).
pub fn bbox_polygon(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Result<Geometry> {
    let mut ring = Geometry::empty(wkbLinearRing).context("empty ring")?;
    ring.add_point_2d((min_x, min_y));
    ring.add_point_2d((max_x, min_y));
    ring.add_point_2d((max_x, max_y));
    ring.add_point_2d((min_x, max_y));
    ring.add_point_2d((min_x, min_y));
    let mut poly = Geometry::empty(wkbPolygon).context("empty polygon")?;
    poly.add_geometry(ring).context("add ring")?;
    Ok(poly)
}

/// Clip convex polygon vertices (CCW) to closed half-plane `a*x + b*y + c <= 0`.
fn clip_convex_to_halfplane(verts: &[(f64, f64)], a: f64, b: f64, c: f64) -> Vec<(f64, f64)> {
    let n = verts.len();
    if n == 0 {
        return vec![];
    }
    let is_in = |x: f64, y: f64| a * x + b * y + c <= 1e-9;
    let intersect_seg = |p1: (f64, f64), p2: (f64, f64)| -> Option<(f64, f64)> {
        let dx = p2.0 - p1.0;
        let dy = p2.1 - p1.1;
        let den = a * dx + b * dy;
        if den.abs() < 1e-15 {
            return None;
        }
        let t = -(a * p1.0 + b * p1.1 + c) / den;
        if t < -1e-9 || t > 1.0 + 1e-9 {
            return None;
        }
        Some((p1.0 + t * dx, p1.1 + t * dy))
    };

    let mut out = Vec::new();
    for i in 0..n {
        let p1 = verts[i];
        let p2 = verts[(i + 1) % n];
        let in1 = is_in(p1.0, p1.1);
        let in2 = is_in(p2.0, p2.1);
        if in1 && in2 {
            out.push(p2);
        } else if in1 && !in2 {
            if let Some(ip) = intersect_seg(p1, p2) {
                out.push(ip);
            }
        } else if !in1 && in2 {
            if let Some(ip) = intersect_seg(p1, p2) {
                out.push(ip);
            }
            out.push(p2);
        }
    }
    out
}

fn verts_to_polygon(verts: &[(f64, f64)]) -> Result<Geometry> {
    if verts.len() < 3 {
        anyhow::bail!("need >= 3 vertices");
    }
    let mut ring = Geometry::empty(wkbLinearRing).context("empty ring")?;
    for &(x, y) in verts {
        ring.add_point_2d((x, y));
    }
    ring.add_point_2d((verts[0].0, verts[0].1));
    let mut poly = Geometry::empty(wkbPolygon).context("empty polygon")?;
    poly.add_geometry(ring).context("add ring")?;
    Ok(poly)
}

/// Voronoi cell for site `fi`: global bbox clipped by all bisector half-planes (c_i vs c_j).
pub fn voronoi_cell_for_site(
    centroids: &[(f64, f64)],
    fi: usize,
    bounds: (f64, f64, f64, f64),
) -> Result<Geometry> {
    let (bx0, by0, bx1, by1) = bounds;
    // CCW rectangle
    let mut verts = vec![
        (bx0, by0),
        (bx1, by0),
        (bx1, by1),
        (bx0, by1),
    ];

    let c_i = centroids[fi];

    for (j, &c_j) in centroids.iter().enumerate() {
        if j == fi {
            continue;
        }
        let vx = c_j.0 - c_i.0;
        let vy = c_j.1 - c_i.1;
        let len2 = vx * vx + vy * vy;
        if len2 < 1e-24 {
            continue;
        }
        let mx = (c_i.0 + c_j.0) * 0.5;
        let my = (c_i.1 + c_j.1) * 0.5;
        // Region closer to c_i than c_j: vx*(x-mx)+vy*(y-my) <= 0
        // => vx*x + vy*y + (-vx*mx - vy*my) <= 0
        let a = vx;
        let b = vy;
        let c = -vx * mx - vy * my;
        verts = clip_convex_to_halfplane(&verts, a, b, c);
        if verts.len() < 3 {
            return Err(anyhow::anyhow!(
                "Voronoi clip emptied polygon for site {fi} vs {j}"
            ));
        }
    }

    verts_to_polygon(&verts)
}

/// If MultiPolygon, keep the polygon with largest area; otherwise clone.
pub fn keep_largest_polygon(geom: &Geometry) -> Geometry {
    let gt = geom.geometry_type();
    let is_multi = gt == wkbMultiPolygon || gt == wkbMultiPolygon25D;
    let is_poly = gt == wkbPolygon || gt == wkbPolygon25D;
    if is_poly || !is_multi {
        return Geometry::from_wkt(&geom.wkt().unwrap_or_default())
            .unwrap_or_else(|_| geom.clone());
    }
    let n = geom.geometry_count();
    if n == 0 {
        return Geometry::from_wkt(&geom.wkt().unwrap_or_default())
            .unwrap_or_else(|_| geom.clone());
    }
    let mut best_idx = 0usize;
    let mut best_area = 0.0_f64;
    for i in 0..n {
        let sub = geom.get_geometry(i);
        let a = sub.area();
        if a > best_area {
            best_area = a;
            best_idx = i;
        }
    }
    let sub = geom.get_geometry(best_idx);
    Geometry::from_wkt(&sub.wkt().unwrap_or_default()).unwrap_or_else(|_| geom.clone())
}
