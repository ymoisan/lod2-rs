use anyhow::{Context, Result};
use gdal::vector::Geometry;
use rstar::{RTree, AABB};
use std::path::Path;
use tracing::info;

const CLASS_BUILDING: u8 = 6;

pub struct BuildingPointIndex {
    tree: RTree<[f64; 2]>,
}

impl BuildingPointIndex {
    pub fn from_laz(path: &Path) -> Result<Self> {
        info!("Reading classified points from {} …", path.display());
        let mut reader = las::Reader::from_path(path)
            .with_context(|| format!("opening {}", path.display()))?;

        let build_class = las::point::Classification::new(CLASS_BUILDING).unwrap_or_default();

        let mut pts: Vec<[f64; 2]> = Vec::new();
        let mut total = 0usize;
        for wrapped in reader.points() {
            let pt = wrapped?;
            total += 1;
            if pt.classification == build_class {
                pts.push([pt.x, pt.y]);
            }
        }
        info!(
            "{total} points total, {} class-6 (building)",
            pts.len()
        );

        let tree = RTree::bulk_load(pts);
        Ok(Self { tree })
    }

    /// Count class-6 building points inside a GDAL polygon/multipolygon geometry.
    pub fn count_in_polygon(&self, geom: &Geometry) -> usize {
        let rings = all_exterior_rings(geom);
        let mut total = 0;
        for verts in &rings {
            if verts.len() < 3 {
                continue;
            }
            let (min_x, min_y, max_x, max_y) = bbox(verts);
            let envelope = AABB::from_corners([min_x, min_y], [max_x, max_y]);
            total += self
                .tree
                .locate_in_envelope(&envelope)
                .filter(|pt| point_in_polygon(pt[0], pt[1], verts))
                .count();
        }
        total
    }
}

/// Extract exterior ring coords from a Polygon or each sub-polygon of a MultiPolygon.
fn all_exterior_rings(geom: &Geometry) -> Vec<Vec<(f64, f64)>> {
    use gdal::vector::OGRwkbGeometryType::*;
    let gt = geom.geometry_type();
    match gt {
        wkbPolygon | wkbPolygon25D | wkbPolygonM | wkbPolygonZM => {
            if geom.geometry_count() == 0 {
                return vec![];
            }
            let ring = geom.get_geometry(0);
            vec![ring.get_point_vec().into_iter().map(|(x, y, _)| (x, y)).collect()]
        }
        wkbMultiPolygon | wkbMultiPolygon25D => {
            let mut rings = Vec::new();
            for i in 0..geom.geometry_count() {
                let poly = geom.get_geometry(i);
                if poly.geometry_count() > 0 {
                    let ring = poly.get_geometry(0);
                    rings.push(ring.get_point_vec().into_iter().map(|(x, y, _)| (x, y)).collect());
                }
            }
            rings
        }
        _ => vec![],
    }
}

fn bbox(verts: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &(x, y) in verts {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    (min_x, min_y, max_x, max_y)
}

/// Ray-casting point-in-polygon test.
fn point_in_polygon(px: f64, py: f64, verts: &[(f64, f64)]) -> bool {
    let n = verts.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = verts[i];
        let (xj, yj) = verts[j];
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}
