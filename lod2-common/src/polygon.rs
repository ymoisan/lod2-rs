use nalgebra::Point3;
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct LinearRing {
    pub vertices: Vec<Point3<f64>>,
}

impl LinearRing {
    pub fn from_vertices(vertices: Vec<Point3<f64>>) -> Self {
        Self { vertices }
    }

    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Polygon3D {
    pub exterior: LinearRing,
    pub interiors: Vec<LinearRing>,
}

impl Polygon3D {
    pub fn new(exterior: LinearRing) -> Self {
        Self { exterior, interiors: Vec::new() }
    }

    pub fn with_interiors(exterior: LinearRing, interiors: Vec<LinearRing>) -> Self {
        Self { exterior, interiors }
    }

    pub fn is_empty(&self) -> bool {
        self.exterior.is_empty()
    }

    /// 2D point-in-polygon test (ray casting).
    pub fn contains_2d(&self, x: f64, y: f64) -> bool {
        if !Self::ring_contains_2d(&self.exterior, x, y) {
            return false;
        }
        for hole in &self.interiors {
            if Self::ring_contains_2d(hole, x, y) {
                return false;
            }
        }
        true
    }

    fn ring_contains_2d(ring: &LinearRing, x: f64, y: f64) -> bool {
        let n = ring.vertices.len();
        if n < 3 {
            return false;
        }
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let vi = &ring.vertices[i];
            let vj = &ring.vertices[j];
            if ((vi.y > y) != (vj.y > y))
                && (x < (vj.x - vi.x) * (y - vi.y) / (vj.y - vi.y) + vi.x)
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Check if a 2D point falls inside any interior ring (courtyard/hole).
    pub fn is_in_hole_2d(&self, x: f64, y: f64) -> bool {
        self.interiors.iter().any(|hole| Self::ring_contains_2d(hole, x, y))
    }

    /// 2D bounding box [min_x, min_y, max_x, max_y].
    pub fn bbox_2d(&self) -> [f64; 4] {
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        for v in &self.exterior.vertices {
            min_x = min_x.min(v.x);
            min_y = min_y.min(v.y);
            max_x = max_x.max(v.x);
            max_y = max_y.max(v.y);
        }
        [min_x, min_y, max_x, max_y]
    }

    /// Compute 2D area (shoelace formula).
    pub fn area_2d(&self) -> f64 {
        Self::ring_area_2d(&self.exterior).abs()
    }

    fn ring_area_2d(ring: &LinearRing) -> f64 {
        let n = ring.vertices.len();
        if n < 3 {
            return 0.0;
        }
        let mut area = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            area += ring.vertices[i].x * ring.vertices[j].y;
            area -= ring.vertices[j].x * ring.vertices[i].y;
        }
        area * 0.5
    }

    /// Centroid (2D).
    pub fn centroid_2d(&self) -> (f64, f64) {
        let n = self.exterior.vertices.len();
        if n == 0 {
            return (0.0, 0.0);
        }
        let mut cx = 0.0;
        let mut cy = 0.0;
        for v in &self.exterior.vertices {
            cx += v.x;
            cy += v.y;
        }
        (cx / n as f64, cy / n as f64)
    }

    /// Ensure the exterior ring is CCW and interior rings are CW.
    /// This is required for consistent face winding in the mesh builder:
    /// ground face is reversed (CW = normal down), walls connect ground to roof,
    /// and roof CDT faces are CCW.  If the exterior ring is CW, the wall-roof
    /// shared edges would go the same direction, breaking manifold topology.
    pub fn ensure_ccw(&mut self) {
        // Remove near-duplicate consecutive vertices (1cm tolerance) before
        // winding check — prevents degenerate wall quads after CityJSON
        // vertex quantization.
        Self::dedup_ring(&mut self.exterior, 0.01);
        for hole in &mut self.interiors {
            Self::dedup_ring(hole, 0.01);
        }
        if Self::ring_area_2d(&self.exterior) < 0.0 {
            self.exterior.vertices.reverse();
        }
        for hole in &mut self.interiors {
            if Self::ring_area_2d(hole) > 0.0 {
                hole.vertices.reverse();
            }
        }
    }

    fn dedup_ring(ring: &mut LinearRing, tolerance: f64) {
        if ring.len() < 4 {
            return;
        }
        let tol_sq = tolerance * tolerance;
        let mut keep = Vec::with_capacity(ring.len());
        keep.push(ring.vertices[0]);
        for v in ring.vertices.iter().skip(1) {
            let prev = keep.last().unwrap();
            let dx = v.x - prev.x;
            let dy = v.y - prev.y;
            if dx * dx + dy * dy > tol_sq {
                keep.push(*v);
            }
        }
        // Check last vs first (closing vertex)
        if keep.len() > 1 {
            let first = keep[0];
            let last = *keep.last().unwrap();
            let dx = last.x - first.x;
            let dy = last.y - first.y;
            if dx * dx + dy * dy <= tol_sq {
                keep.pop();
            }
        }
        ring.vertices = keep;
    }
}

#[derive(Debug, Clone)]
pub enum AttributeValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

#[derive(Debug, Clone, Default)]
pub struct AttributeMap(pub HashMap<String, AttributeValue>);

impl AttributeMap {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn insert_int(&mut self, key: impl Into<String>, val: i64) {
        self.0.insert(key.into(), AttributeValue::Int(val));
    }

    pub fn insert_float(&mut self, key: impl Into<String>, val: f64) {
        self.0.insert(key.into(), AttributeValue::Float(val));
    }

    pub fn insert_string(&mut self, key: impl Into<String>, val: impl Into<String>) {
        self.0.insert(key.into(), AttributeValue::String(val.into()));
    }

    pub fn insert_bool(&mut self, key: impl Into<String>, val: bool) {
        self.0.insert(key.into(), AttributeValue::Bool(val));
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &AttributeValue)> {
        self.0.iter()
    }
}

#[derive(Debug, Clone)]
pub struct Footprint {
    pub id: String,
    pub polygon: Polygon3D,
    pub attributes: AttributeMap,
}

impl Footprint {
    pub fn new(id: impl Into<String>, mut polygon: Polygon3D) -> Self {
        polygon.ensure_ccw();
        Self { id: id.into(), polygon, attributes: AttributeMap::new() }
    }

    pub fn with_attributes(mut self, attributes: AttributeMap) -> Self {
        self.attributes = attributes;
        self
    }

    pub fn contains_2d(&self, x: f64, y: f64) -> bool {
        self.polygon.contains_2d(x, y)
    }

    /// Minimum 2D distance from point (x, y) to all boundary edges
    /// (exterior ring + interior rings).
    pub fn min_distance_to_boundary(&self, x: f64, y: f64) -> f64 {
        let mut min_dist = f64::MAX;
        for ring in std::iter::once(&self.polygon.exterior).chain(&self.polygon.interiors) {
            let n = ring.vertices.len();
            if n < 2 {
                continue;
            }
            for i in 0..n {
                let j = (i + 1) % n;
                let ax = ring.vertices[i].x;
                let ay = ring.vertices[i].y;
                let bx = ring.vertices[j].x;
                let by = ring.vertices[j].y;
                let d = point_to_segment_dist_2d(x, y, ax, ay, bx, by);
                if d < min_dist {
                    min_dist = d;
                }
            }
        }
        min_dist
    }
}

/// 2D point-to-segment distance.
fn point_to_segment_dist_2d(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = bx - ax;
    let dy = by - ay;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-30 {
        return ((px - ax).powi(2) + (py - ay).powi(2)).sqrt();
    }
    let t = ((px - ax) * dx + (py - ay) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let cx = ax + t * dx;
    let cy = ay + t * dy;
    ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
}
