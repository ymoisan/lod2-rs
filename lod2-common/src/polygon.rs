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
    pub fn new(id: impl Into<String>, polygon: Polygon3D) -> Self {
        Self { id: id.into(), polygon, attributes: AttributeMap::new() }
    }

    pub fn with_attributes(mut self, attributes: AttributeMap) -> Self {
        self.attributes = attributes;
        self
    }

    pub fn contains_2d(&self, x: f64, y: f64) -> bool {
        self.polygon.contains_2d(x, y)
    }
}
