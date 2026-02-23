use crate::polygon::AttributeMap;
use nalgebra::{Point3, Vector3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceType {
    GroundSurface,
    WallSurface,
    RoofSurface,
    ClosureSurface,
}

#[derive(Debug, Clone)]
pub struct SemanticSurface {
    pub surface_type: SurfaceType,
    pub on_footprint_edge: Option<bool>,
    pub azimuth: Option<f64>,
    pub slope: Option<f64>,
    pub h_roof_50p: Option<f64>,
    pub h_roof_70p: Option<f64>,
    pub h_roof_min: Option<f64>,
    pub h_roof_max: Option<f64>,
}

impl SemanticSurface {
    pub fn ground() -> Self {
        Self {
            surface_type: SurfaceType::GroundSurface,
            on_footprint_edge: None,
            azimuth: None, slope: None,
            h_roof_50p: None, h_roof_70p: None, h_roof_min: None, h_roof_max: None,
        }
    }

    pub fn wall(on_edge: bool) -> Self {
        Self {
            surface_type: SurfaceType::WallSurface,
            on_footprint_edge: Some(on_edge),
            azimuth: None, slope: None,
            h_roof_50p: None, h_roof_70p: None, h_roof_min: None, h_roof_max: None,
        }
    }

    pub fn roof() -> Self {
        Self {
            surface_type: SurfaceType::RoofSurface,
            on_footprint_edge: None,
            azimuth: None, slope: None,
            h_roof_50p: None, h_roof_70p: None, h_roof_min: None, h_roof_max: None,
        }
    }

    pub fn roof_with_stats(slope: f64, azimuth: f64) -> Self {
        Self {
            surface_type: SurfaceType::RoofSurface,
            on_footprint_edge: None,
            azimuth: Some(azimuth),
            slope: Some(slope),
            h_roof_50p: None, h_roof_70p: None, h_roof_min: None, h_roof_max: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Face {
    pub indices: Vec<u32>,
    pub semantic_index: Option<usize>,
}

impl Face {
    pub fn new(indices: Vec<u32>) -> Self {
        Self { indices, semantic_index: None }
    }

    pub fn with_semantic(mut self, idx: usize) -> Self {
        self.semantic_index = Some(idx);
        self
    }
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Point3<f64>>,
    pub faces: Vec<Face>,
    pub semantics: Vec<SemanticSurface>,
}

impl Mesh {
    pub fn new() -> Self {
        Self { vertices: Vec::new(), faces: Vec::new(), semantics: Vec::new() }
    }

    pub fn add_vertex(&mut self, v: Point3<f64>) -> u32 {
        let idx = self.vertices.len() as u32;
        self.vertices.push(v);
        idx
    }

    pub fn add_face(&mut self, face: Face) {
        self.faces.push(face);
    }

    pub fn add_semantic(&mut self, semantic: SemanticSurface) -> usize {
        let idx = self.semantics.len();
        self.semantics.push(semantic);
        idx
    }

    pub fn face_normal(&self, face_idx: usize) -> Option<Vector3<f64>> {
        let face = &self.faces[face_idx];
        if face.indices.len() < 3 {
            return None;
        }
        let p0 = &self.vertices[face.indices[0] as usize];
        let p1 = &self.vertices[face.indices[1] as usize];
        let p2 = &self.vertices[face.indices[2] as usize];
        let v1 = p1 - p0;
        let v2 = p2 - p0;
        let n = v1.cross(&v2);
        let len = n.norm();
        if len < 1e-15 {
            return None;
        }
        Some(n / len)
    }

    pub fn compute_volume(&self) -> f64 {
        let mut volume = 0.0;
        for face in &self.faces {
            if face.indices.len() < 3 {
                continue;
            }
            let p0 = &self.vertices[face.indices[0] as usize];
            for i in 1..(face.indices.len() - 1) {
                let p1 = &self.vertices[face.indices[i] as usize];
                let p2 = &self.vertices[face.indices[i + 1] as usize];
                volume += p0.coords.dot(&p1.coords.cross(&p2.coords));
            }
        }
        (volume / 6.0).abs()
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct BuildingGeometry {
    pub id: String,
    pub lod12: Option<Mesh>,
    pub lod22: Option<Mesh>,
    pub h_ground: f64,
    pub attributes: AttributeMap,
}

impl BuildingGeometry {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            lod12: None,
            lod22: None,
            h_ground: 0.0,
            attributes: AttributeMap::new(),
        }
    }

    pub fn best_lod(&self) -> Option<&Mesh> {
        self.lod22.as_ref().or(self.lod12.as_ref())
    }

    pub fn geographic_extent(&self) -> Option<[f64; 6]> {
        let mesh = self.best_lod()?;
        if mesh.vertices.is_empty() {
            return None;
        }
        let mut min = [f64::MAX; 3];
        let mut max = [f64::MIN; 3];
        for v in &mesh.vertices {
            min[0] = min[0].min(v.x);
            min[1] = min[1].min(v.y);
            min[2] = min[2].min(v.z);
            max[0] = max[0].max(v.x);
            max[1] = max[1].max(v.y);
            max[2] = max[2].max(v.z);
        }
        Some([min[0], min[1], min[2], max[0], max[1], max[2]])
    }
}
