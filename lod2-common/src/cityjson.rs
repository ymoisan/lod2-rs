use crate::mesh::{BuildingGeometry, Mesh, SurfaceType};
use crate::polygon::AttributeValue;
use serde_json::{json, Map, Value};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CityJsonError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

#[derive(Debug, Clone)]
pub struct CityJsonTransform {
    pub scale: [f64; 3],
    pub translate: [f64; 3],
}

impl Default for CityJsonTransform {
    fn default() -> Self {
        Self { scale: [0.001, 0.001, 0.001], translate: [0.0, 0.0, 0.0] }
    }
}

pub struct CityJsonWriter {
    writer: BufWriter<File>,
    transform: CityJsonTransform,
    reference_system: Option<String>,
    header_written: bool,
}

impl CityJsonWriter {
    pub fn new(path: &Path, transform: CityJsonTransform) -> Result<Self, CityJsonError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            transform,
            reference_system: None,
            header_written: false,
        })
    }

    pub fn with_reference_system(mut self, srs: impl Into<String>) -> Self {
        self.reference_system = Some(srs.into());
        self
    }

    pub fn write_header(&mut self) -> Result<(), CityJsonError> {
        let mut header = json!({
            "type": "CityJSON",
            "version": "2.0",
            "CityObjects": {},
            "vertices": [],
            "transform": {
                "scale": self.transform.scale,
                "translate": self.transform.translate
            }
        });

        if let Some(ref srs) = self.reference_system {
            let epsg = srs.trim_start_matches("EPSG:");
            header["metadata"] = json!({
                "referenceSystem": format!("https://www.opengis.net/def/crs/EPSG/0/{}", epsg),
                "referenceDate": chrono::Utc::now().format("%Y-%m-%d").to_string(),
                "identifier": "0"
            });
        }

        writeln!(self.writer, "{}", serde_json::to_string(&header)?)?;
        self.header_written = true;
        Ok(())
    }

    pub fn write_feature(&mut self, geom: &BuildingGeometry) -> Result<(), CityJsonError> {
        if !self.header_written {
            self.write_header()?;
        }
        let feature = self.build_feature(geom)?;
        writeln!(self.writer, "{}", serde_json::to_string(&feature)?)?;
        Ok(())
    }

    fn build_feature(&self, geom: &BuildingGeometry) -> Result<Value, CityJsonError> {
        let id = &geom.id;
        let mut vertices: Vec<[i64; 3]> = Vec::new();
        let mut city_objects: Map<String, Value> = Map::new();

        let mesh = match geom.best_lod() {
            Some(m) => m,
            None => {
                return Ok(json!({
                    "type": "CityJSONFeature", "id": id,
                    "CityObjects": {}, "vertices": []
                }));
            }
        };

        let mut attributes: Map<String, Value> = Map::new();
        for (key, value) in geom.attributes.iter() {
            let json_value = match value {
                AttributeValue::Null => Value::Null,
                AttributeValue::Bool(v) => Value::Bool(*v),
                AttributeValue::Int(v) => json!(*v),
                AttributeValue::Float(v) => json!(*v),
                AttributeValue::String(v) => Value::String(v.clone()),
            };
            attributes.insert(key.clone(), json_value);
        }

        for v in &mesh.vertices {
            let x = ((v.x - self.transform.translate[0]) / self.transform.scale[0]) as i64;
            let y = ((v.y - self.transform.translate[1]) / self.transform.scale[1]) as i64;
            let z = ((v.z - self.transform.translate[2]) / self.transform.scale[2]) as i64;
            vertices.push([x, y, z]);
        }

        let lod = if geom.lod22.is_some() { "2.2" } else { "1.2" };

        let boundaries = self.build_solid_boundaries(mesh);
        let (sem_surfaces, sem_values) = self.build_semantics(mesh);

        let geom_json = json!({
            "type": "Solid",
            "lod": lod,
            "boundaries": [boundaries],
            "semantics": { "surfaces": sem_surfaces, "values": [sem_values] }
        });

        let building_part_id = format!("{}-0", id);
        city_objects.insert(building_part_id.clone(), json!({
            "type": "BuildingPart",
            "parents": [id],
            "geometry": [geom_json]
        }));

        let footprint_bounds = self.build_footprint_boundaries(mesh);
        let mut building = json!({
            "type": "Building",
            "attributes": attributes,
            "children": [building_part_id]
        });
        if !footprint_bounds.is_empty() {
            building["geometry"] = json!([{
                "type": "MultiSurface", "lod": "0", "boundaries": footprint_bounds
            }]);
        }
        city_objects.insert(id.clone(), building);

        Ok(json!({
            "type": "CityJSONFeature",
            "id": id,
            "CityObjects": city_objects,
            "vertices": vertices
        }))
    }

    fn build_solid_boundaries(&self, mesh: &Mesh) -> Vec<Vec<Vec<u32>>> {
        let mut boundaries = Vec::new();
        for face in &mesh.faces {
            if face.indices.len() < 3 {
                continue;
            }
            let mut ring: Vec<u32> = face.indices.clone();
            if !is_ccw_2d(mesh, &face.indices) {
                ring.reverse();
            }
            boundaries.push(vec![ring]);
        }
        boundaries
    }

    fn build_footprint_boundaries(&self, mesh: &Mesh) -> Vec<Vec<Vec<u32>>> {
        let mut boundaries = Vec::new();
        for face in &mesh.faces {
            let is_ground = face.semantic_index
                .and_then(|idx| mesh.semantics.get(idx))
                .map(|s| s.surface_type == SurfaceType::GroundSurface)
                .unwrap_or(false);
            if is_ground && face.indices.len() >= 3 {
                let mut ring: Vec<u32> = face.indices.clone();
                if !is_ccw_2d(mesh, &face.indices) {
                    ring.reverse();
                }
                boundaries.push(vec![ring]);
            }
        }
        boundaries
    }

    fn build_semantics(&self, mesh: &Mesh) -> (Vec<Value>, Vec<Option<usize>>) {
        let mut surfaces = Vec::new();
        for semantic in &mesh.semantics {
            let mut surface = Map::new();
            surface.insert("type".to_string(), Value::String(match semantic.surface_type {
                SurfaceType::GroundSurface => "GroundSurface".to_string(),
                SurfaceType::WallSurface => "WallSurface".to_string(),
                SurfaceType::RoofSurface => "RoofSurface".to_string(),
                SurfaceType::ClosureSurface => "ClosureSurface".to_string(),
            }));
            if let Some(on_edge) = semantic.on_footprint_edge {
                surface.insert("on_footprint_edge".to_string(), Value::Bool(on_edge));
            }
            if semantic.surface_type == SurfaceType::RoofSurface {
                if let Some(az) = semantic.azimuth { surface.insert("rf_azimuth".to_string(), json!(az)); }
                if let Some(sl) = semantic.slope { surface.insert("rf_slope".to_string(), json!(sl)); }
            }
            surfaces.push(Value::Object(surface));
        }
        let values: Vec<Option<usize>> = mesh.faces.iter().map(|f| f.semantic_index).collect();
        (surfaces, values)
    }

    pub fn finish(mut self) -> Result<(), CityJsonError> {
        self.writer.flush()?;
        Ok(())
    }
}

fn is_ccw_2d(mesh: &Mesh, indices: &[u32]) -> bool {
    if indices.len() < 3 { return false; }
    let mut area = 0.0;
    let n = indices.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let vi = &mesh.vertices[indices[i] as usize];
        let vj = &mesh.vertices[indices[j] as usize];
        area += vi.x * vj.y - vj.x * vi.y;
    }
    area > 0.0
}
