use anyhow::{Context, Result};
use lod2_common::mesh::{BuildingGeometry, Face, Mesh, SemanticSurface, SurfaceType};
use lod2_common::polygon::AttributeMap;
use nalgebra::Point3;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct CityJsonHeader {
    pub scale: [f64; 3],
    pub translate: [f64; 3],
    pub epsg: Option<String>,
}

impl Default for CityJsonHeader {
    fn default() -> Self {
        Self {
            scale: [0.001, 0.001, 0.001],
            translate: [0.0, 0.0, 0.0],
            epsg: None,
        }
    }
}

pub fn read_cityjsonl(path: &Path) -> Result<Vec<BuildingGeometry>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut header = CityJsonHeader::default();
    let mut buildings = Vec::new();

    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("line {}", line_no + 1))?;
        let val: Value = serde_json::from_str(&line)
            .with_context(|| format!("parsing JSON at line {}", line_no + 1))?;

        match val.get("type").and_then(|t| t.as_str()) {
            Some("CityJSON") => {
                if let Some(t) = val.get("transform") {
                    if let Some(s) = t.get("scale").and_then(|v| parse_f64_array(v)) {
                        header.scale = s;
                    }
                    if let Some(tr) = t.get("translate").and_then(|v| parse_f64_array(v)) {
                        header.translate = tr;
                    }
                }
                if let Some(rs) = val
                    .get("metadata")
                    .and_then(|m| m.get("referenceSystem"))
                    .and_then(|r| r.as_str())
                {
                    if let Some(code) = rs.rsplit('/').next() {
                        header.epsg = Some(code.to_string());
                    }
                }
            }
            Some("CityJSONFeature") => {
                if let Some(bldg) = parse_feature(&val, &header) {
                    buildings.push(bldg);
                }
            }
            _ => {}
        }
    }
    Ok(buildings)
}

fn parse_f64_array(val: &Value) -> Option<[f64; 3]> {
    let arr = val.as_array()?;
    if arr.len() < 3 {
        return None;
    }
    Some([arr[0].as_f64()?, arr[1].as_f64()?, arr[2].as_f64()?])
}

fn parse_feature(val: &Value, header: &CityJsonHeader) -> Option<BuildingGeometry> {
    let id = val.get("id")?.as_str()?.to_string();
    let raw_verts = val.get("vertices")?.as_array()?;

    let vertices: Vec<Point3<f64>> = raw_verts
        .iter()
        .filter_map(|v| {
            let arr = v.as_array()?;
            if arr.len() < 3 {
                return None;
            }
            let x = arr[0].as_f64()? * header.scale[0] + header.translate[0];
            let y = arr[1].as_f64()? * header.scale[1] + header.translate[1];
            let z = arr[2].as_f64()? * header.scale[2] + header.translate[2];
            Some(Point3::new(x, y, z))
        })
        .collect();

    let city_objects = val.get("CityObjects")?.as_object()?;

    let mut attributes = AttributeMap::new();
    let mut best_mesh: Option<Mesh> = None;
    let mut h_ground = f64::MAX;

    for (_co_id, co) in city_objects {
        if let Some(attrs) = co.get("attributes").and_then(|a| a.as_object()) {
            for (k, v) in attrs {
                match v {
                    Value::Number(n) => {
                        if let Some(f) = n.as_f64() {
                            attributes.insert_float(k, f);
                        }
                    }
                    Value::String(s) => {
                        attributes.insert_string(k, s);
                    }
                    Value::Bool(b) => {
                        attributes.insert_bool(k, *b);
                    }
                    _ => {}
                }
            }
        }

        let geoms = match co.get("geometry").and_then(|g| g.as_array()) {
            Some(g) => g,
            None => continue,
        };

        for geom_obj in geoms {
            let lod = geom_obj
                .get("lod")
                .and_then(|l| l.as_str().or_else(|| l.as_f64().map(|_| "")).and_then(|_| l.as_str()))
                .unwrap_or("");

            let geom_type = geom_obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
            let boundaries = match geom_obj.get("boundaries") {
                Some(b) => b,
                None => continue,
            };

            let (surfaces_def, values_def) = if let Some(sem) = geom_obj.get("semantics") {
                (
                    sem.get("surfaces").and_then(|s| s.as_array()),
                    sem.get("values"),
                )
            } else {
                (None, None)
            };

            let mut mesh = Mesh::new();
            for v in &vertices {
                mesh.add_vertex(*v);
            }

            let sem_surfaces = parse_semantic_surfaces(surfaces_def);
            for s in &sem_surfaces {
                mesh.add_semantic(s.clone());
            }

            let (shell, shell_values) = if geom_type == "Solid" {
                let shells = boundaries.as_array().unwrap_or(&Vec::new()).clone();
                let outer_shell = shells.first().and_then(|s| s.as_array());
                let outer_vals = values_def
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.first())
                    .and_then(|v| v.as_array());
                (outer_shell.cloned(), outer_vals.cloned())
            } else {
                (
                    boundaries.as_array().cloned(),
                    values_def.and_then(|v| v.as_array()).cloned(),
                )
            };

            if let Some(faces) = shell {
                for (fi, face_val) in faces.iter().enumerate() {
                    let rings = match face_val.as_array() {
                        Some(r) => r,
                        None => continue,
                    };
                    let outer = match rings.first().and_then(|r| r.as_array()) {
                        Some(r) => r,
                        None => continue,
                    };
                    let indices: Vec<u32> = outer
                        .iter()
                        .filter_map(|i| i.as_u64().map(|v| v as u32))
                        .collect();
                    if indices.len() < 3 {
                        continue;
                    }

                    let sem_idx = shell_values
                        .as_ref()
                        .and_then(|sv| sv.get(fi))
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize);

                    let face = if let Some(si) = sem_idx {
                        Face::new(indices).with_semantic(si)
                    } else {
                        Face::new(indices)
                    };

                    for &idx in &face.indices {
                        let z = vertices.get(idx as usize).map(|v| v.z).unwrap_or(f64::MAX);
                        if z < h_ground {
                            h_ground = z;
                        }
                    }

                    mesh.add_face(face);
                }
            }

            if !mesh.faces.is_empty()
                && (lod.starts_with("2") || best_mesh.is_none())
            {
                best_mesh = Some(mesh);
            }
        }
    }

    let mut bldg = BuildingGeometry::new(&id);
    bldg.h_ground = if h_ground < f64::MAX { h_ground } else { 0.0 };
    bldg.attributes = attributes;
    bldg.lod22 = best_mesh;
    Some(bldg)
}

fn parse_semantic_surfaces(surfaces: Option<&Vec<Value>>) -> Vec<SemanticSurface> {
    let surfaces = match surfaces {
        Some(s) => s,
        None => return Vec::new(),
    };
    surfaces
        .iter()
        .map(|s| {
            let stype = s
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("Unknown");
            let surface_type = match stype {
                "GroundSurface" => SurfaceType::GroundSurface,
                "WallSurface" => SurfaceType::WallSurface,
                "RoofSurface" => SurfaceType::RoofSurface,
                "ClosureSurface" => SurfaceType::ClosureSurface,
                _ => SurfaceType::RoofSurface,
            };
            let slope = s.get("rf_slope").and_then(|v| v.as_f64());
            let azimuth = s.get("rf_azimuth").and_then(|v| v.as_f64());
            SemanticSurface {
                surface_type,
                on_footprint_edge: s.get("on_footprint_edge").and_then(|v| v.as_bool()),
                azimuth,
                slope,
                h_roof_50p: None,
                h_roof_70p: None,
                h_roof_min: None,
                h_roof_max: None,
            }
        })
        .collect()
}
