use crate::polygon::{AttributeMap, Footprint, LinearRing, Polygon3D};
use gdal::vector::LayerAccess;
use gdal::Dataset;
use nalgebra::Point3;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VectorError {
    #[error("GDAL error: {0}")]
    GdalError(#[from] gdal::errors::GdalError),
    #[error("File not found: {0}")]
    FileNotFound(String),
}

pub struct VectorReader;

impl VectorReader {
    pub fn read_footprints(path: &Path) -> Result<Vec<Footprint>, VectorError> {
        if !path.exists() {
            return Err(VectorError::FileNotFound(path.display().to_string()));
        }
        let dataset = Dataset::open(path)?;
        let mut layer = dataset.layer(0)?;
        let mut footprints = Vec::new();
        let mut fid_counter = 0u64;

        for feature in layer.features() {
            let id = feature
                .fid()
                .map(|f| f.to_string())
                .unwrap_or_else(|| fid_counter.to_string());

            let geometry = match feature.geometry() {
                Some(g) => g,
                None => continue,
            };

            let polygon = Self::geometry_to_polygon3d(geometry)?;
            if polygon.is_empty() {
                continue;
            }

            let mut attributes = AttributeMap::new();
            for (name, value) in feature.fields() {
                match value {
                    Some(gdal::vector::FieldValue::IntegerValue(v)) => {
                        attributes.insert_int(name, v as i64);
                    }
                    Some(gdal::vector::FieldValue::Integer64Value(v)) => {
                        attributes.insert_int(name, v);
                    }
                    Some(gdal::vector::FieldValue::RealValue(v)) => {
                        attributes.insert_float(name, v);
                    }
                    Some(gdal::vector::FieldValue::StringValue(v)) => {
                        attributes.insert_string(name, v);
                    }
                    _ => {}
                }
            }

            footprints.push(Footprint::new(id, polygon).with_attributes(attributes));
            fid_counter += 1;
        }

        tracing::info!("Read {} footprints from {}", footprints.len(), path.display());
        Ok(footprints)
    }

    pub fn read_crs(path: &Path) -> Result<Option<String>, VectorError> {
        if !path.exists() {
            return Err(VectorError::FileNotFound(path.display().to_string()));
        }
        let dataset = Dataset::open(path)?;
        let layer = dataset.layer(0)?;
        if let Some(srs) = layer.spatial_ref() {
            if let Ok(epsg) = srs.auth_code() {
                tracing::info!("Found CRS: EPSG:{}", epsg);
                return Ok(Some(epsg.to_string()));
            }
        }
        Ok(None)
    }

    fn geometry_to_polygon3d(geometry: &gdal::vector::Geometry) -> Result<Polygon3D, VectorError> {
        let geom_type = geometry.geometry_type();
        match geom_type {
            gdal::vector::OGRwkbGeometryType::wkbPolygon
            | gdal::vector::OGRwkbGeometryType::wkbPolygon25D
            | gdal::vector::OGRwkbGeometryType::wkbPolygonM
            | gdal::vector::OGRwkbGeometryType::wkbPolygonZM => {
                Self::polygon_to_polygon3d(geometry)
            }
            gdal::vector::OGRwkbGeometryType::wkbMultiPolygon
            | gdal::vector::OGRwkbGeometryType::wkbMultiPolygon25D => {
                if geometry.geometry_count() > 0 {
                    Self::polygon_to_polygon3d(&geometry.get_geometry(0))
                } else {
                    Ok(Polygon3D::default())
                }
            }
            _ => Ok(Polygon3D::default()),
        }
    }

    fn polygon_to_polygon3d(geometry: &gdal::vector::Geometry) -> Result<Polygon3D, VectorError> {
        let ring_count = geometry.geometry_count();
        if ring_count == 0 {
            return Ok(Polygon3D::default());
        }
        let ext_ring = geometry.get_geometry(0);
        let exterior = Self::ring_to_linear_ring(&ext_ring);
        let mut interiors = Vec::new();
        for i in 1..ring_count {
            interiors.push(Self::ring_to_linear_ring(&geometry.get_geometry(i)));
        }
        Ok(Polygon3D::with_interiors(exterior, interiors))
    }

    fn ring_to_linear_ring(geometry: &gdal::vector::Geometry) -> LinearRing {
        let points = geometry.get_point_vec();
        let vertices: Vec<Point3<f64>> = points
            .into_iter()
            .map(|(x, y, z)| Point3::new(x, y, z))
            .collect();
        LinearRing::from_vertices(vertices)
    }
}
