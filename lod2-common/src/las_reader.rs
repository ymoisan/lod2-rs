use crate::point_cloud::PointCloud;
use las::Reader;
use nalgebra::Point3;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LasError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("LAS error: {0}")]
    LasError(#[from] las::Error),
    #[error("File not found: {0}")]
    FileNotFound(String),
}

pub struct LasReader;

impl LasReader {
    pub fn read_file(path: &Path) -> Result<PointCloud, LasError> {
        if !path.exists() {
            return Err(LasError::FileNotFound(path.display().to_string()));
        }
        let mut reader = Reader::from_path(path)?;
        let header = reader.header();
        let mut pc = PointCloud::with_capacity(header.number_of_points() as usize);
        for wrapped in reader.points() {
            let pt = wrapped?;
            pc.push(Point3::new(pt.x, pt.y, pt.z));
        }
        tracing::info!("Loaded {} points", pc.len());
        Ok(pc)
    }

    pub fn read_crs(path: &Path) -> Result<Option<String>, LasError> {
        if !path.exists() {
            return Err(LasError::FileNotFound(path.display().to_string()));
        }
        let reader = Reader::from_path(path)?;
        let header = reader.header();

        if let Ok(Some(geotiff)) = header.get_geotiff_crs() {
            for entry in &geotiff.entries {
                if entry.id == 2048 || entry.id == 3072 {
                    if let las::crs::GeoTiffData::U16(epsg) = &entry.data {
                        return Ok(Some(epsg.to_string()));
                    }
                }
            }
        }

        if let Some(wkt_bytes) = header.get_wkt_crs_bytes() {
            let wkt = String::from_utf8_lossy(wkt_bytes);
            let marker = "AUTHORITY[\"EPSG\",\"";
            if let Some(start) = wkt.find(marker) {
                let val_start = start + marker.len();
                if let Some(end_offset) = wkt[val_start..].find('"') {
                    let code = &wkt[val_start..val_start + end_offset];
                    if code.chars().all(|c| c.is_ascii_digit()) && !code.is_empty() {
                        return Ok(Some(code.to_string()));
                    }
                }
            }
        }

        Ok(None)
    }
}
