use crate::cityjson::{CityJsonTransform, CityJsonWriter};
use crate::las_reader::LasReader;
use crate::mesh::BuildingGeometry;
use crate::point_cloud::PointCloud;
use crate::polygon::Footprint;
use crate::vector_reader::VectorReader;
use clap::Parser;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Mutex;

#[derive(Parser, Debug, Clone)]
pub struct PipelineArgs {
    #[arg(long)]
    pub footprints: PathBuf,

    #[arg(long)]
    pub pointcloud: PathBuf,

    #[arg(long)]
    pub output: PathBuf,
}

pub trait Reconstructor: Send + Sync {
    fn name(&self) -> &str;
    fn reconstruct(
        &self,
        footprint: &Footprint,
        points: &PointCloud,
        h_ground: f64,
    ) -> BuildingGeometry;
}

/// Crop points within a footprint's bounding box + polygon test.
pub fn crop_points(pc: &PointCloud, footprint: &Footprint) -> PointCloud {
    let bbox = footprint.polygon.bbox_2d();
    let margin = 2.0;
    let mut cropped = PointCloud::new();
    for p in &pc.positions {
        if p.x >= bbox[0] - margin
            && p.x <= bbox[2] + margin
            && p.y >= bbox[1] - margin
            && p.y <= bbox[3] + margin
            && footprint.contains_2d(p.x, p.y)
        {
            cropped.push(*p);
        }
    }
    cropped
}

/// Run the full pipeline.
pub fn run_pipeline(args: &PipelineArgs, reconstructor: &dyn Reconstructor) -> anyhow::Result<()> {
    tracing::info!("Reading footprints from {}", args.footprints.display());
    let footprints = VectorReader::read_footprints(&args.footprints)?;
    let crs = VectorReader::read_crs(&args.footprints)?;
    tracing::info!("Found CRS: {:?}", crs);
    tracing::info!("Read {} footprints", footprints.len());

    tracing::info!("Reading point cloud from {}", args.pointcloud.display());
    let pc = LasReader::read_file(&args.pointcloud)?;

    let (cx, cy) = compute_centroid(&footprints);
    let transform = CityJsonTransform {
        scale: [0.001, 0.001, 0.001],
        translate: [cx, cy, 0.0],
    };

    let output_path = args.output.join("output.city.jsonl");
    let writer = Mutex::new(
        CityJsonWriter::new(&output_path, transform)?
            .with_reference_system(crs.as_deref().unwrap_or("2154")),
    );

    {
        let mut w = writer.lock().unwrap();
        w.write_header()?;
    }

    tracing::info!("Reconstructing buildings...");
    let results: Vec<BuildingGeometry> = footprints
        .par_iter()
        .map(|fp| {
            let cropped = crop_points(&pc, fp);
            let stats = cropped.compute_statistics();
            let h_ground = stats.z_min;

            // Catch panics from CDT or other geometry issues
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                reconstructor.reconstruct(fp, &cropped, h_ground)
            }));

            match result {
                Ok(geom) => geom,
                Err(_) => {
                    tracing::warn!("Reconstruction panicked for {}, using flat roof fallback", fp.id);
                    let mut geom = BuildingGeometry::new(&fp.id);
                    geom.attributes = fp.attributes.clone();
                    geom.h_ground = h_ground;
                    let h_roof = if stats.count > 0 { stats.z_70p } else { h_ground + 5.0 };
                    geom.lod22 = build_flat_roof(fp, h_ground, h_roof);
                    geom
                }
            }
        })
        .collect();

    let mut succeeded = 0usize;
    let mut failed = 0usize;
    {
        let mut w = writer.lock().unwrap();
        for geom in &results {
            if geom.best_lod().is_some() {
                succeeded += 1;
            } else {
                failed += 1;
            }
            w.write_feature(geom)?;
        }
    }

    writer.into_inner().unwrap().finish()?;
    tracing::info!(
        "Reconstruction complete: {} succeeded, {} failed",
        succeeded,
        failed
    );

    Ok(())
}

fn compute_centroid(footprints: &[Footprint]) -> (f64, f64) {
    if footprints.is_empty() {
        return (0.0, 0.0);
    }
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut n = 0.0;
    for fp in footprints {
        let (x, y) = fp.polygon.centroid_2d();
        cx += x;
        cy += y;
        n += 1.0;
    }
    (cx / n, cy / n)
}

/// Helper to build a flat-roof mesh (fallback for all methods).
pub fn build_flat_roof(
    footprint: &Footprint,
    h_ground: f64,
    h_roof: f64,
) -> Option<crate::mesh::Mesh> {
    use crate::mesh::{Face, Mesh, SemanticSurface};

    let exterior = &footprint.polygon.exterior;
    let n = exterior.len().saturating_sub(1);
    if n < 3 {
        return None;
    }

    let mut mesh = Mesh::new();
    let ground_idx = mesh.add_semantic(SemanticSurface::ground());
    let wall_idx = mesh.add_semantic(SemanticSurface::wall(true));
    let roof_idx = mesh.add_semantic(SemanticSurface::roof());

    let mut bottom = Vec::with_capacity(n);
    let mut top = Vec::with_capacity(n);
    for i in 0..n {
        let v = &exterior.vertices[i];
        bottom.push(mesh.add_vertex(nalgebra::Point3::new(v.x, v.y, h_ground)));
        top.push(mesh.add_vertex(nalgebra::Point3::new(v.x, v.y, h_roof)));
    }

    mesh.add_face(Face::new(bottom.iter().rev().copied().collect()).with_semantic(ground_idx));
    mesh.add_face(Face::new(top.clone()).with_semantic(roof_idx));

    for i in 0..n {
        let j = (i + 1) % n;
        mesh.add_face(
            Face::new(vec![bottom[i], bottom[j], top[j], top[i]]).with_semantic(wall_idx),
        );
    }

    Some(mesh)
}
