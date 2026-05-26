mod cityjson_reader;
mod matching;
mod metrics;
mod report;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

use crate::metrics::{hausdorff, height, iou, topology, volume};
use crate::report::BuildingMetrics;

#[derive(Parser)]
#[command(name = "eval", about = "LOD2 building reconstruction evaluation")]
struct Cli {
    /// Path to ground truth CityJSONL file (Tier 1 buildings)
    #[arg(long)]
    gt: Option<PathBuf>,

    /// Path to reconstruction result CityJSONL file
    #[arg(long)]
    result: PathBuf,

    /// Optional scalar reference height per building (CSV: building_id,height)
    #[arg(long)]
    permit_heights: Option<PathBuf>,

    /// Output CSV path for per-building metrics
    #[arg(long, default_value = "metrics.csv")]
    output: PathBuf,

    /// Output summary text file
    #[arg(long, default_value = "summary.txt")]
    summary: PathBuf,

    /// Maximum centroid distance for matching (meters)
    #[arg(long, default_value_t = 5.0)]
    max_match_distance: f64,

    /// IoU rasterisation resolution (meters)
    #[arg(long, default_value_t = 0.5)]
    iou_resolution: f64,

    /// Number of surface samples for Hausdorff distance
    #[arg(long, default_value_t = 2000)]
    hausdorff_samples: usize,
}

fn load_permit_heights(path: &PathBuf) -> Result<std::collections::HashMap<String, f64>> {
    let mut map = std::collections::HashMap::new();
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    for line in content.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let Ok(h) = parts[1].trim().parse::<f64>() {
                map.insert(parts[0].trim().to_string(), h);
            }
        }
    }
    Ok(map)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    let result_buildings = cityjson_reader::read_cityjsonl(&cli.result)
        .with_context(|| format!("reading result {}", cli.result.display()))?;
    tracing::info!("Result: {} buildings", result_buildings.len());

    let permit_heights = cli
        .permit_heights
        .as_ref()
        .map(load_permit_heights)
        .transpose()?
        .unwrap_or_default();
    if !permit_heights.is_empty() {
        tracing::info!("Loaded {} permit heights", permit_heights.len());
    }

    let mut all_metrics: Vec<BuildingMetrics> = Vec::new();

    if let Some(gt_path) = &cli.gt {
        let gt_buildings = cityjson_reader::read_cityjsonl(gt_path)
            .with_context(|| format!("reading GT {}", gt_path.display()))?;
        tracing::info!("GT: {} buildings", gt_buildings.len());

        let pairs =
            matching::match_buildings(&gt_buildings, &result_buildings, cli.max_match_distance);
        tracing::info!("Matched {} building pairs", pairs.len());

        for pair in &pairs {
            let gt_bldg = &gt_buildings[pair.gt_idx];
            let res_bldg = &result_buildings[pair.result_idx];

            let gt_mesh = gt_bldg.best_lod();
            let res_mesh = res_bldg.best_lod();

            let match_desc = match &pair.match_type {
                matching::MatchType::ById(id) => format!("id:{id}"),
                matching::MatchType::ByCentroidProximity { distance } => {
                    format!("centroid:{distance:.1}m")
                }
            };

            let fp_iou = gt_mesh
                .zip(res_mesh)
                .and_then(|(g, r)| iou::footprint_iou(g, r, cli.iou_resolution));

            let (ridge_err, eave_err) = gt_mesh
                .zip(res_mesh)
                .and_then(|(g, r)| height::height_error(g, gt_bldg.h_ground, r, res_bldg.h_ground))
                .map(|(r, e)| (Some(r), Some(e)))
                .unwrap_or((None, None));

            let height_vs_permit = permit_heights.get(&gt_bldg.id).and_then(|&ref_h| {
                res_mesh.and_then(|m| height::height_error_vs_scalar(m, res_bldg.h_ground, ref_h))
            });

            let (gt_vol, res_vol, vol_rel) = gt_mesh
                .zip(res_mesh)
                .and_then(|(g, r)| volume::volume_error(g, r))
                .map(|(gv, rv, re)| (Some(gv), Some(rv), Some(re)))
                .unwrap_or((None, None, None));

            let haus = gt_mesh
                .zip(res_mesh)
                .and_then(|(g, r)| hausdorff::hausdorff_distance(g, r, cli.hausdorff_samples));

            let msd = gt_mesh
                .zip(res_mesh)
                .and_then(|(g, r)| hausdorff::mean_surface_distance(g, r, cli.hausdorff_samples));

            let (gt_rp, res_rp, rp_diff) = gt_mesh
                .zip(res_mesh)
                .map(|(g, r)| {
                    let (gc, rc, d) = topology::roof_plane_count_error(g, r);
                    (Some(gc), Some(rc), Some(d))
                })
                .unwrap_or((None, None, None));

            let gt_rt = gt_mesh.map(|m| topology::classify_roof_type(m).to_string());
            let res_rt = res_mesh.map(|m| topology::classify_roof_type(m).to_string());

            all_metrics.push(BuildingMetrics {
                building_id: gt_bldg.id.clone(),
                tier: "tier1".to_string(),
                match_type: match_desc,
                footprint_iou: fp_iou,
                ridge_height_error: ridge_err,
                eave_height_error: eave_err,
                height_vs_permit: height_vs_permit,
                gt_volume: gt_vol,
                result_volume: res_vol,
                volume_relative_error: vol_rel,
                hausdorff_distance: haus,
                mean_surface_distance: msd,
                gt_roof_planes: gt_rp,
                result_roof_planes: res_rp,
                roof_plane_diff: rp_diff,
                gt_roof_type: gt_rt,
                result_roof_type: res_rt,
            });
        }
    }

    for res_bldg in &result_buildings {
        let already = all_metrics.iter().any(|m| m.building_id == res_bldg.id);
        if already {
            continue;
        }

        let res_mesh = res_bldg.best_lod();
        let height_vs_permit = permit_heights.get(&res_bldg.id).and_then(|&ref_h| {
            res_mesh.and_then(|m| height::height_error_vs_scalar(m, res_bldg.h_ground, ref_h))
        });

        all_metrics.push(BuildingMetrics {
            building_id: res_bldg.id.clone(),
            tier: "tier2".to_string(),
            match_type: "none".to_string(),
            footprint_iou: None,
            ridge_height_error: None,
            eave_height_error: None,
            height_vs_permit,
            gt_volume: None,
            result_volume: res_mesh.map(|m| m.compute_volume()),
            volume_relative_error: None,
            hausdorff_distance: None,
            mean_surface_distance: None,
            gt_roof_planes: None,
            result_roof_planes: res_mesh.map(topology::roof_plane_count),
            roof_plane_diff: None,
            gt_roof_type: None,
            result_roof_type: res_mesh.map(|m| topology::classify_roof_type(m).to_string()),
        });
    }

    report::write_csv(&all_metrics, &cli.output)
        .with_context(|| format!("writing {}", cli.output.display()))?;
    tracing::info!("CSV: {}", cli.output.display());

    report::write_summary(&all_metrics, &cli.summary)
        .with_context(|| format!("writing {}", cli.summary.display()))?;
    tracing::info!("Summary: {}", cli.summary.display());

    Ok(())
}
