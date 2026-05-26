use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct BuildingMetrics {
    pub building_id: String,
    pub tier: String,
    pub match_type: String,
    pub footprint_iou: Option<f64>,
    pub ridge_height_error: Option<f64>,
    pub eave_height_error: Option<f64>,
    pub height_vs_permit: Option<f64>,
    pub gt_volume: Option<f64>,
    pub result_volume: Option<f64>,
    pub volume_relative_error: Option<f64>,
    pub hausdorff_distance: Option<f64>,
    pub mean_surface_distance: Option<f64>,
    pub gt_roof_planes: Option<usize>,
    pub result_roof_planes: Option<usize>,
    pub roof_plane_diff: Option<i32>,
    pub gt_roof_type: Option<String>,
    pub result_roof_type: Option<String>,
}

pub fn write_csv(metrics: &[BuildingMetrics], path: &Path) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;

    writeln!(
        f,
        "building_id,tier,match_type,\
         footprint_iou,\
         ridge_height_error,eave_height_error,height_vs_permit,\
         gt_volume,result_volume,volume_relative_error,\
         hausdorff_distance,mean_surface_distance,\
         gt_roof_planes,result_roof_planes,roof_plane_diff,\
         gt_roof_type,result_roof_type"
    )?;

    for m in metrics {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            m.building_id,
            m.tier,
            m.match_type,
            fmt_opt_f64(m.footprint_iou),
            fmt_opt_f64(m.ridge_height_error),
            fmt_opt_f64(m.eave_height_error),
            fmt_opt_f64(m.height_vs_permit),
            fmt_opt_f64(m.gt_volume),
            fmt_opt_f64(m.result_volume),
            fmt_opt_f64(m.volume_relative_error),
            fmt_opt_f64(m.hausdorff_distance),
            fmt_opt_f64(m.mean_surface_distance),
            fmt_opt_usize(m.gt_roof_planes),
            fmt_opt_usize(m.result_roof_planes),
            fmt_opt_i32(m.roof_plane_diff),
            m.gt_roof_type.as_deref().unwrap_or(""),
            m.result_roof_type.as_deref().unwrap_or(""),
        )?;
    }
    Ok(())
}

pub fn write_summary(metrics: &[BuildingMetrics], path: &Path) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;

    let tier1: Vec<_> = metrics.iter().filter(|m| m.tier == "tier1").collect();
    let tier2: Vec<_> = metrics.iter().filter(|m| m.tier == "tier2").collect();

    writeln!(f, "=== LOD2 Ground Truth Evaluation Summary ===")?;
    writeln!(f)?;
    writeln!(f, "Total buildings evaluated: {}", metrics.len())?;
    writeln!(f, "  Tier 1 (full 3D): {}", tier1.len())?;
    writeln!(f, "  Tier 2 (lighter):  {}", tier2.len())?;
    writeln!(f)?;

    writeln!(f, "--- Footprint IoU ---")?;
    write_stat_block(&mut f, &collect_f64(metrics, |m| m.footprint_iou))?;

    writeln!(f, "--- Ridge Height Error (m) ---")?;
    write_stat_block(&mut f, &collect_f64(metrics, |m| m.ridge_height_error))?;

    if !tier1.is_empty() {
        writeln!(f, "--- Volume Relative Error (Tier 1 only) ---")?;
        write_stat_block(&mut f, &collect_f64(&tier1, |m| m.volume_relative_error))?;

        writeln!(f, "--- Hausdorff Distance (Tier 1 only, m) ---")?;
        write_stat_block(&mut f, &collect_f64(&tier1, |m| m.hausdorff_distance))?;

        writeln!(f, "--- Mean Surface Distance (Tier 1 only, m) ---")?;
        write_stat_block(&mut f, &collect_f64(&tier1, |m| m.mean_surface_distance))?;

        writeln!(f, "--- Roof Plane Count Diff (Tier 1 only) ---")?;
        let diffs: Vec<f64> = tier1
            .iter()
            .filter_map(|m| m.roof_plane_diff.map(|d| d as f64))
            .collect();
        write_stat_block(&mut f, &diffs)?;
    }

    Ok(())
}

fn collect_f64<T>(items: &[T], extractor: impl Fn(&T) -> Option<f64>) -> Vec<f64> {
    items.iter().filter_map(|m| extractor(m)).collect()
}

fn write_stat_block(f: &mut impl Write, values: &[f64]) -> std::io::Result<()> {
    if values.is_empty() {
        writeln!(f, "  (no data)")?;
        writeln!(f)?;
        return Ok(());
    }
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[n / 2];
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();
    let p10 = sorted[(n as f64 * 0.1) as usize];
    let p90 = sorted[((n as f64 * 0.9) as usize).min(n - 1)];

    writeln!(f, "  n:      {}", n)?;
    writeln!(f, "  mean:   {:.4}", mean)?;
    writeln!(f, "  median: {:.4}", median)?;
    writeln!(f, "  std:    {:.4}", std_dev)?;
    writeln!(f, "  p10:    {:.4}", p10)?;
    writeln!(f, "  p90:    {:.4}", p90)?;
    writeln!(f, "  min:    {:.4}", sorted[0])?;
    writeln!(f, "  max:    {:.4}", sorted[n - 1])?;
    writeln!(f)?;
    Ok(())
}

fn fmt_opt_f64(v: Option<f64>) -> String {
    v.map(|x| format!("{:.6}", x)).unwrap_or_default()
}

fn fmt_opt_usize(v: Option<usize>) -> String {
    v.map(|x| x.to_string()).unwrap_or_default()
}

fn fmt_opt_i32(v: Option<i32>) -> String {
    v.map(|x| x.to_string()).unwrap_or_default()
}
