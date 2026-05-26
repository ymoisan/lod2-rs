use lod2_common::mesh::BuildingGeometry;
use std::collections::HashMap;

#[derive(Debug)]
pub struct MatchedPair {
    pub gt_idx: usize,
    pub result_idx: usize,
    pub match_type: MatchType,
}

#[derive(Debug)]
pub enum MatchType {
    ById(String),
    ByCentroidProximity { distance: f64 },
}

pub fn match_buildings(
    gt: &[BuildingGeometry],
    result: &[BuildingGeometry],
    max_centroid_distance: f64,
) -> Vec<MatchedPair> {
    let mut pairs = Vec::new();
    let mut used_result: Vec<bool> = vec![false; result.len()];

    let gt_ids: HashMap<&str, usize> = gt
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id.as_str(), i))
        .collect();

    let result_ids: HashMap<&str, usize> = result
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id.as_str(), i))
        .collect();

    for (gt_id, &gt_idx) in &gt_ids {
        if let Some(&res_idx) = result_ids.get(gt_id) {
            pairs.push(MatchedPair {
                gt_idx,
                result_idx: res_idx,
                match_type: MatchType::ById(gt_id.to_string()),
            });
            used_result[res_idx] = true;
        }
    }

    let matched_gt: std::collections::HashSet<usize> =
        pairs.iter().map(|p| p.gt_idx).collect();

    for (gt_idx, gt_bldg) in gt.iter().enumerate() {
        if matched_gt.contains(&gt_idx) {
            continue;
        }
        let gt_centroid = match building_centroid_2d(gt_bldg) {
            Some(c) => c,
            None => continue,
        };

        let mut best_dist = max_centroid_distance;
        let mut best_idx = None;

        for (res_idx, res_bldg) in result.iter().enumerate() {
            if used_result[res_idx] {
                continue;
            }
            let res_centroid = match building_centroid_2d(res_bldg) {
                Some(c) => c,
                None => continue,
            };
            let dist = ((gt_centroid.0 - res_centroid.0).powi(2)
                + (gt_centroid.1 - res_centroid.1).powi(2))
            .sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(res_idx);
            }
        }

        if let Some(res_idx) = best_idx {
            pairs.push(MatchedPair {
                gt_idx,
                result_idx: res_idx,
                match_type: MatchType::ByCentroidProximity {
                    distance: best_dist,
                },
            });
            used_result[res_idx] = true;
        }
    }

    pairs
}

fn building_centroid_2d(bldg: &BuildingGeometry) -> Option<(f64, f64)> {
    let mesh = bldg.best_lod()?;
    if mesh.vertices.is_empty() {
        return None;
    }
    let (sx, sy, n) = mesh.vertices.iter().fold((0.0, 0.0, 0usize), |(sx, sy, n), v| {
        (sx + v.x, sy + v.y, n + 1)
    });
    Some((sx / n as f64, sy / n as f64))
}
