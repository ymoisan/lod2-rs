use lod2_common::mesh::{Mesh, SurfaceType};

/// Count the number of distinct roof planes in a mesh.
pub fn roof_plane_count(mesh: &Mesh) -> usize {
    let mut roof_sem_indices = std::collections::HashSet::new();
    for face in &mesh.faces {
        if let Some(idx) = face.semantic_index {
            if let Some(sem) = mesh.semantics.get(idx) {
                if sem.surface_type == SurfaceType::RoofSurface {
                    roof_sem_indices.insert(idx);
                }
            }
        }
    }
    roof_sem_indices.len()
}

/// Compare roof plane counts between GT and result.
/// Returns (gt_count, result_count, difference).
pub fn roof_plane_count_error(gt_mesh: &Mesh, result_mesh: &Mesh) -> (usize, usize, i32) {
    let gt_count = roof_plane_count(gt_mesh);
    let res_count = roof_plane_count(result_mesh);
    (gt_count, res_count, res_count as i32 - gt_count as i32)
}

/// Classify roof type from mesh semantics.
///
/// Simple heuristic based on roof plane count and slope distribution:
/// - 0 or 1 plane with slope < 10: flat
/// - 2 planes: gable
/// - 4 planes: hip
/// - otherwise: complex
pub fn classify_roof_type(mesh: &Mesh) -> &'static str {
    let mut slopes = Vec::new();
    let mut roof_sem_indices = std::collections::HashSet::new();

    for face in &mesh.faces {
        if let Some(idx) = face.semantic_index {
            if let Some(sem) = mesh.semantics.get(idx) {
                if sem.surface_type == SurfaceType::RoofSurface && roof_sem_indices.insert(idx) {
                    if let Some(slope) = sem.slope {
                        slopes.push(slope);
                    }
                }
            }
        }
    }

    let n = roof_sem_indices.len();
    let max_slope = slopes.iter().cloned().fold(0.0f64, f64::max);

    match n {
        0 => "flat",
        1 => {
            if max_slope < 10.0 {
                "flat"
            } else {
                "shed"
            }
        }
        2 => "gable",
        3 => "complex",
        4 => "hip",
        _ => "complex",
    }
}
