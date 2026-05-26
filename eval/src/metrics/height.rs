use lod2_common::mesh::{Mesh, SurfaceType};

#[derive(Debug, Clone)]
pub struct HeightStats {
    pub z_min: f64,
    pub z_max: f64,
    pub z_ground: f64,
    pub z_ridge: f64,
    pub z_eave: f64,
    pub height_above_ground: f64,
}

/// Extract height statistics from a mesh.
pub fn mesh_height_stats(mesh: &Mesh, h_ground: f64) -> Option<HeightStats> {
    if mesh.vertices.is_empty() {
        return None;
    }

    let z_min = mesh.vertices.iter().map(|v| v.z).fold(f64::MAX, f64::min);
    let z_max = mesh.vertices.iter().map(|v| v.z).fold(f64::MIN, f64::max);

    let mut roof_zs = Vec::new();
    for face in &mesh.faces {
        let is_roof = face
            .semantic_index
            .and_then(|idx| mesh.semantics.get(idx))
            .map(|s| s.surface_type == SurfaceType::RoofSurface)
            .unwrap_or(false);
        if is_roof {
            for &idx in &face.indices {
                roof_zs.push(mesh.vertices[idx as usize].z);
            }
        }
    }

    let z_ridge = if roof_zs.is_empty() {
        z_max
    } else {
        roof_zs.iter().cloned().fold(f64::MIN, f64::max)
    };

    let z_eave = if roof_zs.is_empty() {
        z_max
    } else {
        roof_zs.iter().cloned().fold(f64::MAX, f64::min)
    };

    Some(HeightStats {
        z_min,
        z_max,
        z_ground: h_ground,
        z_ridge,
        z_eave,
        height_above_ground: z_ridge - h_ground,
    })
}

/// Height error between two meshes: difference in ridge height and eave height.
pub fn height_error(gt_mesh: &Mesh, gt_h_ground: f64, res_mesh: &Mesh, res_h_ground: f64) -> Option<(f64, f64)> {
    let gt = mesh_height_stats(gt_mesh, gt_h_ground)?;
    let res = mesh_height_stats(res_mesh, res_h_ground)?;
    let ridge_err = res.z_ridge - gt.z_ridge;
    let eave_err = res.z_eave - gt.z_eave;
    Some((ridge_err, eave_err))
}

/// Height error against a scalar reference height (e.g., from a building permit).
pub fn height_error_vs_scalar(
    mesh: &Mesh,
    h_ground: f64,
    reference_height: f64,
) -> Option<f64> {
    let stats = mesh_height_stats(mesh, h_ground)?;
    Some(stats.height_above_ground - reference_height)
}
