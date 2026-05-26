use lod2_common::mesh::Mesh;

/// Volume error between GT and result meshes.
///
/// Uses the divergence theorem (already implemented in `Mesh::compute_volume`).
/// Returns (gt_volume, result_volume, relative_error) where
/// relative_error = (result - gt) / gt.
pub fn volume_error(gt_mesh: &Mesh, result_mesh: &Mesh) -> Option<(f64, f64, f64)> {
    let gt_vol = gt_mesh.compute_volume();
    let res_vol = result_mesh.compute_volume();

    if gt_vol < 1e-6 {
        return None;
    }

    let relative = (res_vol - gt_vol) / gt_vol;
    Some((gt_vol, res_vol, relative))
}
