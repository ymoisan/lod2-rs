//! Line regulariser — clusters near-parallel near-coincident segments and
//! snaps them onto a shared representative line.
//!
//! Counterpart of roofer's `LineRegulariser` (Verdie & Lafarge 2014 style).
//! Input: arbitrary 2D segments (footprint edges + plane–plane ridges).
//! Output: same number of segments, each with endpoints projected onto its
//! cluster's representative line.  Segments that previously differed by a
//! sub-tolerance angle or offset now coincide exactly, eliminating the
//! sub-cell slivers and degenerate intersections that drive the arrangement
//! complexity explosion on fragmented buildings.
//!
//! Clustering keys:
//!   * orientation bin: `θ` mapped to `[0, π)` (lines are undirected) and
//!     quantised by `angle_tol`.
//!   * perpendicular-offset bin: signed distance from origin to the line,
//!     measured in the cluster's representative normal direction, quantised
//!     by `dist_tol`.
//!
//! The representative line per cluster is the length-weighted mean
//! orientation and length-weighted mean offset of its members.  Short input
//! segments contribute proportionally less, so the cluster line stays close
//! to the long dominant edges (footprints) and ignores short noisy ridges.

use std::collections::HashMap;

/// Snap near-parallel, near-coincident segments onto shared lines.
///
/// - `angle_tol` in radians (e.g. 5° = 0.087)
/// - `dist_tol` in metres (e.g. 0.10)
///
/// Segments shorter than `min_len` are left untouched (they are noisy and
/// would skew the representative line).
pub fn regularise_segments(
    segments: &mut [([f64; 2], [f64; 2])],
    angle_tol: f64,
    dist_tol: f64,
    min_len: f64,
) {
    if segments.is_empty() {
        return;
    }

    // Per-segment: (theta in [0,π), perpendicular offset, length).
    let mut params: Vec<(f64, f64, f64)> = Vec::with_capacity(segments.len());
    for &(a, b) in segments.iter() {
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-12 {
            params.push((0.0, 0.0, 0.0));
            continue;
        }
        // Undirected orientation in [0, π).
        let mut theta = dy.atan2(dx);
        if theta < 0.0 {
            theta += std::f64::consts::PI;
        }
        if theta >= std::f64::consts::PI {
            theta -= std::f64::consts::PI;
        }
        // Perpendicular offset = a · n, n = (-sin θ, cos θ).
        let nx = -theta.sin();
        let ny = theta.cos();
        let offset = a[0] * nx + a[1] * ny;
        params.push((theta, offset, len));
    }

    let inv_angle = 1.0 / angle_tol;
    let inv_dist = 1.0 / dist_tol;

    // First pass: assign each segment to a (θ_bin, off_bin) bucket and
    // accumulate length-weighted sums for the representative line.
    // Bucket key wraps θ at π for parallel-but-opposite-sign offsets.
    let mut sums: HashMap<(i32, i32), (f64, f64, f64)> = HashMap::new();
    let mut bucket_of: Vec<Option<(i32, i32)>> = Vec::with_capacity(segments.len());

    for &(theta, offset, len) in &params {
        if len < min_len {
            bucket_of.push(None);
            continue;
        }
        let tb = (theta * inv_angle).round() as i32;
        let ob = (offset * inv_dist).round() as i32;
        let key = (tb, ob);
        let e = sums.entry(key).or_insert((0.0, 0.0, 0.0));
        e.0 += theta * len; // weighted θ
        e.1 += offset * len; // weighted offset
        e.2 += len; // total weight
        bucket_of.push(Some(key));
    }

    // Resolve each bucket to a representative (θ, offset).
    let mut rep: HashMap<(i32, i32), (f64, f64)> = HashMap::with_capacity(sums.len());
    for (k, &(wt_theta, wt_off, w)) in sums.iter() {
        if w <= 0.0 {
            continue;
        }
        rep.insert(*k, (wt_theta / w, wt_off / w));
    }

    // Snap each long-enough segment onto its representative line by projecting
    // both endpoints orthogonally.
    for (i, seg) in segments.iter_mut().enumerate() {
        let key = match bucket_of[i] {
            Some(k) => k,
            None => continue,
        };
        let &(theta_r, off_r) = match rep.get(&key) {
            Some(v) => v,
            None => continue,
        };
        let dx = theta_r.cos();
        let dy = theta_r.sin();
        let nx = -dy;
        let ny = dx;
        // A point on the representative line: off_r * n.
        let p0x = off_r * nx;
        let p0y = off_r * ny;
        // Project endpoint q onto the line: q' = p0 + ((q - p0) · t) * t.
        let project = |q: [f64; 2]| -> [f64; 2] {
            let vx = q[0] - p0x;
            let vy = q[1] - p0y;
            let s = vx * dx + vy * dy;
            [p0x + s * dx, p0y + s * dy]
        };
        seg.0 = project(seg.0);
        seg.1 = project(seg.1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_segments_collapse_to_same_line() {
        // Two horizontal segments 1 mm apart should snap to the same y.
        let mut segs = vec![
            ([0.0_f64, 0.0_f64], [10.0_f64, 0.0_f64]),
            ([0.0_f64, 0.001_f64], [10.0_f64, 0.001_f64]),
        ];
        regularise_segments(&mut segs, 0.087, 0.10, 0.5);
        let y0 = (segs[0].0[1] + segs[0].1[1]) * 0.5;
        let y1 = (segs[1].0[1] + segs[1].1[1]) * 0.5;
        assert!((y0 - y1).abs() < 1e-9, "y0={} y1={}", y0, y1);
    }

    #[test]
    fn perpendicular_segments_stay_distinct() {
        let mut segs = vec![
            ([0.0_f64, 0.0_f64], [10.0_f64, 0.0_f64]),
            ([0.0_f64, 0.0_f64], [0.0_f64, 10.0_f64]),
        ];
        let before = segs.clone();
        regularise_segments(&mut segs, 0.087, 0.10, 0.5);
        // Should be (essentially) unchanged.
        for (a, b) in segs.iter().zip(before.iter()) {
            assert!((a.0[0] - b.0[0]).abs() < 1e-9);
            assert!((a.0[1] - b.0[1]).abs() < 1e-9);
            assert!((a.1[0] - b.1[0]).abs() < 1e-9);
            assert!((a.1[1] - b.1[1]).abs() < 1e-9);
        }
    }

    #[test]
    fn short_segments_are_ignored() {
        // A 1 cm segment near a 10 m one — short one stays put.
        let mut segs = vec![
            ([0.0_f64, 0.0_f64], [10.0_f64, 0.0_f64]),
            ([5.0_f64, 0.05_f64], [5.01_f64, 0.05_f64]),
        ];
        regularise_segments(&mut segs, 0.087, 0.10, 0.5);
        // The short segment's y should still be ~0.05, not snapped to 0.
        assert!((segs[1].0[1] - 0.05).abs() < 1e-9);
    }
}
