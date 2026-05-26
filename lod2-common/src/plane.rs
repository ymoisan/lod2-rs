use nalgebra::{Matrix3, Point3, Vector3};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "pipeline")]
use kiddo::{KdTree, SquaredEuclidean};

/// Global counters for plane detection method usage (reset per run via `reset_detection_stats`).
static REGION_GROWING_COUNT: AtomicUsize = AtomicUsize::new(0);
static RANSAC_FALLBACK_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Reset detection statistics counters.
pub fn reset_detection_stats() {
    REGION_GROWING_COUNT.store(0, Ordering::Relaxed);
    RANSAC_FALLBACK_COUNT.store(0, Ordering::Relaxed);
}

/// Get detection statistics: (region_growing_count, ransac_fallback_count).
pub fn detection_stats() -> (usize, usize) {
    (
        REGION_GROWING_COUNT.load(Ordering::Relaxed),
        RANSAC_FALLBACK_COUNT.load(Ordering::Relaxed),
    )
}

#[derive(Debug, Clone)]
pub struct Plane {
    pub normal: Vector3<f64>,
    pub d: f64,
    pub inliers: Vec<usize>,
    pub rmse: f64,
}

impl Plane {
    pub fn from_normal_and_point(normal: Vector3<f64>, point: &Point3<f64>) -> Self {
        let d = -normal.dot(&point.coords);
        Self { normal, d, inliers: Vec::new(), rmse: 0.0 }
    }

    pub fn distance_to(&self, p: &Point3<f64>) -> f64 {
        (self.normal.dot(&p.coords) + self.d).abs()
    }

    pub fn signed_distance_to(&self, p: &Point3<f64>) -> f64 {
        self.normal.dot(&p.coords) + self.d
    }

    pub fn eval_z(&self, x: f64, y: f64) -> Option<f64> {
        if self.normal.z.abs() < 1e-10 {
            return None;
        }
        Some(-(self.normal.x * x + self.normal.y * y + self.d) / self.normal.z)
    }

    pub fn slope_degrees(&self) -> f64 {
        let up = Vector3::new(0.0, 0.0, 1.0);
        let cos_angle = self.normal.dot(&up).abs();
        cos_angle.acos().to_degrees()
    }

    pub fn azimuth_degrees(&self) -> f64 {
        let mut az = self.normal.y.atan2(self.normal.x).to_degrees();
        if az < 0.0 {
            az += 360.0;
        }
        az
    }

    pub fn is_near_horizontal(&self, threshold_deg: f64) -> bool {
        self.slope_degrees() < threshold_deg
    }

    pub fn is_near_vertical(&self, threshold_deg: f64) -> bool {
        self.slope_degrees() > (90.0 - threshold_deg)
    }
}

/// 2D spatial extent of a plane's inlier points.
/// Used to penalize plane assignment when CDT faces are far from the plane's support region.
#[derive(Debug, Clone)]
pub struct PlaneLocality {
    pub cx: f64,
    pub cy: f64,
    pub radius: f64,
}

/// Compute PlaneLocality for each plane from its inlier indices into the point cloud.
pub fn compute_plane_localities(planes: &[Plane], points: &[Point3<f64>]) -> Vec<PlaneLocality> {
    planes
        .iter()
        .map(|plane| {
            if plane.inliers.is_empty() {
                return PlaneLocality {
                    cx: 0.0,
                    cy: 0.0,
                    radius: 0.0,
                };
            }
            let n = plane.inliers.len() as f64;
            let mut sx = 0.0;
            let mut sy = 0.0;
            for &i in &plane.inliers {
                sx += points[i].x;
                sy += points[i].y;
            }
            let cx = sx / n;
            let cy = sy / n;
            let radius = plane
                .inliers
                .iter()
                .map(|&i| {
                    let dx = points[i].x - cx;
                    let dy = points[i].y - cy;
                    (dx * dx + dy * dy).sqrt()
                })
                .fold(0.0_f64, f64::max);
            PlaneLocality { cx, cy, radius }
        })
        .collect()
}

/// Filter out "rooftop attachments": planes whose 2D inlier footprint is
/// substantially contained within a larger plane's 2D footprint AND whose
/// mean z is higher than that larger plane.  This is the geometric
/// signature of HVAC units, chimneys, parapets, and other rooftop bumps
/// that survive region growing (they form coplanar clusters of ≥ min_pts
/// points) but produce visible "sprouts" in the reconstructed mesh
/// because they win a small handful of arrangement faces in graph-cut.
///
/// A small section that is genuinely part of the building (dormer slope,
/// attached wing) sits *beside* the main roof in (x,y) — not *above* it —
/// and is preserved.
///
/// Criteria (per pair P,Q where |Q.inliers| > |P.inliers|):
///   - |P.inliers| ≤ 0.20 · |Q.inliers|   (P is much smaller; rules out
///     the two slopes of a gable, where one slope can have its centroid
///     inside the other's bbox)
///   - P's centroid (x,y) lies inside Q's inlier AABB
///   - ≥ 50% of P's inliers (x,y) lie inside Q's inlier AABB
///   - mean_z(P) > mean_z(Q) + 0.15 m
/// If any Q satisfies all four, P is dropped.
pub fn filter_rooftop_attachments(
    planes: Vec<Plane>,
    points: &[Point3<f64>],
) -> Vec<Plane> {
    if planes.len() < 2 {
        return planes;
    }
    // Pre-compute per-plane: AABB, mean_z, centroid (x,y).
    struct PlaneAabb {
        xmin: f64, xmax: f64, ymin: f64, ymax: f64,
        cx: f64, cy: f64, mean_z: f64,
        n: usize,
    }
    let stats: Vec<PlaneAabb> = planes
        .iter()
        .map(|p| {
            if p.inliers.is_empty() {
                return PlaneAabb {
                    xmin: 0.0, xmax: 0.0, ymin: 0.0, ymax: 0.0,
                    cx: 0.0, cy: 0.0, mean_z: 0.0, n: 0,
                };
            }
            let mut xmin = f64::INFINITY;
            let mut xmax = f64::NEG_INFINITY;
            let mut ymin = f64::INFINITY;
            let mut ymax = f64::NEG_INFINITY;
            let mut sx = 0.0;
            let mut sy = 0.0;
            let mut sz = 0.0;
            for &i in &p.inliers {
                let pt = &points[i];
                if pt.x < xmin { xmin = pt.x; }
                if pt.x > xmax { xmax = pt.x; }
                if pt.y < ymin { ymin = pt.y; }
                if pt.y > ymax { ymax = pt.y; }
                sx += pt.x; sy += pt.y; sz += pt.z;
            }
            let n = p.inliers.len() as f64;
            PlaneAabb {
                xmin, xmax, ymin, ymax,
                cx: sx / n, cy: sy / n, mean_z: sz / n,
                n: p.inliers.len(),
            }
        })
        .collect();

    let mut keep = vec![true; planes.len()];
    for i in 0..planes.len() {
        if stats[i].n == 0 { continue; }
        for j in 0..planes.len() {
            if i == j || stats[j].n <= stats[i].n { continue; }
            let q = &stats[j];
            // Size ratio: an attachment must be much smaller than its
            // parent.  Two slopes of a gable can both have centroids
            // inside each other's bbox; the size ratio is what tells
            // them apart from a HVAC bump.
            if (stats[i].n as f64) > 0.20 * (q.n as f64) { continue; }
            // Centroid of P in Q's bbox?
            if stats[i].cx < q.xmin || stats[i].cx > q.xmax
                || stats[i].cy < q.ymin || stats[i].cy > q.ymax {
                continue;
            }
            // mean_z separation: P must sit above Q.
            if stats[i].mean_z <= q.mean_z + 0.15 { continue; }
            // Fraction of P's inliers within Q's bbox.
            let mut inside = 0usize;
            for &pi in &planes[i].inliers {
                let pt = &points[pi];
                if pt.x >= q.xmin && pt.x <= q.xmax
                    && pt.y >= q.ymin && pt.y <= q.ymax {
                    inside += 1;
                }
            }
            if (inside as f64) >= 0.5 * (stats[i].n as f64) {
                keep[i] = false;
                break;
            }
        }
    }

    planes
        .into_iter()
        .enumerate()
        .filter_map(|(i, p)| if keep[i] { Some(p) } else { None })
        .collect()
}

// ─── Detection method selection ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneDetectionMethod {
    RegionGrowing,
    Ransac,
}

impl Default for PlaneDetectionMethod {
    fn default() -> Self {
        Self::RegionGrowing
    }
}

// ─── Configuration ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PlaneDetectorConfig {
    pub method: PlaneDetectionMethod,
    // Region growing parameters
    pub normal_k: usize,
    pub plane_k: usize,
    pub epsilon: f64,
    pub normal_threshold: f64,
    pub n_refit: usize,
    // Shared parameters
    pub min_points: usize,
    pub max_planes: usize,
    pub wall_angle_threshold: f64,
    pub merge_angle_degrees: f64,
    pub merge_distance: f64,
    pub merge_centroid_2d_distance: f64,
    // RANSAC-only parameters
    pub max_iterations: usize,
}

impl Default for PlaneDetectorConfig {
    fn default() -> Self {
        Self {
            method: PlaneDetectionMethod::RegionGrowing,
            normal_k: 5,
            plane_k: 15,
            epsilon: 0.2,
            normal_threshold: 0.75,
            n_refit: 5,
            min_points: 15,
            max_planes: usize::MAX,
            wall_angle_threshold: 15.0,
            merge_angle_degrees: 7.5,
            merge_distance: 0.5,
            merge_centroid_2d_distance: 5.0,
            max_iterations: 2000,
        }
    }
}

/// Backwards-compatible alias.
pub type RansacConfig = PlaneDetectorConfig;

pub struct PlaneDetector {
    config: PlaneDetectorConfig,
}

impl PlaneDetector {
    pub fn new(config: PlaneDetectorConfig) -> Self {
        Self { config }
    }

    /// Detect multiple planes from a point cloud.
    pub fn detect_multiple(&self, points: &[Point3<f64>], max_planes: usize) -> Vec<Plane> {
        match self.config.method {
            #[cfg(feature = "pipeline")]
            PlaneDetectionMethod::RegionGrowing => {
                // Try region growing; fall back to RANSAC if KD-tree construction panics
                // (e.g., too many coincident points on one axis).
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.detect_region_growing(points, max_planes)
                }));
                match result {
                    Ok(planes) => {
                        REGION_GROWING_COUNT.fetch_add(1, Ordering::Relaxed);
                        planes
                    }
                    Err(_) => {
                        RANSAC_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
                        self.detect_ransac(points, max_planes)
                    }
                }
            }
            #[cfg(not(feature = "pipeline"))]
            PlaneDetectionMethod::RegionGrowing => {
                RANSAC_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
                self.detect_ransac(points, max_planes)
            }
            PlaneDetectionMethod::Ransac => {
                RANSAC_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
                self.detect_ransac(points, max_planes)
            }
        }
    }

    // ─── Region Growing ─────────────────────────────────────────────────────

    #[cfg(feature = "pipeline")]
    fn detect_region_growing(&self, points: &[Point3<f64>], max_planes: usize) -> Vec<Plane> {
        let n = points.len();
        if n < self.config.min_points {
            return Vec::new();
        }

        // Build KD-tree
        let tree: KdTree<f64, 3> = points
            .iter()
            .enumerate()
            .map(|(i, p)| ([p.x, p.y, p.z], i as u64))
            .collect();

        // Step 1: Estimate per-point normals via PCA on normal_k neighbors
        let normals = self.estimate_normals(points, &tree);

        // Step 2: Generate seeds sorted by planarity quality (best first)
        let seeds = self.generate_seeds(points, &tree);

        // Step 3: Region growing
        let mut assigned = vec![false; n];
        let mut planes = Vec::new();

        for seed_idx in seeds {
            if assigned[seed_idx] {
                continue;
            }
            if planes.len() >= max_planes.min(self.config.max_planes) {
                break;
            }

            if let Some(plane) = self.grow_region(
                points, &normals, &tree, &mut assigned, seed_idx,
            ) {
                if !plane.is_near_vertical(self.config.wall_angle_threshold) {
                    planes.push(plane);
                }
            }
        }

        self.merge_similar_planes(points, planes)
    }

    #[cfg(feature = "pipeline")]
    fn estimate_normals(&self, points: &[Point3<f64>], tree: &KdTree<f64, 3>) -> Vec<Vector3<f64>> {
        let k = self.config.normal_k + 1; // +1 because query includes the point itself
        points
            .iter()
            .map(|p| {
                let neighbours = tree.nearest_n::<SquaredEuclidean>(&[p.x, p.y, p.z], k);
                let indices: Vec<usize> = neighbours.iter().map(|nb| nb.item as usize).collect();
                let normal = Self::pca_normal(points, &indices);
                // Orient upward
                if normal.z < 0.0 { -normal } else { normal }
            })
            .collect()
    }

    #[cfg(feature = "pipeline")]
    fn generate_seeds(&self, points: &[Point3<f64>], tree: &KdTree<f64, 3>) -> Vec<usize> {
        let k = self.config.plane_k + 1;
        let mut seed_qualities: Vec<(usize, f64)> = points
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                let neighbours = tree.nearest_n::<SquaredEuclidean>(&[p.x, p.y, p.z], k);
                let indices: Vec<usize> = neighbours.iter().map(|nb| nb.item as usize).collect();
                let quality = Self::planarity(points, &indices);
                (idx, quality)
            })
            .collect();

        // Sort by quality descending (higher planarity = better seed)
        seed_qualities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        seed_qualities.into_iter().map(|(idx, _)| idx).collect()
    }

    /// Grow a region from a seed point using BFS.
    #[cfg(feature = "pipeline")]
    fn grow_region(
        &self,
        points: &[Point3<f64>],
        normals: &[Vector3<f64>],
        tree: &KdTree<f64, 3>,
        assigned: &mut [bool],
        seed: usize,
    ) -> Option<Plane> {
        let k = self.config.plane_k + 1;
        let eps_sq = self.config.epsilon * self.config.epsilon;

        // Initialize plane from seed's normal
        let seed_normal = normals[seed];
        let mut plane = Plane::from_normal_and_point(seed_normal, &points[seed]);
        let mut inliers: Vec<usize> = vec![seed];
        assigned[seed] = true;

        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(seed);
        let mut added_since_refit = 0usize;

        while let Some(current) = queue.pop_front() {
            let p = &points[current];
            let neighbours = tree.nearest_n::<SquaredEuclidean>(&[p.x, p.y, p.z], k);

            for nb in &neighbours {
                let nb_idx = nb.item as usize;
                if assigned[nb_idx] {
                    continue;
                }

                // Validity test: distance to plane AND normal agreement
                let dist_sq = {
                    let d = plane.distance_to(&points[nb_idx]);
                    d * d
                };
                if dist_sq >= eps_sq {
                    continue;
                }
                let normal_dot = plane.normal.dot(&normals[nb_idx]).abs();
                if normal_dot < self.config.normal_threshold {
                    continue;
                }

                // Accept this point
                assigned[nb_idx] = true;
                inliers.push(nb_idx);
                queue.push_back(nb_idx);
                added_since_refit += 1;

                // Refit plane periodically
                if added_since_refit >= self.config.n_refit {
                    if let Some(refined) = Self::fit_plane_ls(points, &inliers) {
                        plane = refined;
                    }
                    added_since_refit = 0;
                }
            }
        }

        if inliers.len() < self.config.min_points {
            // Release points back
            for &i in &inliers {
                assigned[i] = false;
            }
            return None;
        }

        // Final fit
        let mut final_plane = Self::fit_plane_ls(points, &inliers)?;
        final_plane.inliers = inliers;
        Some(final_plane)
    }

    /// Compute PCA normal from a set of point indices (eigenvector of smallest eigenvalue).
    fn pca_normal(points: &[Point3<f64>], indices: &[usize]) -> Vector3<f64> {
        if indices.len() < 3 {
            return Vector3::new(0.0, 0.0, 1.0);
        }
        let n = indices.len() as f64;
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for &i in indices {
            cx += points[i].x;
            cy += points[i].y;
            cz += points[i].z;
        }
        cx /= n;
        cy /= n;
        cz /= n;

        let mut cov = Matrix3::<f64>::zeros();
        for &i in indices {
            let dx = points[i].x - cx;
            let dy = points[i].y - cy;
            let dz = points[i].z - cz;
            cov[(0, 0)] += dx * dx;
            cov[(0, 1)] += dx * dy;
            cov[(0, 2)] += dx * dz;
            cov[(1, 0)] += dy * dx;
            cov[(1, 1)] += dy * dy;
            cov[(1, 2)] += dy * dz;
            cov[(2, 0)] += dz * dx;
            cov[(2, 1)] += dz * dy;
            cov[(2, 2)] += dz * dz;
        }

        let eig = cov.symmetric_eigen();
        let eigenvalues: &nalgebra::Vector3<f64> = &eig.eigenvalues;
        let mut min_idx = 0;
        let mut min_val = eigenvalues[0].abs();
        for i in 1..3 {
            if eigenvalues[i].abs() < min_val {
                min_val = eigenvalues[i].abs();
                min_idx = i;
            }
        }

        let mut normal = eig.eigenvectors.column(min_idx).into_owned();
        let len = normal.norm();
        if len < 1e-10 {
            return Vector3::new(0.0, 0.0, 1.0);
        }
        normal / len
    }

    /// Planarity metric: ratio λ_min / (λ0 + λ1 + λ2). Lower = more planar.
    /// We return 1.0 - ratio so that higher values = better seeds.
    fn planarity(points: &[Point3<f64>], indices: &[usize]) -> f64 {
        if indices.len() < 3 {
            return 0.0;
        }
        let n = indices.len() as f64;
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for &i in indices {
            cx += points[i].x;
            cy += points[i].y;
            cz += points[i].z;
        }
        cx /= n;
        cy /= n;
        cz /= n;

        let mut cov = Matrix3::<f64>::zeros();
        for &i in indices {
            let dx = points[i].x - cx;
            let dy = points[i].y - cy;
            let dz = points[i].z - cz;
            cov[(0, 0)] += dx * dx;
            cov[(0, 1)] += dx * dy;
            cov[(0, 2)] += dx * dz;
            cov[(1, 0)] += dy * dx;
            cov[(1, 1)] += dy * dy;
            cov[(1, 2)] += dy * dz;
            cov[(2, 0)] += dz * dx;
            cov[(2, 1)] += dz * dy;
            cov[(2, 2)] += dz * dz;
        }

        let eig = cov.symmetric_eigen();
        let eigenvalues: &nalgebra::Vector3<f64> = &eig.eigenvalues;
        let sum: f64 = eigenvalues.iter().map(|v| v.abs()).sum();
        if sum < 1e-15 {
            return 0.0;
        }
        let min_ev = eigenvalues.iter().map(|v| v.abs()).fold(f64::MAX, f64::min);
        1.0 - (min_ev / sum)
    }

    // ─── RANSAC (fallback) ──────────────────────────────────────────────────

    fn detect_ransac(&self, points: &[Point3<f64>], max_planes: usize) -> Vec<Plane> {
        let mut available: Vec<usize> = (0..points.len()).collect();
        let mut planes = Vec::new();

        for _ in 0..max_planes.min(self.config.max_planes) {
            let plane = match self.ransac_one(points, &available) {
                Some(p) => p,
                None => break,
            };

            let inlier_set: std::collections::HashSet<usize> =
                plane.inliers.iter().copied().collect();
            available.retain(|i| !inlier_set.contains(i));
            planes.push(plane);

            if available.len() < self.config.min_points {
                break;
            }
        }

        self.merge_similar_planes(points, planes)
    }

    /// Fit a plane to 3 points.
    fn fit_plane_3pts(p0: &Point3<f64>, p1: &Point3<f64>, p2: &Point3<f64>) -> Option<Plane> {
        let v1 = p1 - p0;
        let v2 = p2 - p0;
        let normal = v1.cross(&v2);
        let len = normal.norm();
        if len < 1e-10 {
            return None;
        }
        let normal = normal / len;
        Some(Plane::from_normal_and_point(normal, p0))
    }

    /// Least-squares plane fit via PCA (eigendecomposition on covariance matrix).
    fn fit_plane_ls(points: &[Point3<f64>], indices: &[usize]) -> Option<Plane> {
        if indices.len() < 3 {
            return None;
        }
        let n = indices.len() as f64;
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for &i in indices {
            cx += points[i].x;
            cy += points[i].y;
            cz += points[i].z;
        }
        cx /= n;
        cy /= n;
        cz /= n;
        let centroid = Point3::new(cx, cy, cz);

        let mut cov = Matrix3::zeros();
        for &i in indices {
            let dx = points[i].x - cx;
            let dy = points[i].y - cy;
            let dz = points[i].z - cz;
            cov[(0, 0)] += dx * dx;
            cov[(0, 1)] += dx * dy;
            cov[(0, 2)] += dx * dz;
            cov[(1, 0)] += dy * dx;
            cov[(1, 1)] += dy * dy;
            cov[(1, 2)] += dy * dz;
            cov[(2, 0)] += dz * dx;
            cov[(2, 1)] += dz * dy;
            cov[(2, 2)] += dz * dz;
        }

        let eig = cov.symmetric_eigen();
        let eigenvalues: &nalgebra::Vector3<f64> = &eig.eigenvalues;
        let mut min_idx = 0;
        let mut min_val = eigenvalues[0].abs();
        for i in 1..3 {
            if eigenvalues[i].abs() < min_val {
                min_val = eigenvalues[i].abs();
                min_idx = i;
            }
        }

        let mut normal = eig.eigenvectors.column(min_idx).into_owned();
        let len = normal.norm();
        if len < 1e-10 {
            return None;
        }
        normal /= len;
        if normal.z < 0.0 {
            normal = -normal;
        }

        let mut plane = Plane::from_normal_and_point(normal, &centroid);
        let mut sum_sq = 0.0;
        for &i in indices {
            let d = plane.distance_to(&points[i]);
            sum_sq += d * d;
        }
        plane.rmse = (sum_sq / n).sqrt();
        Some(plane)
    }

    /// RANSAC plane detection on a set of points.
    fn ransac_one(&self, points: &[Point3<f64>], available: &[usize]) -> Option<Plane> {
        if available.len() < self.config.min_points {
            return None;
        }

        let mut rng = thread_rng();
        let mut best_inliers: Vec<usize> = Vec::new();

        for _ in 0..self.config.max_iterations {
            let sample: Vec<&usize> = available.choose_multiple(&mut rng, 3).collect();
            if sample.len() < 3 {
                break;
            }

            let plane = match Self::fit_plane_3pts(
                &points[*sample[0]],
                &points[*sample[1]],
                &points[*sample[2]],
            ) {
                Some(p) => p,
                None => continue,
            };

            if plane.is_near_vertical(self.config.wall_angle_threshold) {
                continue;
            }

            let inliers: Vec<usize> = available
                .iter()
                .filter(|&&i| plane.distance_to(&points[i]) < self.config.epsilon)
                .copied()
                .collect();

            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
            }
        }

        if best_inliers.len() < self.config.min_points {
            return None;
        }

        let mut plane = Self::fit_plane_ls(points, &best_inliers)?;

        // Recompute inliers with refined plane
        let refined_inliers: Vec<usize> = available
            .iter()
            .filter(|&&i| plane.distance_to(&points[i]) < self.config.epsilon)
            .copied()
            .collect();
        plane.inliers = refined_inliers;

        if plane.inliers.len() < self.config.min_points || plane.is_near_vertical(self.config.wall_angle_threshold) {
            return None;
        }

        Some(plane)
    }

    // ─── Shared utilities ───────────────────────────────────────────────────

    /// Merge planes with similar orientations and proximity.
    fn merge_similar_planes(&self, points: &[Point3<f64>], planes: Vec<Plane>) -> Vec<Plane> {
        // Iterate to fixed point.  A single pass refits each merged cluster
        // to least-squares, which often shifts the refitted plane's normal
        // and offset enough that previously-incompatible neighbours now
        // satisfy the angle + distance criteria.  Without this loop, a
        // noisy hip roof gets stuck with 5-7 near-duplicate ~33° planes
        // that cascade into thousands of arrangement slivers.
        let mut current = planes;
        loop {
            let before = current.len();
            current = self.merge_similar_planes_once(points, current);
            if current.len() == before {
                return current;
            }
        }
    }

    fn merge_similar_planes_once(&self, points: &[Point3<f64>], planes: Vec<Plane>) -> Vec<Plane> {
        if planes.len() <= 1 {
            return planes;
        }

        let angle_thresh = self.config.merge_angle_degrees.to_radians();
        let dist_thresh = self.config.merge_distance;

        let mut merged = vec![false; planes.len()];
        let mut result = Vec::new();

        for i in 0..planes.len() {
            if merged[i] {
                continue;
            }
            let mut combined_inliers = planes[i].inliers.clone();

            for j in (i + 1)..planes.len() {
                if merged[j] {
                    continue;
                }
                let angle = planes[i].normal.dot(&planes[j].normal).abs().acos();
                if angle > angle_thresh {
                    continue;
                }
                let ci = Self::centroid_of_inliers(points, &planes[i].inliers);
                let dist = planes[j].distance_to(&ci);
                if dist > dist_thresh {
                    continue;
                }
                // Prevent merging planes whose inlier clouds are far apart in 2D
                let cj = Self::centroid_of_inliers(points, &planes[j].inliers);
                let dx_2d = ci.x - cj.x;
                let dy_2d = ci.y - cj.y;
                if (dx_2d * dx_2d + dy_2d * dy_2d).sqrt() > self.config.merge_centroid_2d_distance {
                    continue;
                }
                combined_inliers.extend_from_slice(&planes[j].inliers);
                merged[j] = true;
            }

            if let Some(mut p) = Self::fit_plane_ls(points, &combined_inliers) {
                p.inliers = combined_inliers;
                result.push(p);
            }
        }

        result
    }

    fn centroid_of_inliers(points: &[Point3<f64>], inliers: &[usize]) -> Point3<f64> {
        let n = inliers.len() as f64;
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for &i in inliers {
            cx += points[i].x;
            cy += points[i].y;
            cz += points[i].z;
        }
        Point3::new(cx / n, cy / n, cz / n)
    }
}

/// Reject planes whose inliers are predominantly vegetation based on NDVI.
///
/// A plane is rejected when fewer than `min_building_fraction` of its inliers
/// have NDVI below `ndvi_threshold`. Points without finite NDVI are assumed
/// non-vegetation (they pass).
pub fn filter_vegetation_planes(
    planes: Vec<Plane>,
    ndvi: &[f32],
    ndvi_threshold: f32,
    min_building_fraction: f64,
) -> (Vec<Plane>, usize) {
    let mut kept = Vec::new();
    let mut n_rejected = 0usize;
    for plane in planes {
        let n_total = plane.inliers.len();
        if n_total == 0 {
            kept.push(plane);
            continue;
        }
        let n_building = plane.inliers.iter()
            .filter(|&&i| !ndvi[i].is_finite() || ndvi[i] <= ndvi_threshold)
            .count();
        let fraction = n_building as f64 / n_total as f64;
        if fraction >= min_building_fraction {
            kept.push(plane);
        } else {
            n_rejected += 1;
        }
    }
    (kept, n_rejected)
}

/// Intersect two planes, returning a line (point + direction) if they're not parallel.
pub fn intersect_planes(a: &Plane, b: &Plane) -> Option<(Point3<f64>, Vector3<f64>)> {
    let dir = a.normal.cross(&b.normal);
    let len = dir.norm();
    if len < 1e-10 {
        return None;
    }
    let dir = dir / len;

    let n1 = a.normal;
    let n2 = b.normal;
    let d1 = -a.d;
    let d2 = -b.d;

    let denom = n1.dot(&n1) * n2.dot(&n2) - n1.dot(&n2).powi(2);
    if denom.abs() < 1e-10 {
        return None;
    }
    let c1 = (d1 * n2.dot(&n2) - d2 * n1.dot(&n2)) / denom;
    let c2 = (d2 * n1.dot(&n1) - d1 * n1.dot(&n2)) / denom;
    let point = Point3::from(c1 * n1 + c2 * n2);

    Some((point, dir))
}

/// Evaluate the Z on a plane-plane intersection line at a given (x, y).
///
/// Given the intersection line `origin + t * dir`, finds t such that the
/// projected (x, y) matches, then returns `origin.z + t * dir.z`.
pub fn intersection_z_at(origin: &Point3<f64>, dir: &Vector3<f64>, x: f64, y: f64) -> Option<f64> {
    let t = if dir.x.abs() > dir.y.abs() {
        if dir.x.abs() < 1e-12 {
            return None;
        }
        (x - origin.x) / dir.x
    } else if dir.y.abs() > 1e-12 {
        (y - origin.y) / dir.y
    } else {
        return None;
    };
    Some(origin.z + t * dir.z)
}

/// Clip a 3D line to a 2D bounding box, returning a 2D segment.
pub fn clip_line_to_bbox(
    origin: &Point3<f64>,
    dir: &Vector3<f64>,
    bbox: &[f64; 4],
) -> Option<([f64; 2], [f64; 2])> {
    let (mut t_min, mut t_max) = (-1e10_f64, 1e10_f64);

    for axis in 0..2 {
        let (o, d, lo, hi) = match axis {
            0 => (origin.x, dir.x, bbox[0], bbox[2]),
            _ => (origin.y, dir.y, bbox[1], bbox[3]),
        };
        if d.abs() < 1e-15 {
            if o < lo || o > hi {
                return None;
            }
        } else {
            let t1 = (lo - o) / d;
            let t2 = (hi - o) / d;
            let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
            t_min = t_min.max(t_near);
            t_max = t_max.min(t_far);
            if t_min > t_max {
                return None;
            }
        }
    }

    let p1 = [origin.x + dir.x * t_min, origin.y + dir.y * t_min];
    let p2 = [origin.x + dir.x * t_max, origin.y + dir.y * t_max];

    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    if (dx * dx + dy * dy) < 1e-10 {
        return None;
    }

    Some((p1, p2))
}

/// Decide whether a clipped ridge segment corresponds to a real shared
/// roof edge between two planes (vs. a phantom intersection of two
/// surfaces that never touch on the roof).
///
/// Test: project each plane's inlier (x,y) onto the segment direction
/// to get a parametric range along the line.  A real shared edge
/// requires both ranges to overlap by a meaningful stretch.  Two
/// planes whose footprints don't share extent along their intersection
/// line cannot have a real shared roof edge there — the line is a
/// phantom and emitting it shatters the arrangement into slivers.
///
/// (Note: we project ALL inliers onto the line direction without any
/// perpendicular filter.  Inliers of a gable's slope are typically
/// metres perpendicular from the ridge — using only the t-coordinate
/// captures the line-aligned extent of each plane's support.)
pub fn ridge_has_inlier_support(
    p1: &[f64; 2],
    p2: &[f64; 2],
    inliers_a: &[(f64, f64)],
    inliers_b: &[(f64, f64)],
) -> bool {
    if inliers_a.is_empty() || inliers_b.is_empty() {
        return false;
    }
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-12 {
        return false;
    }
    let len = len2.sqrt();
    let project_range = |pts: &[(f64, f64)]| -> (f64, f64) {
        let mut tmin = f64::INFINITY;
        let mut tmax = f64::NEG_INFINITY;
        for &(x, y) in pts {
            let ex = x - p1[0];
            let ey = y - p1[1];
            let t = (ex * dx + ey * dy) / len2;
            if t < tmin { tmin = t; }
            if t > tmax { tmax = t; }
        }
        (tmin, tmax)
    };
    let (ta_min, ta_max) = project_range(inliers_a);
    let (tb_min, tb_max) = project_range(inliers_b);
    let lo = ta_min.max(tb_min);
    let hi = ta_max.min(tb_max);
    if hi <= lo { return false; }
    // Require shared stretch of at least 1 m AND at least 10% of segment.
    let overlap_len = (hi - lo) * len;
    overlap_len >= 1.0 && overlap_len >= 0.10 * len
}

/// Trim a clipped ridge segment to the longest sub-segment that is
/// supported by inliers from *both* planes within a 2D proximity radius.
///
/// Rationale: the geometric intersection of two plane equations is a
/// world-spanning line.  Bbox-clipping only removes the parts outside
/// the building.  But two roof surfaces that face different cardinal
/// directions (e.g. two south-facing dormers, or two limbs of an
/// H-shaped hip roof) can have overlapping (x,y) extents without ever
/// sharing a real edge — the "ridge" produced by their plane equations
/// is fictional.  Emitting it produces phantom subdivisions that
/// shatter the arrangement into hundreds of slivers.
///
/// A real ridge has, at every point along it, points from both planes
/// within roof-detail distance.  We sample the bbox-clipped segment and
/// keep the longest contiguous stretch where both planes have an
/// inlier within `proximity` metres.
pub fn trim_ridge_to_inlier_support(
    p1: &[f64; 2],
    p2: &[f64; 2],
    inliers_a: &[(f64, f64)],
    inliers_b: &[(f64, f64)],
    proximity: f64,
) -> Option<([f64; 2], [f64; 2])> {
    if inliers_a.is_empty() || inliers_b.is_empty() {
        return None;
    }
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    let length = (dx * dx + dy * dy).sqrt();
    if length < 1e-6 {
        return None;
    }
    // Sample density: ~ one sample per `proximity` metres, capped.
    let samples = ((length / proximity).ceil() as usize).clamp(8, 128);
    let r2 = proximity * proximity;
    let mut valid = Vec::with_capacity(samples);
    for k in 0..samples {
        let t = k as f64 / (samples - 1) as f64;
        let sx = p1[0] + t * dx;
        let sy = p1[1] + t * dy;
        let near_a = inliers_a.iter().any(|&(x, y)| {
            let dx = x - sx;
            let dy = y - sy;
            dx * dx + dy * dy < r2
        });
        let near_b = if near_a {
            inliers_b.iter().any(|&(x, y)| {
                let dx = x - sx;
                let dy = y - sy;
                dx * dx + dy * dy < r2
            })
        } else {
            false
        };
        valid.push(near_a && near_b);
    }
    // Longest contiguous run of true.
    let mut best_start = 0usize;
    let mut best_len = 0usize;
    let mut cur_start = 0usize;
    let mut cur_len = 0usize;
    for (k, &v) in valid.iter().enumerate() {
        if v {
            if cur_len == 0 { cur_start = k; }
            cur_len += 1;
            if cur_len > best_len {
                best_len = cur_len;
                best_start = cur_start;
            }
        } else {
            cur_len = 0;
        }
    }
    // Require at least two contiguous samples (a single point is noise).
    if best_len < 2 {
        return None;
    }
    let t1 = best_start as f64 / (samples - 1) as f64;
    let t2 = (best_start + best_len - 1) as f64 / (samples - 1) as f64;
    let q1 = [p1[0] + t1 * dx, p1[1] + t1 * dy];
    let q2 = [p1[0] + t2 * dx, p1[1] + t2 * dy];
    Some((q1, q2))
}

/// Compute intersection of two 2D segments (a0->a1) and (b0->b1).
/// Returns `Some((point, t_a, t_b))` for strict interior crossings only
/// (eps < t < 1-eps), excluding endpoint coincidences which are handled by snapping.
pub fn segment_intersection_2d(
    a0: &[f64; 2],
    a1: &[f64; 2],
    b0: &[f64; 2],
    b1: &[f64; 2],
) -> Option<([f64; 2], f64, f64)> {
    let eps = 1e-8;
    let dx_a = a1[0] - a0[0];
    let dy_a = a1[1] - a0[1];
    let dx_b = b1[0] - b0[0];
    let dy_b = b1[1] - b0[1];

    let denom = dx_a * dy_b - dy_a * dx_b;
    if denom.abs() < eps {
        return None; // Parallel or collinear
    }

    let dx_ab = b0[0] - a0[0];
    let dy_ab = b0[1] - a0[1];

    let t = (dx_ab * dy_b - dy_ab * dx_b) / denom;
    let s = (dx_ab * dy_a - dy_ab * dx_a) / denom;

    if t <= eps || t >= 1.0 - eps || s <= eps || s >= 1.0 - eps {
        return None; // Not a strict interior crossing
    }

    let px = a0[0] + t * dx_a;
    let py = a0[1] + t * dy_a;
    Some(([px, py], t, s))
}

/// Pre-split ridge segments at all mutual intersections and at intersections
/// with footprint boundary edges. Returns non-crossing sub-segments with
/// inherited metadata.
///
/// Footprint edges are NOT split — they are already in the CDT as a closed
/// ring of constraints. Only ridgeline segments are subdivided.
pub fn split_segments_at_intersections<M: Clone>(
    footprint_edges: &[([f64; 2], [f64; 2])],
    ridge_segments: &[([f64; 2], [f64; 2], M)],
    snap_tolerance: f64,
) -> Vec<([f64; 2], [f64; 2], M)> {
    if ridge_segments.is_empty() {
        return Vec::new();
    }

    let n = ridge_segments.len();

    // Phase 1: Collect split parameters for each ridge segment
    let mut splits: Vec<Vec<f64>> = vec![Vec::new(); n];

    // Ridge-ridge intersections
    for i in 0..n {
        for j in (i + 1)..n {
            if let Some((_pt, t_a, t_b)) = segment_intersection_2d(
                &ridge_segments[i].0,
                &ridge_segments[i].1,
                &ridge_segments[j].0,
                &ridge_segments[j].1,
            ) {
                splits[i].push(t_a);
                splits[j].push(t_b);
            }
        }
    }

    // Ridge-footprint intersections
    for i in 0..n {
        for fp_edge in footprint_edges {
            if let Some((_pt, t_a, _t_b)) = segment_intersection_2d(
                &ridge_segments[i].0,
                &ridge_segments[i].1,
                &fp_edge.0,
                &fp_edge.1,
            ) {
                splits[i].push(t_a);
            }
        }
    }

    // Phase 2: Split each ridge into sub-segments
    let mut result: Vec<([f64; 2], [f64; 2], M)> = Vec::new();

    for (i, seg) in ridge_segments.iter().enumerate() {
        let (p0, p1, meta) = seg;
        let dx = p1[0] - p0[0];
        let dy = p1[1] - p0[1];
        let seg_len = (dx * dx + dy * dy).sqrt();

        let t_eps = if seg_len > 0.0 {
            snap_tolerance / seg_len
        } else {
            continue;
        };

        // Sort and deduplicate t values
        splits[i].push(0.0);
        splits[i].push(1.0);
        splits[i].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        splits[i].dedup_by(|a, b| (*a - *b).abs() < t_eps);

        // Create sub-segments
        for w in splits[i].windows(2) {
            let t0 = w[0];
            let t1 = w[1];
            if (t1 - t0) * seg_len < snap_tolerance {
                continue; // Degenerate sub-segment
            }
            let sp0 = [p0[0] + t0 * dx, p0[1] + t0 * dy];
            let sp1 = [p0[0] + t1 * dx, p0[1] + t1 * dy];
            result.push((sp0, sp1, meta.clone()));
        }
    }

    // Phase 3: Snap near-coincident endpoints
    let mut canonical: Vec<[f64; 2]> = Vec::new();
    let tol_sq = snap_tolerance * snap_tolerance;

    let snap = |pt: &mut [f64; 2], canonical: &mut Vec<[f64; 2]>| {
        for c in canonical.iter() {
            let dx = pt[0] - c[0];
            let dy = pt[1] - c[1];
            if dx * dx + dy * dy < tol_sq {
                pt[0] = c[0];
                pt[1] = c[1];
                return;
            }
        }
        canonical.push(*pt);
    };

    for seg in &mut result {
        snap(&mut seg.0, &mut canonical);
        snap(&mut seg.1, &mut canonical);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_intersection_crossing() {
        // X-shaped crossing at (0.5, 0.5)
        let a0 = [0.0, 0.0];
        let a1 = [1.0, 1.0];
        let b0 = [1.0, 0.0];
        let b1 = [0.0, 1.0];
        let result = segment_intersection_2d(&a0, &a1, &b0, &b1);
        assert!(result.is_some());
        let (pt, t, s) = result.unwrap();
        assert!((pt[0] - 0.5).abs() < 1e-6);
        assert!((pt[1] - 0.5).abs() < 1e-6);
        assert!((t - 0.5).abs() < 1e-6);
        assert!((s - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_segment_intersection_parallel() {
        let a0 = [0.0, 0.0];
        let a1 = [1.0, 0.0];
        let b0 = [0.0, 1.0];
        let b1 = [1.0, 1.0];
        assert!(segment_intersection_2d(&a0, &a1, &b0, &b1).is_none());
    }

    #[test]
    fn test_segment_intersection_collinear() {
        let a0 = [0.0, 0.0];
        let a1 = [2.0, 0.0];
        let b0 = [1.0, 0.0];
        let b1 = [3.0, 0.0];
        assert!(segment_intersection_2d(&a0, &a1, &b0, &b1).is_none());
    }

    #[test]
    fn test_segment_intersection_t_shape_endpoint() {
        // Endpoint of B touches interior of A — should return None (excluded by eps)
        let a0 = [0.0, 0.0];
        let a1 = [2.0, 0.0];
        let b0 = [1.0, 1.0];
        let b1 = [1.0, 0.0]; // touches A at (1,0) which is b1 (s=1)
        assert!(segment_intersection_2d(&a0, &a1, &b0, &b1).is_none());
    }

    #[test]
    fn test_split_crossing_ridges() {
        // Two crossing ridges, no footprint edges
        let r0 = ([0.0, 0.0], [1.0, 1.0], ());
        let r1 = ([1.0, 0.0], [0.0, 1.0], ());
        let result = split_segments_at_intersections(&[], &[r0, r1], 1e-3);
        // Each ridge split into 2 sub-segments → 4 total
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_split_ridge_crosses_footprint_edge() {
        // One ridge crossing one footprint edge
        let fp = [([0.0, 0.5], [1.0, 0.5])]; // horizontal edge at y=0.5
        let ridge = ([0.5, 0.0], [0.5, 1.0], ()); // vertical ridge
        let result = split_segments_at_intersections(&fp, &[ridge], 1e-3);
        assert_eq!(result.len(), 2);
        // Sub-segment endpoints should be near (0.5, 0.5)
        let mid0 = result[0].1;
        let mid1 = result[1].0;
        assert!((mid0[1] - 0.5).abs() < 1e-6);
        assert!((mid1[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_split_no_intersections() {
        // Parallel ridges — no splitting needed
        let r0 = ([0.0, 0.0], [1.0, 0.0], ());
        let r1 = ([0.0, 1.0], [1.0, 1.0], ());
        let result = split_segments_at_intersections(&[], &[r0, r1], 1e-3);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_split_degenerate_removal() {
        // Two ridges that cross very close to an endpoint — the tiny sub-segment should be removed
        let r0 = ([0.0, 0.0], [1.0, 1.0], ());
        let r1 = ([0.999, 0.0], [0.0, 0.999], ()); // crosses r0 near (0.5, 0.5)
        let result = split_segments_at_intersections(&[], &[r0, r1], 1e-3);
        // All sub-segments should be longer than snap_tolerance
        for (p0, p1, _) in &result {
            let dx = p1[0] - p0[0];
            let dy = p1[1] - p0[1];
            assert!((dx * dx + dy * dy).sqrt() >= 1e-3);
        }
    }
}
