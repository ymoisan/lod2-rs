use nalgebra::{Matrix3, Point3, Vector3};
use rand::seq::SliceRandom;
use rand::thread_rng;

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

#[derive(Debug, Clone)]
pub struct RansacConfig {
    pub epsilon: f64,
    pub max_iterations: usize,
    pub min_points: usize,
    pub max_planes: usize,
    pub wall_angle_threshold: f64,
    pub merge_angle_degrees: f64,
    pub merge_distance: f64,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.3,
            max_iterations: 1000,
            min_points: 15,
            max_planes: 20,
            wall_angle_threshold: 70.0,
            merge_angle_degrees: 7.5,
            merge_distance: 0.5,
        }
    }
}

pub struct PlaneDetector {
    config: RansacConfig,
}

impl PlaneDetector {
    pub fn new(config: RansacConfig) -> Self {
        Self { config }
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

    /// Least-squares plane fit via PCA (SVD on centered points).
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

    /// Detect multiple planes from a point cloud.
    pub fn detect_multiple(&self, points: &[Point3<f64>], max_planes: usize) -> Vec<Plane> {
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

    /// Merge planes with similar orientations and proximity.
    fn merge_similar_planes(&self, points: &[Point3<f64>], planes: Vec<Plane>) -> Vec<Plane> {
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
