use nalgebra::Point3;

#[derive(Debug, Clone)]
pub struct PointCloud {
    pub positions: Vec<Point3<f64>>,
}

#[derive(Debug, Clone)]
pub struct PointCloudStats {
    pub z_min: f64,
    pub z_max: f64,
    pub z_mean: f64,
    pub z_50p: f64,
    pub z_70p: f64,
    pub count: usize,
}

impl PointCloud {
    pub fn new() -> Self {
        Self { positions: Vec::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self { positions: Vec::with_capacity(cap) }
    }

    pub fn push(&mut self, p: Point3<f64>) {
        self.positions.push(p);
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn extend(&mut self, other: &PointCloud) {
        self.positions.extend_from_slice(&other.positions);
    }

    pub fn compute_statistics(&self) -> PointCloudStats {
        if self.positions.is_empty() {
            return PointCloudStats {
                z_min: 0.0, z_max: 0.0, z_mean: 0.0, z_50p: 0.0, z_70p: 0.0, count: 0,
            };
        }
        let mut zs: Vec<f64> = self.positions.iter().map(|p| p.z).collect();
        zs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = zs.len();
        let z_sum: f64 = zs.iter().sum();
        PointCloudStats {
            z_min: zs[0],
            z_max: zs[n - 1],
            z_mean: z_sum / n as f64,
            z_50p: zs[n / 2],
            z_70p: zs[(n as f64 * 0.7) as usize],
            count: n,
        }
    }
}

impl Default for PointCloud {
    fn default() -> Self {
        Self::new()
    }
}
