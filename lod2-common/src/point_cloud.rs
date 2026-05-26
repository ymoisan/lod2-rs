use nalgebra::Point3;

#[derive(Debug, Clone)]
pub struct PointCloud {
    pub positions: Vec<Point3<f64>>,
    pub classifications: Vec<u8>,
    /// Precomputed NDVI per point: (NIR - Red) / (NIR + Red).
    /// None when the input has no spectral data (format 6 or earlier).
    pub ndvi: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct PointCloudStats {
    pub z_min: f64,
    pub z_max: f64,
    pub z_mean: f64,
    pub z_05p: f64,
    pub z_10p: f64,
    pub z_50p: f64,
    pub z_70p: f64,
    pub z_95p: f64,
    pub x_mean: f64,
    pub y_mean: f64,
    pub count: usize,
}

impl PointCloud {
    pub fn new() -> Self {
        Self { positions: Vec::new(), classifications: Vec::new(), ndvi: None }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            positions: Vec::with_capacity(cap),
            classifications: Vec::with_capacity(cap),
            ndvi: None,
        }
    }

    /// Push a point with default classification (class 6 = building).
    pub fn push(&mut self, p: Point3<f64>) {
        self.positions.push(p);
        self.classifications.push(6);
        if let Some(ref mut ndvi) = self.ndvi {
            ndvi.push(f32::NAN);
        }
    }

    /// Push a point with explicit LAS classification.
    pub fn push_classified(&mut self, p: Point3<f64>, class: u8) {
        self.positions.push(p);
        self.classifications.push(class);
        if let Some(ref mut ndvi) = self.ndvi {
            ndvi.push(f32::NAN);
        }
    }

    /// Push a point with classification and precomputed NDVI.
    /// Lazily initializes the ndvi vec (backfilling NAN for earlier points).
    pub fn push_with_ndvi(&mut self, p: Point3<f64>, class: u8, ndvi_value: f32) {
        let n = self.positions.len();
        self.positions.push(p);
        self.classifications.push(class);
        match self.ndvi {
            Some(ref mut v) => v.push(ndvi_value),
            None => {
                let mut v = vec![f32::NAN; n];
                v.push(ndvi_value);
                self.ndvi = Some(v);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Whether this point cloud has spectral (NDVI) data.
    pub fn has_spectral(&self) -> bool {
        self.ndvi.is_some()
    }

    pub fn extend(&mut self, other: &PointCloud) {
        self.positions.extend_from_slice(&other.positions);
        self.classifications.extend_from_slice(&other.classifications);
        match (&mut self.ndvi, &other.ndvi) {
            (Some(ref mut a), Some(b)) => a.extend_from_slice(b),
            (Some(ref mut a), None) => a.extend(std::iter::repeat(f32::NAN).take(other.len())),
            (None, Some(b)) => {
                let mut v = vec![f32::NAN; self.positions.len() - other.len()];
                v.extend_from_slice(b);
                self.ndvi = Some(v);
            }
            (None, None) => {}
        }
    }

    /// Return a new PointCloud containing only points with the given classification.
    pub fn filter_by_class(&self, class: u8) -> PointCloud {
        let mut out = PointCloud::new();
        for (i, &c) in self.classifications.iter().enumerate() {
            if c == class {
                if let Some(ref ndvi) = self.ndvi {
                    out.push_with_ndvi(self.positions[i], c, ndvi[i]);
                } else {
                    out.push_classified(self.positions[i], c);
                }
            }
        }
        out
    }

    /// Return a new PointCloud excluding points with NDVI above threshold.
    /// Points without NDVI data are always kept.
    pub fn filter_vegetation(&self, ndvi_threshold: f32) -> (PointCloud, usize) {
        let mut out = PointCloud::new();
        let mut n_removed = 0usize;
        for i in 0..self.len() {
            let dominated_by_vegetation = self.ndvi.as_ref()
                .map(|v| v[i].is_finite() && v[i] > ndvi_threshold)
                .unwrap_or(false);
            if dominated_by_vegetation {
                n_removed += 1;
            } else if let Some(ref ndvi) = self.ndvi {
                out.push_with_ndvi(self.positions[i], self.classifications[i], ndvi[i]);
            } else {
                out.push_classified(self.positions[i], self.classifications[i]);
            }
        }
        (out, n_removed)
    }

    pub fn compute_statistics(&self) -> PointCloudStats {
        if self.positions.is_empty() {
            return PointCloudStats {
                z_min: 0.0, z_max: 0.0, z_mean: 0.0, z_05p: 0.0, z_10p: 0.0, z_50p: 0.0, z_70p: 0.0, z_95p: 0.0,
                x_mean: 0.0, y_mean: 0.0, count: 0,
            };
        }
        let mut zs: Vec<f64> = self.positions.iter().map(|p| p.z).collect();
        zs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = zs.len();
        let z_sum: f64 = zs.iter().sum();
        let x_sum: f64 = self.positions.iter().map(|p| p.x).sum();
        let y_sum: f64 = self.positions.iter().map(|p| p.y).sum();
        PointCloudStats {
            z_min: zs[0],
            z_max: zs[n - 1],
            z_mean: z_sum / n as f64,
            z_05p: zs[(n as f64 * 0.05) as usize],
            z_10p: zs[(n as f64 * 0.10) as usize],
            z_50p: zs[n / 2],
            z_70p: zs[(n as f64 * 0.7) as usize],
            z_95p: zs[((n as f64 * 0.95) as usize).min(n - 1)],
            x_mean: x_sum / n as f64,
            y_mean: y_sum / n as f64,
            count: n,
        }
    }
}

/// Compute NDVI from Red and NIR channel values.
/// Returns NAN if both channels are zero.
pub fn compute_ndvi(red: u16, nir: u16) -> f32 {
    let r = red as f32;
    let n = nir as f32;
    let denom = n + r;
    if denom == 0.0 { f32::NAN } else { (n - r) / denom }
}

impl Default for PointCloud {
    fn default() -> Self {
        Self::new()
    }
}
