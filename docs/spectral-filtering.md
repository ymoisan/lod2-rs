# Spectral Filtering (NDVI)

## Motivation

Aerial LiDAR classification sometimes assigns tree canopy points to class 6
(building). When these misclassified points cluster spatially, they can form
spurious roof planes during RANSAC — producing ghost buildings or distorting
real roof geometry.

When the input point cloud carries **RGB + NIR** channels (LAS format 7 or 8),
we can compute the Normalized Difference Vegetation Index (NDVI) per point and
use it to reject vegetation before and after plane fitting.

## How It Works

### NDVI Computation

NDVI exploits the fact that chlorophyll strongly reflects near-infrared (NIR)
and absorbs red light:

```
NDVI = (NIR - Red) / (NIR + Red)
```

| Surface            | Typical NDVI |
|--------------------|-------------|
| Dense vegetation   | 0.4 – 0.9  |
| Sparse vegetation  | 0.2 – 0.4  |
| Bare soil          | 0.1 – 0.2  |
| Built surfaces     | -0.1 – 0.1 |
| Water              | -0.3 – 0.0 |

NDVI is computed once per point at read time and stored as `f32` alongside the
point coordinates. Cost: 4 bytes/point (40 MB for 10M points).

### Two-Level Filtering

**Level 1 — Point pre-filtering** (before RANSAC):
Class-6 points with NDVI above the threshold are removed from the building
point set before any reconstruction begins. This prevents vegetation points
from participating in plane fitting at all.

**Level 2 — Plane post-validation** (after RANSAC):
After planes are detected, each plane's inlier set is checked. If fewer than
60% of the inliers are non-vegetation (NDVI ≤ threshold), the plane is
rejected. This catches cases where a flat canopy surface appears geometrically
planar but is spectrally vegetation.

## Usage

```bash
buildex \
  --pointcloud input_rgb.laz \
  --footprints footprints.gpkg \
  --output output/ \
  --ndvi-threshold 0.3
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ndvi-threshold` | *(disabled)* | NDVI value above which class-6 points are considered vegetation. Typical: **0.25–0.35**. |

The threshold is optional. When omitted, no spectral filtering occurs and
buildex behaves identically to before (fully backward-compatible).

### Input Requirements

- LAS/LAZ file with point format **7** (RGB) or **8** (RGB + NIR)
- NIR channel must be populated (format 8) for meaningful NDVI
- If only RGB is available (format 7, no NIR), NDVI cannot be computed and
  spectral filtering is silently skipped

## Output Diagnostics

When spectral filtering is active, the following attributes are added to each
building in the CityJSONL output:

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_spectral_removed` | int | Number of class-6 points removed by NDVI filter for this footprint |
| `mean_ndvi` | float | Mean NDVI of the remaining building points (after filtering) |

These can be used to audit filtering behavior:
- High `n_spectral_removed` suggests heavy tree overhang on the building
- `mean_ndvi` close to 0 confirms a clean built surface; values > 0.2 warrant
  inspection (possible green roof or residual vegetation)

## Limitations

1. **Seasonal effects**: Deciduous trees in winter have low NDVI (no leaves)
   but still physically occlude the roof below. Spectral filtering won't help
   in leaf-off conditions.

2. **Green roofs / rooftop gardens**: These have high NDVI but ARE buildings.
   The plane post-validation (Level 2) partially mitigates this because green
   roof surfaces are planar — if ≥60% of inliers are "building-like", the
   plane is kept.

3. **Dark roofs in shadow**: Shadowed surfaces can have anomalous spectral
   values. The threshold should not be set too aggressively low (< 0.2).

4. **Partial coverage**: If only part of the point cloud has RGB+NIR (e.g.,
   from a spatial join), points without spectral data are assumed
   non-vegetation and always kept.

## Architecture

```
Input LAS (format 8)
    │
    ▼  [read + compute NDVI]
PointCloud { positions, classifications, ndvi: Option<Vec<f32>> }
    │
    ▼  [crop to footprint]
building_pts (class 6, inside footprint)
    │
    ▼  [Level 1: filter_vegetation(threshold)]
building_pts_filtered  ─── n_spectral_removed logged
    │
    ▼  [RANSAC plane detection]
candidate planes
    │
    ▼  [Level 2: filter_vegetation_planes(planes, ndvi, threshold, 0.6)]
validated planes  ─── rejected planes logged
    │
    ▼  [roof reconstruction]
BuildingGeometry
```

### Key Files

| File | Role |
|------|------|
| `lod2-common/src/point_cloud.rs` | `PointCloud` struct, `ndvi` field, `filter_vegetation()`, `compute_ndvi()` |
| `lod2-common/src/plane.rs` | `filter_vegetation_planes()` — Level 2 validation |
| `lod2-common/src/pipeline.rs` | `--ndvi-threshold` CLI arg, Level 1 integration in reconstruction loop |
| `hybrid/src/main.rs` | NDVI computation at read time, Level 2 integration in `reconstruct()` |
