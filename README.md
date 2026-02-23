# lod2-rs

This repo hosts [LoD2](https://osmbuildings.org/blog/2018-02-28_level_of_detail/) 3D buildings reconstruction methods from aerial lidar with a deliberate intent to use Rust.  Pretty much all code here will be written by AI, following careful planning and continuous "human steering" (whence the choice of the [unlicense](https://unlicense.org/)).

For background see [A new benchmark on LoD 2 building reconstruction from aerial lidar and footprints](https://isprs-archives.copernicus.org/articles/XLVIII-1-W6-2025/83/2025/isprs-archives-XLVIII-1-W6-2025-83-2025.pdf).

---

## Workspace structure

```
lod2-rs/
├── lod2-common/       # shared I/O, geometry and pipeline crate
├── plane-extrude/     # TerraScan-style method
├── arrangement/       # City3D-style method
└── graph-cut/         # Roofer-style method
```

### lod2-common

Shared library used by all three reconstruction binaries.  Provides:

| Module | Contents |
|--------|----------|
| `las_reader` | Read `.laz` / `.las` point clouds via the `las` crate |
| `vector_reader` | Read building footprints from `.gpkg` / any OGR source via `gdal` |
| `point_cloud` | `PointCloud` struct and percentile statistics (`z_min`, `z_50p`, `z_70p`, …) |
| `polygon` | `LinearRing`, `Polygon3D`, point-in-polygon test, 2D area / centroid |
| `plane` | RANSAC plane detection, least-squares PCA refinement, plane merging, plane intersection |
| `mesh` | `Mesh` / `Face` / `SemanticSurface` / `BuildingGeometry` — semantic LoD2 mesh representation |
| `cityjson` | Write [CityJSONL](https://www.cityjson.org/cityjsonl/) (one feature per line) |
| `pipeline` | CLI args (`clap`), `Reconstructor` trait, parallel `rayon` driver with `catch_unwind` fallback to flat roof |

### plane-extrude — TerraScan style

1. Detect roof planes with RANSAC.
2. Project each plane's inlier points to 2D and compute their convex hull.
3. Extrude each hull as walls down to ground level.

**Characteristic:** produces the highest vertex and face counts per building but with LoD1-style convex shapes; concavities (porches, U-shapes) are filled in by the convex hull step.

### arrangement — City3D style

1. Detect roof planes with RANSAC.
2. Compute pairwise plane intersection lines (ridgelines) and clip them to the footprint bounding box.
3. Build a Constrained Delaunay Triangulation (CDT) with the footprint boundary and all ridgelines as constraints (via `spade`).
4. Score each triangle against all planes and assign it to the best-fitting one.
5. Extrude walls from each triangle edge that borders a differently-labelled triangle or the footprint boundary.
6. Falls back to flat-roof extrusion on CDT constraint failures.

### graph-cut — Roofer style

Same CDT-based arrangement as above, but uses a **greedy data-cost labeling** strategy: each triangle face is assigned the plane label that minimises the mean squared distance of nearby inlier points to the plane.  This is the architectural scaffold for a full alpha-expansion graph-cut; the energy minimisation step is prepared but not yet connected.

---

## Dependencies

Key crates (see `Cargo.toml` for pinned versions):

| Crate | Role |
|-------|------|
| `nalgebra` | Linear algebra (vectors, matrices, eigendecomposition) |
| `spade` | Constrained Delaunay Triangulation |
| `las` | LAS/LAZ point cloud I/O |
| `gdal` | OGR/GDAL vector I/O (footprints) |
| `rayon` | Data-parallel building reconstruction |
| `rand` | RANSAC random sampling |
| `serde_json` | CityJSONL serialisation |
| `clap` | CLI argument parsing |
| `tracing` | Structured logging |
| `rstar` | R-tree spatial index |
| `kiddo` | k-d tree nearest-neighbour queries |
| `anyhow` / `thiserror` | Error handling |

---

## Building

The `gdal` crate requires GDAL development headers.  Inside the `roofer` Nix dev-shell:

```bash
cd /path/to/lod2-rs
cargo build --release
```

Binaries are placed in `target/release/`.

---

## Running

Each binary accepts the same three arguments:

```bash
plane-extrude --pointcloud INPUT.laz --footprints INPUT.gpkg --output OUTPUT_DIR
arrangement   --pointcloud INPUT.laz --footprints INPUT.gpkg --output OUTPUT_DIR
graph-cut     --pointcloud INPUT.laz --footprints INPUT.gpkg --output OUTPUT_DIR
```

Output is a CityJSONL file at `OUTPUT_DIR/output.city.jsonl` ready for downstream tiling with [tyler](https://github.com/3DBAG/tyler).

---

## Benchmark results (IGN dataset)

Tested on three of the four IGN benchmark datasets (`periurban1`, `periurban2`, `urban2`).  All three methods achieve **100% reconstruction success** (LoD2.2 BuildingPart generated for every input footprint) via the flat-roof fallback in `lod2-common::pipeline`.

| Metric | plane-extrude | arrangement | graph-cut |
|--------|:---:|:---:|:---:|
| Reconstruction success | 100% | 100% | 100% |
| Speed (periurban2, 546 bldgs) | 46.7 s | 45.2 s | 45.7 s |
| Peak memory (periurban2) | 1112 MB | 1087 MB | 1074 MB |
| Total vertices (periurban2) | 135,960 | 13,741 | 13,800 |
| CityJSONL size (periurban2) | 4.7 MB | 0.7 MB | 0.7 MB |

`arrangement` and `graph-cut` produce **~10× more compact output** than `plane-extrude` because the CDT-based approach partitions the footprint into non-overlapping cells rather than extruding independent convex hulls.

`plane-extrude` exhibits systematic geometry inflation (projected roof area ~10× larger than other methods) due to the convex-hull step filling in concavities; see the full analysis in the comparison document.
