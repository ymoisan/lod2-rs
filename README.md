# lod2-rs

This repo hosts [LoD2](https://osmbuildings.org/blog/2018-02-28_level_of_detail/) 3D buildings reconstruction methods from aerial lidar with a deliberate intent to use Rust.  Pretty much all code here will be written by AI, following careful planning and continuous "human steering" (whence the choice of the [unlicense](https://unlicense.org/)).  As a short term goal, these methods will be implemented : 

- plane-extrude (TerraScan-style approach) <br>
- arrangement (City3D-style; 3D plane arrangement) <br>
- graph-cut (Roofer-style; graph-cut optimization) <br>

For details see [A new benchmark on LoD 2 building reconstruction from aerial lidar and footprints](https://isprs-archives.copernicus.org/articles/XLVIII-1-W6-2025/83/2025/isprs-archives-XLVIII-1-W6-2025-83-2025.pdf).
