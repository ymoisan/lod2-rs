#!/usr/bin/env python3
"""
Produce a multi-layer GeoPackage showing the hybrid reconstruction decisions.

Layers:
  graphcut_wins     – footprints where graph-cut was the chosen method
  arrangement_wins  – footprints where arrangement was chosen
  extrude_wins      – footprints where plane-extrude was chosen
  flat_fallback     – footprints that fell back to flat roof (no scorable mesh)

Each footprint carries the full decision audit as attributes.
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


def parse_cityjsonl(path: Path) -> dict[str, dict]:
    """Extract per-building attributes keyed by feature_id."""
    records = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") != "CityJSONFeature":
                continue
            for co_id, co in obj.get("CityObjects", {}).items():
                attrs = co.get("attributes", {})
                fid = attrs.get("feature_id")
                if fid is None:
                    continue
                records[fid] = {
                    "roof_reason": attrs.get("roof_reason", ""),
                    "method_used": attrs.get("method_used", ""),
                    "gc_status": attrs.get("gc_status", ""),
                    "ar_status": attrs.get("ar_status", ""),
                    "pe_status": attrs.get("pe_status", ""),
                    "gc_score": attrs.get("gc_score"),
                    "ar_score": attrs.get("ar_score"),
                    "pe_score": attrs.get("pe_score"),
                    "n_planes": attrs.get("n_planes"),
                    "n_roof_pts": attrs.get("n_roof_pts"),
                    "decision": attrs.get("decision", ""),
                }
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--footprints", required=True, help="Input footprints GeoPackage")
    parser.add_argument("--cityjsonl", required=True, help="Hybrid CityJSONL output")
    parser.add_argument("--output", required=True, help="Output GeoPackage path")
    args = parser.parse_args()

    fp_path = Path(args.footprints)
    cj_path = Path(args.cityjsonl)
    out_path = Path(args.output)

    print(f"Reading footprints from {fp_path}")
    gdf = gpd.read_file(fp_path)
    print(f"  {len(gdf)} footprints, CRS={gdf.crs}")

    print(f"Parsing CityJSONL from {cj_path}")
    attrs = parse_cityjsonl(cj_path)
    print(f"  {len(attrs)} building records")

    attr_df = pd.DataFrame.from_dict(attrs, orient="index")
    attr_df.index.name = "feature_id"
    attr_df.reset_index(inplace=True)

    merged = gdf.merge(attr_df, on="feature_id", how="left")
    merged["method_used"] = merged["method_used"].fillna("")
    merged["roof_reason"] = merged["roof_reason"].fillna("")
    merged["decision"] = merged["decision"].fillna("")

    # Early exits that never entered scoring: mark them clearly
    mask_early = merged["gc_status"] == ""
    merged.loc[mask_early, "decision"] = merged.loc[mask_early, "roof_reason"].apply(
        lambda r: f"early_exit: {r}" if r else "early_exit: unknown"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    layer_map = {
        "graphcut_wins": merged[merged["method_used"] == "graph-cut"],
        "arrangement_wins": merged[merged["method_used"] == "arrangement"],
        "extrude_wins": merged[merged["method_used"] == "plane-extrude"],
        "flat_fallback": merged[~merged["method_used"].isin(["graph-cut", "arrangement", "plane-extrude"])],
    }

    for layer_name, layer_gdf in layer_map.items():
        if layer_gdf.empty:
            print(f"  Layer '{layer_name}': 0 features (skipped)")
            continue
        layer_gdf.to_file(str(out_path), layer=layer_name, driver="GPKG")
        print(f"  Layer '{layer_name}': {len(layer_gdf)} features")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
