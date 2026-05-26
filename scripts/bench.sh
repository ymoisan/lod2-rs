#!/usr/bin/env bash
# bench.sh — run buildex (lod2-rs/hybrid) and roofer (CGAL C++) on a fixed
# bench set of Halifax fids and produce per-engine CityJSONSeq output.
#
# Usage: bench.sh [buildex|roofer|both]    (default: both)
#
# Inputs and binaries are fixed by env vars below; override on the command line:
#   BUILDEX=/path/to/hybrid ROOFER=/path/to/roofer FOOTPRINTS=... POINTCLOUD=... BENCH_DIR=...

set -euo pipefail

: "${BUILDEX:=/mnt/d/github/lod2-rs/buildex}"
: "${ROOFER:=/home/yvm001/builds/roofer/apps/roofer-app/roofer}"
: "${FOOTPRINTS:=/mnt/d/halifax-county/AutoBuilding.gpkg}"
: "${POINTCLOUD:=/mnt/d/halifax-county/CG11074_2018_PointCloud_CGVD2013.laz}"
: "${BENCH_DIR:=/mnt/d/halifax-county/bench}"
: "${FIDS:=3579,10201,4091,21522,27046,106110,94257,98818,134780,50314}"

mode="${1:-both}"

mkdir -p "$BENCH_DIR/buildex" "$BENCH_DIR/roofer"

run_buildex() {
  echo "[bench] buildex on fids=$FIDS"
  rm -f "$BENCH_DIR/buildex"/*.city.jsonl "$BENCH_DIR/buildex"/*.report.json
  "$BUILDEX" \
    --footprints "$FOOTPRINTS" \
    --pointcloud "$POINTCLOUD" \
    --output    "$BENCH_DIR/buildex" \
    --fids      "$FIDS" 2>&1 | tee "$BENCH_DIR/buildex/run.log"
  ls -la "$BENCH_DIR/buildex"
}

run_roofer() {
  local fids_sql
  fids_sql="$(echo "$FIDS" | tr ',' ',')"
  echo "[bench] roofer on fids=$FIDS  (filter: fid IN ($fids_sql))"
  rm -f "$BENCH_DIR/roofer"/*.city.jsonl
  "$ROOFER" \
    --id-attribute fid \
    --filter "fid IN ($fids_sql)" \
    -j 4 \
    "$POINTCLOUD" "$FOOTPRINTS" "$BENCH_DIR/roofer" 2>&1 | tee "$BENCH_DIR/roofer/run.log"
  ls -la "$BENCH_DIR/roofer"
}

case "$mode" in
  buildex) run_buildex ;;
  roofer)  run_roofer  ;;
  both)    run_buildex; run_roofer ;;
  *) echo "usage: $0 [buildex|roofer|both]"; exit 2 ;;
esac

echo "[bench] done. outputs in $BENCH_DIR/{buildex,roofer}/"
