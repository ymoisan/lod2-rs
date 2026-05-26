#!/usr/bin/env python3
"""Per-feature mesh sanity report for CityJSONSeq output.

For each LoD (typically 2.2) MultiSolid in each feature, computes:
  V, F, E, Euler = V - E + F
  manifold       = every undirected edge is used by exactly 2 oriented half-edges
  closed         = no boundary edges (use-count == 1)
  signed_volume  = sum of tetra-volumes from origin (sign matters for orientation)
  n_boundary_edges, n_nonmanifold_edges
  bbox

Usage:
  cityjsonl_sanity.py <one_or_more.city.jsonl> [--lod 2.2] [--id-attr fid]

Emits a JSON array (one object per (feature, lod)) and a short table to stderr.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path


def decode_vertices(transform, raw_verts):
    sx, sy, sz = transform["scale"]
    tx, ty, tz = transform["translate"]
    return [(v[0] * sx + tx, v[1] * sy + ty, v[2] * sz + tz) for v in raw_verts]


def _depth_to_int(node):
    d = 0
    while isinstance(node, list) and node:
        node = node[0]
        d += 1
    return d


def iter_solid_polygons(boundaries, lod_name):
    """Yield outer-ring vertex-index lists, depth-agnostic.

    Standard CityJSON depths (root list always counted):
      MultiSurface : 3  [poly[ring[int]]]
      Solid        : 4  [shell[poly[ring[int]]]]
      MultiSolid   : 5  [solid[shell[poly[ring[int]]]]]
    Some writers emit Solid as a single-shell MultiSurface (depth 3).
    We descend to depth 3 and yield outer rings (poly[0]).
    """
    d = _depth_to_int(boundaries)
    # Normalise to a list of polys (depth 3 from poly downward).
    if d <= 0:
        return
    if d == 3:
        polys = boundaries
        for p in polys:
            if p:
                yield p[0]
    elif d == 4:
        for shell in boundaries:
            for p in shell:
                if p:
                    yield p[0]
    elif d == 5:
        for solid in boundaries:
            for shell in solid:
                for p in shell:
                    if p:
                        yield p[0]


def signed_volume(tris, verts):
    vol = 0.0
    for a, b, c in tris:
        ax, ay, az = verts[a]
        bx, by, bz = verts[b]
        cx, cy, cz = verts[c]
        vol += (ax * (by * cz - bz * cy)
                - ay * (bx * cz - bz * cx)
                + az * (bx * cy - by * cx))
    return vol / 6.0


def fan_triangulate(ring):
    if len(ring) < 3:
        return []
    a = ring[0]
    return [(a, ring[i], ring[i + 1]) for i in range(1, len(ring) - 1)]


def analyse_feature(feat, transform, lod_name, id_attr):
    cobjs = feat.get("CityObjects", {})
    verts = decode_vertices(transform, feat.get("vertices", []))
    out = []
    for obj_id, obj in cobjs.items():
        for geom in obj.get("geometry", []) or []:
            lod = str(geom.get("lod", ""))
            if lod_name and lod != lod_name:
                continue
            boundaries = geom.get("boundaries", [])
            faces = []  # list of outer rings
            for ring in iter_solid_polygons(boundaries, lod_name):
                faces.append(ring)
            if not faces:
                continue
            # build edge-use map (oriented)
            edge_uses = Counter()
            tris = []
            vertex_set = set()
            for ring in faces:
                vertex_set.update(ring)
                n = len(ring)
                for i in range(n):
                    a, b = ring[i], ring[(i + 1) % n]
                    edge_uses[(min(a, b), max(a, b))] += 1
                tris.extend(fan_triangulate(ring))
            E = len(edge_uses)
            V = len(vertex_set)
            F = len(faces)
            euler = V - E + F
            n_boundary = sum(1 for u in edge_uses.values() if u == 1)
            n_nonmanifold = sum(1 for u in edge_uses.values() if u > 2)
            n_good = sum(1 for u in edge_uses.values() if u == 2)
            closed = (n_boundary == 0)
            manifold = closed and (n_nonmanifold == 0)
            vol = signed_volume(tris, verts) if tris else 0.0
            xs = [verts[i][0] for i in vertex_set]
            ys = [verts[i][1] for i in vertex_set]
            zs = [verts[i][2] for i in vertex_set]
            attrs = obj.get("attributes", {}) or {}
            out.append({
                "id": attrs.get(id_attr) or obj_id,
                "lod": lod,
                "V": V, "F": F, "E": E, "euler": euler,
                "edge_use_2": n_good,
                "edge_use_1": n_boundary,
                "edge_use_ge3": n_nonmanifold,
                "closed": closed,
                "manifold": manifold,
                "signed_volume": vol,
                "bbox_z": [min(zs), max(zs)] if zs else [0, 0],
                "bbox_xy": [min(xs), min(ys), max(xs), max(ys)] if xs else None,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--lod", default="2.2")
    ap.add_argument("--id-attr", default="fid")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write JSON array to this file (default: stdout)")
    args = ap.parse_args()

    all_reports = []
    for p in args.inputs:
        transform = None
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("type") == "CityJSON":
                    transform = obj.get("transform", {"scale": [1, 1, 1], "translate": [0, 0, 0]})
                    continue
                if obj.get("type") != "CityJSONFeature":
                    continue
                if transform is None:
                    transform = obj.get("transform", {"scale": [1, 1, 1], "translate": [0, 0, 0]})
                reps = analyse_feature(obj, transform, args.lod, args.id_attr)
                for r in reps:
                    r["source"] = str(p)
                    all_reports.append(r)

    out_text = json.dumps(all_reports, indent=2)
    if args.out:
        args.out.write_text(out_text)
    else:
        print(out_text)

    # Short stderr table
    print(f"\n=== mesh sanity (lod={args.lod}) ===", file=sys.stderr)
    print(f"{'source':<20} {'id':<10} {'V':>5} {'F':>4} {'E':>5} "
          f"{'eu':>4} {'b':>3} {'nm':>3} {'closed':>6} {'mfd':>4} {'vol':>10}",
          file=sys.stderr)
    n_closed = n_manifold = n_total = 0
    for r in all_reports:
        n_total += 1
        n_closed += int(r["closed"])
        n_manifold += int(r["manifold"])
        print(f"{Path(r['source']).stem[:20]:<20} {str(r['id'])[:10]:<10} "
              f"{r['V']:>5} {r['F']:>4} {r['E']:>5} {r['euler']:>4} "
              f"{r['edge_use_1']:>3} {r['edge_use_ge3']:>3} "
              f"{str(r['closed'])[:5]:>6} {str(r['manifold'])[:4]:>4} "
              f"{r['signed_volume']:>10.1f}",
              file=sys.stderr)
    print(f"--- totals: closed {n_closed}/{n_total}, manifold {n_manifold}/{n_total}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
