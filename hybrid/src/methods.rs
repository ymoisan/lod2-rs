use lod2_common::hints::{BuildingHint, RoofShape};
use lod2_common::mesh::{Face, Mesh, SemanticSurface};
use lod2_common::plane::{clip_line_to_bbox, compute_plane_localities, intersect_planes, intersection_z_at, ridge_has_inlier_support, split_segments_at_intersections, Plane, PlaneLocality};
use lod2_common::polygon::Footprint;
use lod2_common::graph_cut::{GcFace, GcNeighborPair, alpha_expansion, build_plane_inlier_sets};
use lod2_common::arrangement::Arrangement;
use nalgebra::Point3;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Shared bearing helpers (used by graph-cut and arrangement)
// ---------------------------------------------------------------------------

fn segment_bearing(p1: &[f64; 2], p2: &[f64; 2]) -> f64 {
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    let mut b = dy.atan2(dx).to_degrees();
    if b < 0.0 {
        b += 360.0;
    }
    b
}

fn bearing_within(a: f64, b: f64, tolerance: f64) -> bool {
    let mut diff = (a - b).abs() % 360.0;
    if diff > 180.0 {
        diff = 360.0 - diff;
    }
    diff <= tolerance || (180.0 - diff).abs() <= tolerance
}

fn allowed_bearings(hint: &BuildingHint) -> Vec<f64> {
    let bearing = match hint.roof_direction {
        Some(b) => b,
        None => return vec![],
    };
    match hint.roof_shape {
        Some(RoofShape::Gabled) | Some(RoofShape::Skillion) => vec![bearing],
        Some(RoofShape::Hipped) | Some(RoofShape::Pyramidal) => {
            vec![bearing, (bearing + 90.0) % 360.0]
        }
        _ => vec![bearing, (bearing + 90.0) % 360.0],
    }
}

// ---------------------------------------------------------------------------
// Footprint boundary edge extraction (for pre-splitting)
// ---------------------------------------------------------------------------

fn footprint_boundary_edges(footprint: &Footprint) -> Vec<([f64; 2], [f64; 2])> {
    let ext = &footprint.polygon.exterior;
    let n = ext.len().saturating_sub(1); // closed ring: last == first
    let mut edges = Vec::with_capacity(n);
    for i in 0..n {
        let a = &ext.vertices[i];
        let b = &ext.vertices[(i + 1) % n];
        edges.push(([a.x, a.y], [b.x, b.y]));
    }
    // Interior rings (courtyards)
    for hole in &footprint.polygon.interiors {
        let hn = hole.len().saturating_sub(1);
        for i in 0..hn {
            let a = &hole.vertices[i];
            let b = &hole.vertices[(i + 1) % hn];
            edges.push(([a.x, a.y], [b.x, b.y]));
        }
    }
    edges
}

// ===========================================================================
// Graph-cut method
// ===========================================================================

pub fn build_graphcut_lod22(
    footprint: &Footprint,
    planes: &[Plane],
    points: &[Point3<f64>],
    h_ground: f64,
    hint: &BuildingHint,
) -> Option<Mesh> {
    if planes.is_empty() {
        return None;
    }

    let bbox = footprint.polygon.bbox_2d();
    let localities = compute_plane_localities(planes, points);
    // Pre-extract each plane's 2D inlier coordinates for ridge support testing.
    let inlier_xy: Vec<Vec<(f64, f64)>> = planes
        .iter()
        .map(|p| {
            let mut v = Vec::with_capacity(p.inliers.len());
            for &i in &p.inliers {
                v.push((points[i].x, points[i].y));
            }
            v
        })
        .collect();
    let mut ridge_segments: Vec<([f64; 2], [f64; 2])> = Vec::new();
    for i in 0..planes.len() {
        for j in (i + 1)..planes.len() {
            // Only intersect spatially adjacent planes
            let li = &localities[i];
            let lj = &localities[j];
            let dx = li.cx - lj.cx;
            let dy = li.cy - lj.cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > li.radius + lj.radius + 5.0 {
                continue;
            }
            if let Some((origin, dir)) = intersect_planes(&planes[i], &planes[j]) {
                if let Some((p1, p2)) = clip_line_to_bbox(&origin, &dir, &bbox) {
                    // Require the two planes' inlier extents to overlap
                    // along the ridge direction.  Two planes whose
                    // (x,y) footprints don't share a stretch along
                    // their intersection line cannot have a real shared
                    // roof edge there — the line is a phantom and
                    // emitting it shatters the arrangement into
                    // slivers.  Perpendicular tolerance 2 m lets the
                    // line pass slightly off-centre between two slope
                    // clusters (typical ridge geometry).
                    if ridge_has_inlier_support(
                        &p1, &p2, &inlier_xy[i], &inlier_xy[j],
                    ) {
                        ridge_segments.push((p1, p2));
                    }
                }
            }
        }
    }

    if hint.roof_direction.is_some() {
        let allowed = allowed_bearings(hint);
        ridge_segments.retain(|&(p1, p2)| {
            let seg_bearing = segment_bearing(&p1, &p2);
            allowed
                .iter()
                .any(|&b| bearing_within(seg_bearing, b, 25.0))
        });
    }

    // Pre-split ridgelines at mutual intersections and footprint boundary crossings
    let fp_edges = footprint_boundary_edges(footprint);
    let ridge_with_meta: Vec<_> = ridge_segments.iter().map(|&(p1, p2)| (p1, p2, ())).collect();
    let split = split_segments_at_intersections(&fp_edges, &ridge_with_meta, 1e-3);
    let ridge_segments: Vec<_> = split
        .into_iter()
        .map(|(p1, p2, _)| (p1, p2))
        .filter(|(p1, p2)| {
            let mx = (p1[0] + p2[0]) * 0.5;
            let my = (p1[1] + p2[1]) * 0.5;
            footprint.contains_2d(mx, my)
        })
        .collect();

    build_arrangement_roof_mesh(footprint, planes, &ridge_segments, points, h_ground)
}

// ===========================================================================
// Arrangement-based mesh construction (replaces CDT approach)
// ===========================================================================

fn build_arrangement_roof_mesh(
    footprint: &Footprint,
    planes: &[Plane],
    ridge_segments: &[([f64; 2], [f64; 2])],
    points: &[Point3<f64>],
    h_ground: f64,
) -> Option<Mesh> {
    let exterior = &footprint.polygon.exterior;
    // After ensure_ccw + dedup_ring, the ring has N unique vertices (no closing
    // vertex).  We need N edges (implicitly closed: i → (i+1) % N).
    let n_ext = exterior.len();
    if n_ext < 3 {
        return None;
    }

    // Collect ALL segments: footprint boundary + ridges
    let mut all_segments: Vec<([f64; 2], [f64; 2])> = Vec::new();

    // Exterior ring edges
    for i in 0..n_ext {
        let a = &exterior.vertices[i];
        let b = &exterior.vertices[(i + 1) % n_ext];
        all_segments.push(([a.x, a.y], [b.x, b.y]));
    }

    // Interior ring edges (courtyards)
    for hole in &footprint.polygon.interiors {
        let hn = hole.len();
        for i in 0..hn {
            let a = &hole.vertices[i];
            let b = &hole.vertices[(i + 1) % hn];
            all_segments.push(([a.x, a.y], [b.x, b.y]));
        }
    }

    // Ridge segments (already pre-split and filtered to inside footprint)
    all_segments.extend_from_slice(ridge_segments);

    // Phase 3 — LineRegulariser. Snap near-parallel near-coincident inputs
    // onto shared representative lines BEFORE the arrangement so that
    // segments differing by a sub-tolerance angle/offset do not spawn
    // sub-cell slivers. Tolerances chosen to be conservative: 3° / 5 cm.
    // `min_len` = 0.30 m so very short noisy ridge fragments cannot pull
    // long footprint edges off-axis.
    lod2_common::line_regularise::regularise_segments(
        &mut all_segments,
        5.0_f64.to_radians(),
        0.15,
        0.30,
    );

    // Build the 2D arrangement.
    // Snap = 5 cm: matches the Z-cluster tolerance used in build_arrangement_roof_mesh
    // and collapses sub-cell slivers from near-coincident segments (root cause of
    // the "pole forest" on heavily-fragmented buildings like 94257). At this snap
    // the Phase 3 regulariser above is a free no-op — its effect will surface once
    // we adopt a finer/exact arrangement backend.
    let (arr, endpoint_map) = Arrangement::build(&all_segments, 0.05);
    if std::env::var("BUILDEX_DEBUG").is_ok() {
        eprintln!(
            "[buildex-debug-arr] fp={} ext_verts={} n_holes={} ridge_segs={} all_segs={} arr_verts={} arr_faces={}",
            footprint.id,
            n_ext,
            footprint.polygon.interiors.len(),
            ridge_segments.len(),
            all_segments.len(),
            arr.vertices.len(),
            arr.faces.len(),
        );
    }
    if arr.faces.is_empty() {
        return None;
    }

    // Classify interior faces: centroid must be inside footprint, not in a hole.
    // For non-convex faces whose centroid may land outside, also check if any vertex
    // is inside the footprint.
    let mut interior_face_indices: Vec<u32> = Vec::new();
    for (fi, face) in arr.faces.iter().enumerate() {
        let inside = if footprint.contains_2d(face.cx, face.cy)
            && !footprint.polygon.is_in_hole_2d(face.cx, face.cy)
        {
            true
        } else {
            // Fallback: check if any face vertex is inside the footprint
            let vids = arr.face_vertex_indices(fi as u32);
            vids.iter().any(|&vi| {
                let v = &arr.vertices[vi as usize];
                footprint.contains_2d(v.x, v.y)
                    && !footprint.polygon.is_in_hole_2d(v.x, v.y)
            })
        };
        if inside {
            interior_face_indices.push(fi as u32);
        }
    }

    if interior_face_indices.is_empty() {
        return None;
    }

    if std::env::var("BUILDEX_DEBUG").is_ok() {
        let mut areas: Vec<f64> = interior_face_indices
            .iter()
            .map(|&fi| {
                let ring: Vec<(f64, f64)> = arr
                    .face_vertex_indices(fi)
                    .iter()
                    .map(|&vi| (arr.vertices[vi as usize].x, arr.vertices[vi as usize].y))
                    .collect();
                let mut a = 0.0;
                let n = ring.len();
                for i in 0..n {
                    let j = (i + 1) % n;
                    a += ring[i].0 * ring[j].1 - ring[j].0 * ring[i].1;
                }
                (a * 0.5).abs()
            })
            .collect();
        let total: f64 = areas.iter().sum();
        let n_lt_05 = areas.iter().filter(|&&a| a < 0.5).count();
        let n_lt_1  = areas.iter().filter(|&&a| a < 1.0).count();
        let n_lt_5  = areas.iter().filter(|&&a| a < 5.0).count();
        areas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if areas.is_empty() { 0.0 } else { areas[areas.len() / 2] };
        let max = areas.last().copied().unwrap_or(0.0);
        eprintln!(
            "[buildex-debug-faces] fp={} n={} total_area={:.1} median={:.3} max={:.1} (<0.5m²={}, <1m²={}, <5m²={})",
            footprint.id, areas.len(), total, median, max, n_lt_05, n_lt_1, n_lt_5,
        );
    }

    // Compute fallback height from point cloud
    let fallback_z = {
        let mut zs: Vec<f64> = points.iter().map(|p| p.z).collect();
        if zs.is_empty() { h_ground + 3.0 } else {
            zs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            zs[zs.len() * 7 / 10]
        }
    };

    let localities = compute_plane_localities(planes, points);

    // Build per-face point lists for plane-residual data cost (Phase 4a).
    // Each LiDAR point goes to at most one interior face (faces tile the
    // footprint).  Brute force: O(F·N) with bbox prefilter — fast enough
    // for typical buildings (<= ~10k points × ~100 faces).
    let face_rings: Vec<(u32, Vec<(f64, f64)>, [f64; 4])> = interior_face_indices
        .iter()
        .map(|&fi| {
            let ring: Vec<(f64, f64)> = arr
                .face_vertex_indices(fi)
                .iter()
                .map(|&vi| (arr.vertices[vi as usize].x, arr.vertices[vi as usize].y))
                .collect();
            let (mut xmin, mut ymin) = (f64::INFINITY, f64::INFINITY);
            let (mut xmax, mut ymax) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            for &(x, y) in &ring {
                if x < xmin { xmin = x; }
                if x > xmax { xmax = x; }
                if y < ymin { ymin = y; }
                if y > ymax { ymax = y; }
            }
            (fi, ring, [xmin, ymin, xmax, ymax])
        })
        .collect();
    let mut face_points: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, p) in points.iter().enumerate() {
        for (fi, ring, bb) in &face_rings {
            if p.x < bb[0] || p.x > bb[2] || p.y < bb[1] || p.y > bb[3] {
                continue;
            }
            if point_in_ring(ring, p.x, p.y) {
                face_points.entry(*fi).or_default().push(i);
                break;
            }
        }
    }

    // Build GcFace for each interior face
    let mut gc_faces: Vec<GcFace> = Vec::new();
    let mut initial_labels: Vec<usize> = Vec::new();
    let mut face_to_gc: HashMap<u32, usize> = HashMap::new();

    for &fi in &interior_face_indices {
        let face = &arr.faces[fi as usize];
        let local_z = local_height_at(points, face.cx, face.cy, fallback_z);
        let greedy_label = greedy_best_plane(face.cx, face.cy, local_z, planes, &localities);

        let gc_idx = gc_faces.len();
        face_to_gc.insert(fi, gc_idx);
        let pts = face_points.remove(&fi).unwrap_or_default();
        gc_faces.push(GcFace {
            idx: gc_idx,
            cx: face.cx,
            cy: face.cy,
            local_z,
            pts,
        });
        initial_labels.push(greedy_label);
    }

    if gc_faces.is_empty() {
        return None;
    }

    // Build neighbor pairs from arrangement topology
    let mut gc_neighbors: Vec<GcNeighborPair> = Vec::new();
    let mut seen_pairs: HashSet<(usize, usize)> = HashSet::new();

    for &fi in &interior_face_indices {
        let gc_a = face_to_gc[&fi];
        for nfi in arr.face_neighbors(fi) {
            if let Some(&gc_b) = face_to_gc.get(&nfi) {
                let pair = if gc_a < gc_b { (gc_a, gc_b) } else { (gc_b, gc_a) };
                if seen_pairs.insert(pair) {
                    let edge_length = arr.shared_edge_length(fi, nfi);
                    gc_neighbors.push(GcNeighborPair {
                        face_a: gc_a,
                        face_b: gc_b,
                        edge_length,
                    });
                }
            }
        }
    }

    // Run alpha-expansion graph-cut.  Lambda 0.1 so the inlier-coverage
    // data cost (Phase 7) dominates over edge-length smoothness.  The
    // coverage penalty acts as a hard exclusion that prevents the
    // "compromise plane" failure mode — a plane with zero inliers in a
    // face pays an 8 m² surcharge, dwarfing any per-point residual.
    let plane_inlier_sets = build_plane_inlier_sets(planes);
    let labels = if gc_faces.len() >= 3 && gc_faces.len() <= 2000 && planes.len() >= 2 {
        alpha_expansion(&gc_faces, &gc_neighbors, planes, &plane_inlier_sets, points, &initial_labels, 0.1)
    } else {
        initial_labels.clone()
    };

    if std::env::var("BUILDEX_DEBUG").is_ok() {
        let mut label_counts: HashMap<usize, usize> = HashMap::new();
        for &l in &labels {
            *label_counts.entry(l).or_default() += 1;
        }
        eprintln!(
            "[buildex-debug] fp={} planes={} interior_faces={} labels={:?}",
            footprint.id, planes.len(), gc_faces.len(), label_counts,
        );
        for (i, p) in planes.iter().enumerate() {
            if p.inliers.is_empty() {
                continue;
            }
            let zs: Vec<f64> = p.inliers.iter().map(|&j| points[j].z).collect();
            let zmin = zs.iter().cloned().fold(f64::INFINITY, f64::min);
            let zmax = zs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let zmean = zs.iter().sum::<f64>() / zs.len() as f64;
            let used = label_counts.get(&i).copied().unwrap_or(0);
            eprintln!(
                "[buildex-debug-plane] fp={} plane={} n={} z[min/mean/max]={:.2}/{:.2}/{:.2} slope={:.1}° faces={}",
                footprint.id, i, p.inliers.len(), zmin, zmean, zmax,
                p.slope_degrees(), used,
            );
        }
    }

    // Build face_labels map: interior face index → plane label
    let mut face_labels: HashMap<u32, usize> = HashMap::new();
    for (gc_idx, &fi) in interior_face_indices.iter().enumerate() {
        face_labels.insert(fi, labels[gc_idx]);
    }
    let interior_set: HashSet<u32> = interior_face_indices.iter().copied().collect();

    // Compute z ceiling for clamping (avoids spike vertices when an
    // arrangement vertex lies far outside the plane's confidence region).
    let global_z_ceil = {
        let mut zs: Vec<f64> = points.iter().map(|p| p.z).collect();
        if zs.is_empty() {
            h_ground + 10.0
        } else {
            zs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            zs[((zs.len() as f64 * 0.95) as usize).min(zs.len() - 1)]
        }
    };
    let ceiling = global_z_ceil + 1.5;

    // Step B' (2026-05-25): tight per-plane Z clamp re-introduced
    // WITHOUT slack.  A plane is defined by its inliers; evaluating
    // eval_z at an arrangement vertex that lies outside the inlier
    // (xy) hull is extrapolation, which can rocket arbitrarily high
    // for steep planes and produces visible "sprouts" on the roof.
    // Clamping to [zmin_inlier, zmax_inlier] is principled because the
    // inlier-coverage data cost (Step A) now guarantees that the plane
    // really IS the right surface for this face — we only need to
    // suppress extrapolation noise OUTSIDE the inlier support.  No
    // slack so legitimate ridges are not artificially flattened.
    let plane_z_floor: Vec<f64> = planes
        .iter()
        .map(|p| {
            if p.inliers.is_empty() {
                h_ground
            } else {
                p.inliers
                    .iter()
                    .map(|&i| points[i].z)
                    .fold(f64::INFINITY, f64::min)
                    .max(h_ground)
            }
        })
        .collect();
    let plane_z_ceil: Vec<f64> = planes
        .iter()
        .map(|p| {
            if p.inliers.is_empty() {
                ceiling
            } else {
                p.inliers
                    .iter()
                    .map(|&i| points[i].z)
                    .fold(f64::NEG_INFINITY, f64::max)
                    .min(ceiling)
            }
        })
        .collect();

    // Build the mesh: per-(face, vertex) columns. Adjacent faces with
    // different labels (or different heights at shared vertices) get
    // independent top vertices, enabling interior step-walls between them.
    let mut mesh = Mesh::new();
    let ground_idx = mesh.add_semantic(SemanticSurface::ground());
    let wall_idx = mesh.add_semantic(SemanticSurface::wall(true));
    let mut plane_to_semantic: HashMap<usize, usize> = HashMap::new();

    // Per-(face, arr-vertex) → mesh vertex.  Within a single arrangement
    // vertex, faces whose evaluated plane Z agree (within `snap_z`) share
    // the same mesh vertex.  This collapses same-label and near-coplanar
    // adjacencies onto one column vertex, leaving real Z-steps as the
    // only sources of interior step walls.
    let snap_z = 0.05_f64; // 5 cm

    // Phase 5 (ArrangementDissolver) — union-find over interior faces:
    // adjacent faces sharing the same plane label are merged into one
    // cluster.  We then key columns and emit roof polygons by cluster id,
    // which (a) drops every interior edge between same-label faces (no
    // wall, no T-junction), and (b) reduces the per-face mesh count from
    // "one per CDT triangle" to "one per merged region" once polygon
    // triangulation runs on the cluster boundary (below).
    let mut uf_parent: HashMap<u32, u32> =
        interior_face_indices.iter().map(|&fi| (fi, fi)).collect();
    fn uf_find(parent: &mut HashMap<u32, u32>, mut x: u32) -> u32 {
        loop {
            let p = parent[&x];
            if p == x {
                return x;
            }
            let g = parent[&p];
            parent.insert(x, g);
            x = g;
        }
    }
    for he in &arr.half_edges {
        if he.origin == u32::MAX || he.face == u32::MAX {
            continue;
        }
        let twin_face = arr.half_edges[he.twin as usize].face;
        if twin_face == u32::MAX {
            continue;
        }
        if !interior_set.contains(&he.face) || !interior_set.contains(&twin_face) {
            continue;
        }
        if he.face >= twin_face {
            continue;
        }
        if face_labels[&he.face] != face_labels[&twin_face] {
            continue;
        }
        let ra = uf_find(&mut uf_parent, he.face);
        let rb = uf_find(&mut uf_parent, twin_face);
        if ra != rb {
            uf_parent.insert(ra, rb);
        }
    }
    let mut face_to_cluster: HashMap<u32, u32> = HashMap::with_capacity(uf_parent.len());
    for &fi in &interior_face_indices {
        face_to_cluster.insert(fi, uf_find(&mut uf_parent, fi));
    }
    // Cluster label is just any member's label (all equal by construction).
    let mut cluster_label: HashMap<u32, usize> = HashMap::new();
    for &fi in &interior_face_indices {
        let c = face_to_cluster[&fi];
        cluster_label.entry(c).or_insert(face_labels[&fi]);
    }

    // ---- Absorb zero-evidence clusters ----
    // A cluster whose member arrangement faces contain ZERO LiDAR points
    // got its plane label from extrapolated centroid-residual noise,
    // not from real evidence.  Such clusters appear as visible perimeter
    // slivers (sub-0.5 m² faces with a label that disagrees with every
    // neighbour).  Find each zero-evidence cluster's neighbour with the
    // longest shared boundary AND non-zero evidence; rewrite all its
    // member faces' labels to that neighbour's label.  Then re-run the
    // union-find so the dissolver merges them naturally.
    //
    // Iterate to a fixed point: after one absorption pass, new
    // zero-evidence clusters may appear if a cluster was surrounded
    // only by other zero-evidence clusters (rare but possible).
    loop {
        // Per-cluster LiDAR-point evidence count.
        let mut cluster_pts: HashMap<u32, usize> = HashMap::new();
        for &fi in &interior_face_indices {
            let c = face_to_cluster[&fi];
            let gc_idx = face_to_gc[&fi];
            *cluster_pts.entry(c).or_insert(0) += gc_faces[gc_idx].pts.len();
        }
        // Find zero-evidence clusters.
        let mut zero_clusters: Vec<u32> = Vec::with_capacity(cluster_pts.len());
        for (&c, &n) in &cluster_pts {
            if n == 0 {
                zero_clusters.push(c);
            }
        }
        if zero_clusters.is_empty() {
            break;
        }
        // For each (cluster_a, cluster_b) pair with shared half-edges,
        // accumulate boundary length.  Only need entries where one
        // cluster is zero-evidence; key by zero cluster.
        let zero_set: HashSet<u32> = zero_clusters.iter().copied().collect();
        let mut neighbour_len: HashMap<(u32, u32), f64> = HashMap::new(); // (zero, other) -> length
        for he in &arr.half_edges {
            if he.origin == u32::MAX || he.face == u32::MAX { continue; }
            let twin = &arr.half_edges[he.twin as usize];
            if twin.face == u32::MAX { continue; }
            if !interior_set.contains(&he.face) || !interior_set.contains(&twin.face) { continue; }
            let ca = face_to_cluster[&he.face];
            let cb = face_to_cluster[&twin.face];
            if ca == cb { continue; }
            // We want at least one side to be zero-evidence.
            let (zero_c, other_c) = if zero_set.contains(&ca) && !zero_set.contains(&cb) {
                (ca, cb)
            } else if zero_set.contains(&cb) && !zero_set.contains(&ca) {
                (cb, ca)
            } else {
                continue; // both zero, or neither zero — handled in later pass
            };
            // 2D length of this half-edge.
            let v0 = &arr.vertices[he.origin as usize];
            let v1 = &arr.vertices[arr.half_edges[he.next as usize].origin as usize];
            let dx = v1.x - v0.x;
            let dy = v1.y - v0.y;
            let len = (dx * dx + dy * dy).sqrt();
            *neighbour_len.entry((zero_c, other_c)).or_insert(0.0) += len;
        }
        if neighbour_len.is_empty() {
            // No zero cluster touches a non-zero neighbour.  Avoid an
            // infinite loop; leave remaining zero clusters alone.
            break;
        }
        // For each zero cluster, pick the neighbour with longest shared boundary.
        let mut best_for: HashMap<u32, (u32, f64)> = HashMap::new();
        for (&(z, other), &len) in &neighbour_len {
            let entry = best_for.entry(z).or_insert((other, 0.0));
            if len > entry.1 {
                *entry = (other, len);
            }
        }
        // Rewrite face_labels.
        let mut changed = false;
        for &fi in &interior_face_indices {
            let c = face_to_cluster[&fi];
            if let Some(&(target_cluster, _)) = best_for.get(&c) {
                let new_label = cluster_label[&target_cluster];
                if face_labels[&fi] != new_label {
                    face_labels.insert(fi, new_label);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
        // Re-run union-find.
        uf_parent = interior_face_indices.iter().map(|&fi| (fi, fi)).collect();
        for he in &arr.half_edges {
            if he.origin == u32::MAX || he.face == u32::MAX { continue; }
            let twin_face = arr.half_edges[he.twin as usize].face;
            if twin_face == u32::MAX { continue; }
            if !interior_set.contains(&he.face) || !interior_set.contains(&twin_face) { continue; }
            if he.face >= twin_face { continue; }
            if face_labels[&he.face] != face_labels[&twin_face] { continue; }
            let ra = uf_find(&mut uf_parent, he.face);
            let rb = uf_find(&mut uf_parent, twin_face);
            if ra != rb {
                uf_parent.insert(ra, rb);
            }
        }
        face_to_cluster.clear();
        for &fi in &interior_face_indices {
            face_to_cluster.insert(fi, uf_find(&mut uf_parent, fi));
        }
        cluster_label.clear();
        for &fi in &interior_face_indices {
            let c = face_to_cluster[&fi];
            cluster_label.entry(c).or_insert(face_labels[&fi]);
        }
    }

    // Per-(cluster, arr-vertex) → mesh vertex.  Phase 6.5: at each
    // arrangement vertex, reconcile all incident clusters' plane-Zs via
    // union-find with a 25 cm merge tolerance.  Cluster columns that
    // agree (within tol) collapse to one shared mesh vertex placed at
    // their mean Z.  Only true ridges (Z disagreement > tol) emit a
    // step-wall between distinct columns.
    let mut incident_clusters: HashMap<u32, Vec<u32>> = HashMap::new();
    for &fi in &interior_face_indices {
        let c = face_to_cluster[&fi];
        for vi in arr.face_vertex_indices(fi) {
            let bucket = incident_clusters.entry(vi).or_default();
            if !bucket.contains(&c) {
                bucket.push(c);
            }
        }
    }
    let merge_tol = 0.05_f64; // 5 cm: keep existing snap; outlier clamp does the heavy lifting
    let outlier_clamp = 1.5_f64; // ±1.5 m around median of incident-cluster Zs
    let mut cluster_vert: HashMap<(u32, u32), u32> = HashMap::new();
    for (vi, cids) in &incident_clusters {
        let v = &arr.vertices[*vi as usize];
        // Step 1: each cluster's evaluated Z at this vertex.
        let mut zs: Vec<f64> = cids
            .iter()
            .map(|&c| {
                let label = cluster_label[&c];
                planes[label]
                    .eval_z(v.x, v.y)
                    .unwrap_or(h_ground + 3.0)
                    .max(plane_z_floor[label])
                    .min(plane_z_ceil[label])
            })
            .collect();
        // Step 1b: clamp far-outlier cluster Zs to the median ± outlier_clamp.
        // A single mis-fit plane evaluated at the far end of its arrangement
        // extent often sprouts a spike vertex; cap it to within ±1.5 m of
        // the local consensus to suppress visual noise.  Skip when only one
        // cluster (no consensus available).
        if zs.len() >= 2 {
            let mut sorted = zs.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = sorted[sorted.len() / 2];
            for z in zs.iter_mut() {
                *z = z.clamp(median - outlier_clamp, median + outlier_clamp);
            }
        }
        // Step 2: cluster indices into groups whose Zs lie within merge_tol
        // of any other index already in the group (greedy single-link over
        // sorted Zs).
        let mut order: Vec<usize> = (0..zs.len()).collect();
        order.sort_by(|&a, &b| zs[a].partial_cmp(&zs[b]).unwrap_or(std::cmp::Ordering::Equal));
        let mut group_of: Vec<usize> = vec![0; zs.len()];
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for &i in &order {
            let z = zs[i];
            // Try to extend last group (Zs are sorted).
            let extend = match groups.last() {
                Some(last) => {
                    let last_z_max = last
                        .iter()
                        .map(|&j| zs[j])
                        .fold(f64::NEG_INFINITY, f64::max);
                    (z - last_z_max).abs() <= merge_tol
                }
                None => false,
            };
            if extend {
                let g_idx = groups.len() - 1;
                group_of[i] = g_idx;
                groups[g_idx].push(i);
            } else {
                group_of[i] = groups.len();
                groups.push(vec![i]);
            }
        }
        // Step 3: per group, emit one mesh vertex at mean Z.
        let group_mids: Vec<u32> = groups
            .iter()
            .map(|g| {
                let z_mean = g.iter().map(|&j| zs[j]).sum::<f64>() / g.len() as f64;
                mesh.add_vertex(Point3::new(v.x, v.y, z_mean))
            })
            .collect();
        // Step 4: map (cluster, vi) → group mid.
        for (i, &c) in cids.iter().enumerate() {
            cluster_vert.insert((c, *vi), group_mids[group_of[i]]);
        }
    }

    // Build the merged-face list (one polygon per cluster, with holes).
    let merged_faces = arr.build_merged_faces(&face_to_cluster);
    let mut clusters_with_polygon: HashSet<u32> = HashSet::new();
    for mf in &merged_faces {
        // Look up the cluster id from any member (all share the same root).
        if let Some(&fi) = mf.members.first() {
            clusters_with_polygon.insert(face_to_cluster[&fi]);
        }
    }

    if std::env::var("BUILDEX_DEBUG").is_ok() {
        // Per-cluster area distribution — what actually becomes emitted
        // mesh polygons (one per cluster, after same-label union).
        use std::collections::BTreeMap;
        let mut cluster_area: BTreeMap<u32, f64> = BTreeMap::new();
        for &fi in &interior_face_indices {
            let c = face_to_cluster[&fi];
            let ring: Vec<(f64, f64)> = arr
                .face_vertex_indices(fi)
                .iter()
                .map(|&vi| (arr.vertices[vi as usize].x, arr.vertices[vi as usize].y))
                .collect();
            let mut a = 0.0;
            let n = ring.len();
            for i in 0..n {
                let j = (i + 1) % n;
                a += ring[i].0 * ring[j].1 - ring[j].0 * ring[i].1;
            }
            *cluster_area.entry(c).or_insert(0.0) += (a * 0.5).abs();
        }
        let mut areas: Vec<f64> = cluster_area.values().copied().collect();
        areas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = areas.len();
        let median = if n == 0 { 0.0 } else { areas[n / 2] };
        let max = areas.last().copied().unwrap_or(0.0);
        let n_lt_05 = areas.iter().filter(|&&a| a < 0.5).count();
        let n_lt_1 = areas.iter().filter(|&&a| a < 1.0).count();
        let n_lt_5 = areas.iter().filter(|&&a| a < 5.0).count();
        eprintln!(
            "[buildex-debug-clusters] fp={} n_clusters={} n_polygons={} median={:.3} max={:.1} (<0.5m²={}, <1m²={}, <5m²={})",
            footprint.id, n, merged_faces.len(), median, max, n_lt_05, n_lt_1, n_lt_5,
        );
    }

    // Roof faces — one triangulation per merged cluster.
    for mf in &merged_faces {
        let Some(&first_fi) = mf.members.first() else {
            continue;
        };
        let c = face_to_cluster[&first_fi];
        let label = cluster_label[&c];
        let sem_idx = *plane_to_semantic.entry(label).or_insert_with(|| {
            mesh.add_semantic(SemanticSurface::roof_with_stats(
                planes[label].slope_degrees(),
                planes[label].azimuth_degrees(),
            ))
        });

        // Triangulate the merged polygon (outer ring + holes) via earcutr;
        // emit one mesh face per triangle, indexed through the cluster's
        // column vertices.
        for tri in mf.triangulate(&arr) {
            let i0 = match cluster_vert.get(&(c, tri[0])) {
                Some(&m) => m,
                None => continue,
            };
            let i1 = match cluster_vert.get(&(c, tri[1])) {
                Some(&m) => m,
                None => continue,
            };
            let i2 = match cluster_vert.get(&(c, tri[2])) {
                Some(&m) => m,
                None => continue,
            };
            mesh.add_face(Face::new(vec![i0, i1, i2]).with_semantic(sem_idx));
        }
    }

    // Fallback per-face triangulation for any cluster that failed to
    // produce a merged-face entry (defensive: keeps geometry rather than
    // dropping it).  In practice this should be empty.
    for &fi in &interior_face_indices {
        let c = face_to_cluster[&fi];
        if clusters_with_polygon.contains(&c) {
            continue;
        }
        let label = cluster_label[&c];
        let sem_idx = *plane_to_semantic.entry(label).or_insert_with(|| {
            mesh.add_semantic(SemanticSurface::roof_with_stats(
                planes[label].slope_degrees(),
                planes[label].azimuth_degrees(),
            ))
        });
        for tri in arr.triangulate_face(fi) {
            let i0 = match cluster_vert.get(&(c, tri[0])) {
                Some(&m) => m,
                None => continue,
            };
            let i1 = match cluster_vert.get(&(c, tri[1])) {
                Some(&m) => m,
                None => continue,
            };
            let i2 = match cluster_vert.get(&(c, tri[2])) {
                Some(&m) => m,
                None => continue,
            };
            mesh.add_face(Face::new(vec![i0, i1, i2]).with_semantic(sem_idx));
        }
    }

    // Ground vertices: every arrangement vertex on a NO_FACE cycle (outer
    // boundary or any hole boundary) gets a ground-level mesh vertex.
    let mut ground_vert: HashMap<u32, u32> = HashMap::new();
    for he in &arr.half_edges {
        if he.face == u32::MAX && he.origin != u32::MAX {
            ground_vert.entry(he.origin).or_insert_with(|| {
                let v = &arr.vertices[he.origin as usize];
                mesh.add_vertex(Point3::new(v.x, v.y, h_ground))
            });
        }
    }

    // Walk NO_FACE cycles to recover the outer boundary ring and any hole rings.
    let mut visited_he: Vec<bool> = vec![false; arr.half_edges.len()];
    let mut ground_rings: Vec<(Vec<u32>, f64)> = Vec::new();
    for hi in 0..arr.half_edges.len() {
        if visited_he[hi] {
            continue;
        }
        let he = &arr.half_edges[hi];
        if he.face != u32::MAX || he.origin == u32::MAX || he.next == u32::MAX {
            visited_he[hi] = true;
            continue;
        }
        let start = hi as u32;
        let mut cur = start;
        let mut ring = Vec::new();
        let max_steps = arr.half_edges.len();
        for _ in 0..max_steps {
            if visited_he[cur as usize] {
                break;
            }
            visited_he[cur as usize] = true;
            ring.push(arr.half_edges[cur as usize].origin);
            cur = arr.half_edges[cur as usize].next;
            if cur == start || cur == u32::MAX {
                break;
            }
        }
        if ring.len() >= 3 {
            // Signed area (positive CCW from above)
            let mut a2 = 0.0;
            for k in 0..ring.len() {
                let a = &arr.vertices[ring[k] as usize];
                let b = &arr.vertices[ring[(k + 1) % ring.len()] as usize];
                a2 += a.x * b.y - b.x * a.y;
            }
            ground_rings.push((ring, a2 * 0.5));
        }
    }

    if !ground_rings.is_empty() {
        // Outer boundary cycle = most negative area (CW from above).
        // Hole boundary cycles (around courtyards) have positive area
        // (CCW from above, since they bound a NO_FACE region from inside).
        let outer_idx = ground_rings
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let outer_ring = &ground_rings[outer_idx].0;
        // Ground face normal points down, so outer ring should be CW from above
        // (= CCW from below). Our outer cycle is already CW from above — keep as is.
        let bottom_outer: Vec<u32> = outer_ring
            .iter()
            .map(|vi| ground_vert[vi])
            .collect();
        let mut ground_face = Face::new(bottom_outer).with_semantic(ground_idx);
        for (i, (ring, _)) in ground_rings.iter().enumerate() {
            if i == outer_idx {
                continue;
            }
            // Hole rings are CCW from above on a down-facing face → use as-is.
            let hole_ring: Vec<u32> = ring.iter().map(|vi| ground_vert[vi]).collect();
            ground_face.holes.push(hole_ring);
        }
        mesh.add_face(ground_face);
    }

    // Walls.  Iterate every interior half-edge.
    //   * If its twin is NOT an interior face (NO_FACE or interior of a hole),
    //     emit an exterior wall from ground up to that face's roof column.
    //   * If both sides are interior faces, only the side with the lower face
    //     index emits the wall (dedup), and only when the labels differ OR
    //     the per-face Z's at the shared endpoints differ noticeably.
    for hi in 0..arr.half_edges.len() {
        let he = arr.half_edges[hi].clone();
        if he.face == u32::MAX || he.origin == u32::MAX || he.next == u32::MAX {
            continue;
        }
        if !interior_set.contains(&he.face) {
            continue;
        }
        let vi = he.origin;
        let vj = arr.half_edges[he.next as usize].origin;
        let twin_face = arr.half_edges[he.twin as usize].face;
        let is_exterior_side = !interior_set.contains(&twin_face);

        if is_exterior_side {
            let cluster_a = face_to_cluster[&he.face];
            let top_i = match cluster_vert.get(&(cluster_a, vi)) {
                Some(&m) => m,
                None => continue,
            };
            let top_j = match cluster_vert.get(&(cluster_a, vj)) {
                Some(&m) => m,
                None => continue,
            };
            let bot_i = match ground_vert.get(&vi) {
                Some(&m) => m,
                None => {
                    let v = &arr.vertices[vi as usize];
                    let m = mesh.add_vertex(Point3::new(v.x, v.y, h_ground));
                    ground_vert.insert(vi, m);
                    m
                }
            };
            let bot_j = match ground_vert.get(&vj) {
                Some(&m) => m,
                None => {
                    let v = &arr.vertices[vj as usize];
                    let m = mesh.add_vertex(Point3::new(v.x, v.y, h_ground));
                    ground_vert.insert(vj, m);
                    m
                }
            };
            // Outward-facing CCW quad (face A's interior is on the left of i→j).
            mesh.add_face(
                Face::new(vec![bot_i, bot_j, top_j, top_i]).with_semantic(wall_idx),
            );
            continue;
        }

        // Interior-interior shared edge: dedup by face index.
        if he.face >= twin_face {
            continue;
        }
        let cluster_a = face_to_cluster[&he.face];
        let cluster_b = face_to_cluster[&twin_face];
        // Same cluster → no wall (the shared edge is interior to one merged
        // polygon, so its two half-edges meet at identical column vertices
        // and form a manifold use=2 internal edge in the roof triangulation).
        if cluster_a == cluster_b {
            continue;
        }
        let a_top_i = match cluster_vert.get(&(cluster_a, vi)) {
            Some(&m) => m,
            None => continue,
        };
        let a_top_j = match cluster_vert.get(&(cluster_a, vj)) {
            Some(&m) => m,
            None => continue,
        };
        let b_top_i = match cluster_vert.get(&(cluster_b, vi)) {
            Some(&m) => m,
            None => continue,
        };
        let b_top_j = match cluster_vert.get(&(cluster_b, vj)) {
            Some(&m) => m,
            None => continue,
        };
        // Skip wall when the two clusters' column vertices coincide at both
        // endpoints (the inter-cluster Z snap merged them into one mesh
        // vertex — no step to bridge).
        if a_top_i == b_top_i && a_top_j == b_top_j {
            continue;
        }
        // Wall quad bridging the two roof columns.  Orientation:
        // when traversing from face A's i→j (interior of A on the left),
        // outward = toward face B → CCW quad as below.
        mesh.add_face(
            Face::new(vec![a_top_i, a_top_j, b_top_j, b_top_i]).with_semantic(wall_idx),
        );
    }

    // Column caps.  At each arrangement vertex, walk outgoing half-edges in
    // CCW order; the "wedge face" between two consecutive outgoing edges is
    // the face on the LEFT of the latter (= he.face).  Each wedge contributes
    // one column vertex (its per-(face,vi) mesh vertex if interior, else the
    // ground vertex at vi).  Walking CCW yields a closed cycle of column
    // vertices; emitting that polygon as a face closes every column edge
    // (which would otherwise be a boundary).  This is roofer's
    // ArrangementExtruder column construction in spirit.
    {
        // Build outgoing list per vertex once (O(E)).
        let mut outgoing_per_vi: Vec<Vec<u32>> = vec![Vec::new(); arr.vertices.len()];
        for (hi, he) in arr.half_edges.iter().enumerate() {
            if he.origin != u32::MAX {
                outgoing_per_vi[he.origin as usize].push(hi as u32);
            }
        }
        for (vi_usize, mut outgoing) in outgoing_per_vi.into_iter().enumerate() {
            if outgoing.len() < 3 {
                continue;
            }
            let vi = vi_usize as u32;
            let v = &arr.vertices[vi_usize];
            outgoing.sort_by(|&a, &b| {
                let da = arr.half_edges[arr.half_edges[a as usize].twin as usize].origin;
                let db = arr.half_edges[arr.half_edges[b as usize].twin as usize].origin;
                let pa = &arr.vertices[da as usize];
                let pb = &arr.vertices[db as usize];
                let aa = (pa.y - v.y).atan2(pa.x - v.x);
                let ab = (pb.y - v.y).atan2(pb.x - v.x);
                aa.partial_cmp(&ab).unwrap_or(std::cmp::Ordering::Equal)
            });
            // Wedge face for outgoing he_j = he_j.face; lookup by its cluster.
            let mut column: Vec<u32> = Vec::with_capacity(outgoing.len());
            let mut have_step = false;
            for &he in &outgoing {
                let f = arr.half_edges[he as usize].face;
                let mid = if f != u32::MAX && interior_set.contains(&f) {
                    let c = face_to_cluster[&f];
                    match cluster_vert.get(&(c, vi)) {
                        Some(&m) => m,
                        None => continue,
                    }
                } else {
                    *ground_vert.entry(vi).or_insert_with(|| {
                        mesh.add_vertex(Point3::new(v.x, v.y, h_ground))
                    })
                };
                if let Some(&first) = column.first() {
                    if mid != first {
                        have_step = true;
                    }
                }
                column.push(mid);
            }
            if !have_step {
                continue;
            }
            // Dedup consecutive matching vertices (same column → no edge).
            let mut dedup: Vec<u32> = Vec::with_capacity(column.len());
            for &m in &column {
                if dedup.last() != Some(&m) {
                    dedup.push(m);
                }
            }
            while dedup.len() > 1 && dedup.first() == dedup.last() {
                dedup.pop();
            }
            if dedup.len() < 3 {
                continue;
            }
            // Split self-touching caps at repeated vertices into simple
            // sub-cycles.  If `dedup` lists the same mesh vertex more than
            // once, the polygon "pinches" at that vertex.  We can decompose
            // it deterministically with a stack walk: when we see a vertex
            // already on the stack, pop everything down to (and including)
            // that vertex, emit it as a sub-cycle, then continue.  Each
            // sub-cycle has unique vertices and contributes valid manifold
            // faces; the original "pinch" vertex remains shared between
            // sub-cycles, which is fine since walls / other caps cover it.
            let mut stack: Vec<u32> = Vec::with_capacity(dedup.len());
            let mut on_stack: std::collections::HashMap<u32, usize> =
                std::collections::HashMap::with_capacity(dedup.len());
            let mut sub_cycles: Vec<Vec<u32>> = Vec::new();
            for &m in &dedup {
                if let Some(&pos) = on_stack.get(&m) {
                    // Emit stack[pos..] as a sub-cycle.
                    let cycle: Vec<u32> = stack.drain(pos..).collect();
                    for &x in &cycle {
                        on_stack.remove(&x);
                    }
                    if cycle.len() >= 3 {
                        sub_cycles.push(cycle);
                    }
                }
                on_stack.insert(m, stack.len());
                stack.push(m);
            }
            // The residual stack is the outermost cycle.
            if stack.len() >= 3 {
                sub_cycles.push(stack);
            }
            for cycle in sub_cycles {
                // Final dedup pass: each sub-cycle must have unique vertices
                // (the stack walk guarantees this) and ≥3 vertices.
                debug_assert!(
                    {
                        let mut s = cycle.clone();
                        s.sort_unstable();
                        s.windows(2).all(|w| w[0] != w[1])
                    },
                    "sub-cycle must be simple"
                );
                mesh.add_face(Face::new(cycle));
            }
        }
    }

    // NOTE: merge_coplanar_roof_faces() can drop collinear boundary vertices
    // from merged polygons; those dropped vertices are still referenced by
    // wall quads, leaving T-junctions / boundary edges.  Skip it for now;
    // re-enable once mesh sealing (Phase 7) handles T-junction insertion.
    // mesh.merge_coplanar_roof_faces();

    // Phase 7 — normalise face winding so signed volume is positive (outward
    // normals).  Mesh sealing (seal_holes) is disabled: the current walk
    // cannot disambiguate boundary multigraphs and the fan regresses
    // manifoldness more than it closes holes.  Revisit once boundaries are
    // walked as a planar half-edge structure.
    // let _sealed = mesh.seal_holes(8);
    mesh.normalise_winding();

    Some(mesh)
}

/// Compute Z for all arrangement vertices participating in interior faces.
fn compute_arrangement_vertex_heights(
    arr: &Arrangement,
    interior_face_indices: &[u32],
    face_labels: &HashMap<u32, usize>,
    planes: &[Plane],
    points: &[Point3<f64>],
    h_ground: f64,
) -> HashMap<u32, f64> {
    let mut vertex_z: HashMap<u32, f64> = HashMap::new();

    // Compute z_95p ceiling
    let global_z_ceil = {
        let mut zs: Vec<f64> = points.iter().map(|p| p.z).collect();
        if zs.is_empty() {
            h_ground + 10.0
        } else {
            zs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            zs[((zs.len() as f64 * 0.95) as usize).min(zs.len() - 1)]
        }
    };
    let ceiling = global_z_ceil + 1.5;

    // Collect all vertices used by interior faces
    let mut interior_verts: HashSet<u32> = HashSet::new();
    for &fi in interior_face_indices {
        for vi in arr.face_vertex_indices(fi) {
            interior_verts.insert(vi);
        }
    }

    for &vi in &interior_verts {
        let v = &arr.vertices[vi as usize];

        // Collect adjacent interior face labels
        let mut label_counts: HashMap<usize, usize> = HashMap::new();

        // Walk outgoing half-edges from this vertex to find adjacent faces
        for (_hi, he) in arr.half_edges.iter().enumerate() {
            if he.origin == vi && he.face != u32::MAX {
                if let Some(&label) = face_labels.get(&he.face) {
                    *label_counts.entry(label).or_insert(0) += 1;
                }
            }
        }

        let z = if label_counts.is_empty() {
            h_ground + 3.0
        } else if label_counts.len() == 1 {
            let &label = label_counts.keys().next().unwrap();
            planes[label].eval_z(v.x, v.y).unwrap_or(h_ground + 3.0)
        } else if label_counts.len() == 2 {
            let labels: Vec<usize> = label_counts.keys().copied().collect();
            let (la, lb) = (labels[0], labels[1]);

            let both_horizontal = planes[la].normal.z.abs() > 0.996
                && planes[lb].normal.z.abs() > 0.996;
            if both_horizontal {
                let winner = if label_counts[&la] >= label_counts[&lb] { la } else { lb };
                planes[winner].eval_z(v.x, v.y).unwrap_or(h_ground + 3.0)
            } else {
                let weighted_avg = || -> f64 {
                    let ca = label_counts[&la] as f64;
                    let cb = label_counts[&lb] as f64;
                    let za = planes[la].eval_z(v.x, v.y).unwrap_or(h_ground + 3.0);
                    let zb = planes[lb].eval_z(v.x, v.y).unwrap_or(h_ground + 3.0);
                    (za * ca + zb * cb) / (ca + cb)
                };
                if let Some((origin, dir)) = intersect_planes(&planes[la], &planes[lb]) {
                    match intersection_z_at(&origin, &dir, v.x, v.y) {
                        Some(z_int) => {
                            let za = planes[la].eval_z(v.x, v.y).unwrap_or(h_ground + 3.0);
                            let zb = planes[lb].eval_z(v.x, v.y).unwrap_or(h_ground + 3.0);
                            let z_mid = (za + zb) * 0.5;
                            if (z_int - z_mid).abs() > 5.0 {
                                weighted_avg()
                            } else {
                                z_int
                            }
                        }
                        None => weighted_avg(),
                    }
                } else {
                    weighted_avg()
                }
            }
        } else {
            // 3+ labels: weighted average
            let mut z_sum = 0.0;
            let mut w_sum = 0.0;
            for (&label, &count) in &label_counts {
                let w = count as f64;
                let z = planes[label].eval_z(v.x, v.y).unwrap_or(h_ground + 3.0);
                z_sum += z * w;
                w_sum += w;
            }
            z_sum / w_sum
        };

        let z_clamped = z.max(h_ground).min(ceiling);
        vertex_z.insert(vi, z_clamped);
    }

    vertex_z
}

/// Ray-cast point-in-polygon test for a 2D ring (CCW or CW).
fn point_in_ring(ring: &[(f64, f64)], x: f64, y: f64) -> bool {
    if ring.len() < 3 {
        return false;
    }
    let mut inside = false;
    let n = ring.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = ring[i];
        let (xj, yj) = ring[j];
        if (yi > y) != (yj > y) {
            let denom = yj - yi;
            if denom.abs() > 0.0 {
                let x_intersect = xi + (y - yi) * (xj - xi) / denom;
                if x < x_intersect {
                    inside = !inside;
                }
            }
        }
        j = i;
    }
    inside
}

fn local_height_at(points: &[Point3<f64>], x: f64, y: f64, fallback: f64) -> f64 {
    let radius_sq = 4.0; // 2m radius
    let mut nearby_z: Vec<f64> = points
        .iter()
        .filter(|p| {
            let dx = p.x - x;
            let dy = p.y - y;
            dx * dx + dy * dy < radius_sq
        })
        .map(|p| p.z)
        .collect();
    if nearby_z.is_empty() {
        return fallback;
    }
    nearby_z.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    nearby_z[nearby_z.len() * 7 / 10]
}

/// Find greedy best plane for a face based on height residual and 2D proximity.
fn greedy_best_plane(cx: f64, cy: f64, local_z: f64, planes: &[Plane], localities: &[PlaneLocality]) -> usize {
    let proximity_weight = 0.5;
    let mut best_idx = 0;
    let mut best_score = f64::MAX;
    for (i, plane) in planes.iter().enumerate() {
        if let Some(z) = plane.eval_z(cx, cy) {
            let z_residual = (z - local_z).abs();
            let loc = &localities[i];
            let dx = cx - loc.cx;
            let dy = cy - loc.cy;
            let xy_dist = (dx * dx + dy * dy).sqrt();
            let overshoot = (xy_dist - loc.radius).max(0.0);
            let score = z_residual + proximity_weight * overshoot;
            if score < best_score {
                best_score = score;
                best_idx = i;
            }
        }
    }
    best_idx
}

