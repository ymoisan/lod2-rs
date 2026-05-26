//! 2D Line Arrangement — builds a DCEL (half-edge data structure) from a set
//! of 2D line segments. Used for building watertight roof meshes.
//!
//! Algorithm:
//! 1. Split all segments at mutual intersections → non-crossing sub-segments
//! 2. Snap endpoints to a canonical vertex pool (tolerance-based dedup)
//! 3. Build half-edge pairs, sort outgoing edges per vertex by angle
//! 4. Link next/prev pointers via radial ordering
//! 5. Prune dangling edges (degree-1 vertices)
//! 6. Extract face cycles and compute signed areas

use crate::plane::segment_intersection_2d;

const NO_EDGE: u32 = u32::MAX;
const NO_FACE: u32 = u32::MAX;

// ───────────────────── Data structures ─────────────────────

#[derive(Debug, Clone)]
pub struct ArrVertex {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone)]
pub struct ArrHalfEdge {
    pub origin: u32,
    pub twin: u32,
    pub next: u32,
    pub prev: u32,
    pub face: u32, // NO_FACE for unbounded
}

#[derive(Debug, Clone)]
pub struct ArrFace {
    pub edge: u32,     // one half-edge on this face's boundary
    pub area: f64,     // signed area (positive = CCW = bounded interior)
    pub cx: f64,       // centroid x
    pub cy: f64,       // centroid y
}

#[derive(Debug, Clone)]
pub struct Arrangement {
    pub vertices: Vec<ArrVertex>,
    pub half_edges: Vec<ArrHalfEdge>,
    pub faces: Vec<ArrFace>,
}

// ───────────────────── Builder ─────────────────────

impl Arrangement {
    /// Build a 2D line arrangement from a set of line segments.
    ///
    /// `snap` is the tolerance for merging nearby vertices (typically 1e-3 meters).
    /// Returns the arrangement and an endpoint map: `endpoint_map[2*i]` is the
    /// arrangement vertex index for the start of segment `i`, `endpoint_map[2*i+1]`
    /// for the end.
    pub fn build(segments: &[([f64; 2], [f64; 2])], snap: f64) -> (Self, Vec<u32>) {
        if segments.is_empty() {
            return (Arrangement {
                vertices: Vec::new(),
                half_edges: Vec::new(),
                faces: Vec::new(),
            }, Vec::new());
        }

        // Phase 1: Split all segments at mutual intersections
        let (verts, edges, endpoint_map) = split_all_segments(segments, snap);

        if edges.is_empty() {
            return (Arrangement {
                vertices: verts.into_iter().map(|v| ArrVertex { x: v[0], y: v[1] }).collect(),
                half_edges: Vec::new(),
                faces: Vec::new(),
            }, endpoint_map);
        }

        // Phase 2: Build half-edge pairs
        let n_edges = edges.len();
        let mut half_edges: Vec<ArrHalfEdge> = Vec::with_capacity(n_edges * 2);
        // outgoing[v] = list of half-edge indices originating at vertex v
        let mut outgoing: Vec<Vec<u32>> = vec![Vec::new(); verts.len()];

        for &(u, v) in &edges {
            let h_uv = half_edges.len() as u32;
            let h_vu = h_uv + 1;
            half_edges.push(ArrHalfEdge {
                origin: u,
                twin: h_vu,
                next: NO_EDGE,
                prev: NO_EDGE,
                face: NO_FACE,
            });
            half_edges.push(ArrHalfEdge {
                origin: v,
                twin: h_uv,
                next: NO_EDGE,
                prev: NO_EDGE,
                face: NO_FACE,
            });
            outgoing[u as usize].push(h_uv);
            outgoing[v as usize].push(h_vu);
        }

        // Phase 3: Sort outgoing edges by angle and link next pointers
        for (vi, out) in outgoing.iter_mut().enumerate() {
            if out.is_empty() {
                continue;
            }
            let vx = verts[vi][0];
            let vy = verts[vi][1];

            // Sort by angle of the outgoing direction
            out.sort_by(|&a, &b| {
                let dest_a = half_edges[half_edges[a as usize].twin as usize].origin;
                let dest_b = half_edges[half_edges[b as usize].twin as usize].origin;
                let angle_a = (verts[dest_a as usize][1] - vy)
                    .atan2(verts[dest_a as usize][0] - vx);
                let angle_b = (verts[dest_b as usize][1] - vy)
                    .atan2(verts[dest_b as usize][0] - vx);
                angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Link: twin(e_i).next = e_{(i-1+k) % k}
            // When arriving at vertex vi via twin(e_i), the face to the left
            // continues with the previous outgoing edge in CCW order (= next CW).
            let k = out.len();
            for i in 0..k {
                let arriving = half_edges[out[i] as usize].twin; // half-edge arriving at vi
                let departing = out[(i + k - 1) % k];            // prev outgoing in CCW = next CW
                half_edges[arriving as usize].next = departing;
                half_edges[departing as usize].prev = arriving;
            }
        }

        // Phase 4: Prune dangling edges (degree-1 vertices)
        prune_dangling(&mut half_edges, &mut outgoing, &verts);

        // Phase 5: Extract face cycles
        let faces = extract_faces(&verts, &mut half_edges);

        (Arrangement {
            vertices: verts.into_iter().map(|v| ArrVertex { x: v[0], y: v[1] }).collect(),
            half_edges,
            faces,
        }, endpoint_map)
    }

    /// Get ordered vertex indices for a face boundary.
    pub fn face_vertex_indices(&self, face_idx: u32) -> Vec<u32> {
        let start = self.faces[face_idx as usize].edge;
        let mut result = Vec::new();
        let mut he = start;
        loop {
            result.push(self.half_edges[he as usize].origin);
            he = self.half_edges[he as usize].next;
            if he == start {
                break;
            }
            if result.len() > self.half_edges.len() {
                // Safety: prevent infinite loop on malformed DCEL
                break;
            }
        }
        result
    }

    /// Get adjacent face indices for a face (via twin edges).
    pub fn face_neighbors(&self, face_idx: u32) -> Vec<u32> {
        let vindices = self.face_vertex_indices(face_idx);
        let start = self.faces[face_idx as usize].edge;
        let mut neighbors = Vec::new();
        let mut he = start;
        for _ in 0..vindices.len() {
            let twin = self.half_edges[he as usize].twin;
            let twin_face = self.half_edges[twin as usize].face;
            if twin_face != NO_FACE && twin_face != face_idx {
                if !neighbors.contains(&twin_face) {
                    neighbors.push(twin_face);
                }
            }
            he = self.half_edges[he as usize].next;
        }
        neighbors
    }

    /// Compute shared edge length between two adjacent faces.
    pub fn shared_edge_length(&self, face_a: u32, face_b: u32) -> f64 {
        let start = self.faces[face_a as usize].edge;
        let mut total = 0.0;
        let mut he = start;
        loop {
            let twin = self.half_edges[he as usize].twin;
            if self.half_edges[twin as usize].face == face_b {
                let o = self.half_edges[he as usize].origin as usize;
                let d = self.half_edges[self.half_edges[he as usize].next as usize].origin as usize;
                let dx = self.vertices[d].x - self.vertices[o].x;
                let dy = self.vertices[d].y - self.vertices[o].y;
                total += (dx * dx + dy * dy).sqrt();
            }
            he = self.half_edges[he as usize].next;
            if he == start {
                break;
            }
        }
        total
    }

    /// Triangulate a face using ear-clipping, returning triangles as arrangement vertex index triples.
    pub fn triangulate_face(&self, face_idx: u32) -> Vec<[u32; 3]> {
        let vindices = self.face_vertex_indices(face_idx);
        let n = vindices.len();
        if n < 3 {
            return Vec::new();
        }
        if n == 3 {
            return vec![[vindices[0], vindices[1], vindices[2]]];
        }

        // Build flat coordinate array for earcutr
        let mut coords: Vec<f64> = Vec::with_capacity(n * 2);
        for &vi in &vindices {
            coords.push(self.vertices[vi as usize].x);
            coords.push(self.vertices[vi as usize].y);
        }

        let tri_indices = earcutr::earcut(&coords, &[], 2).unwrap_or_default();
        let mut triangles = Vec::with_capacity(tri_indices.len() / 3);
        for chunk in tri_indices.chunks_exact(3) {
            triangles.push([
                vindices[chunk[0]],
                vindices[chunk[1]],
                vindices[chunk[2]],
            ]);
        }
        triangles
    }
}

// ───────────────────── Segment splitting ─────────────────────

/// Split all segments at mutual intersections and T-junctions, snap endpoints.
/// Returns (vertex_pool, edge_list) where edges reference vertex indices.
/// Snap-rounded segment arrangement.
///
/// All endpoints are quantised to a fixed grid of cell-size `snap` (typically
/// 1 mm) and indexed by integer keys.  This eliminates the "vertex chain"
/// pathology of tolerance-based linear-scan dedup, where A1 is within `snap`
/// of A0 and A2 within `snap` of A1 but A0–A2 are 2·snap apart, spawning
/// near-duplicate vertices and the slivers that follow.  Hash-based lookup
/// also drops vertex insert from O(N) to O(1) and edge dedup from O(E) to
/// O(1), which matters on heavily-fragmented buildings (94257 etc.).
fn split_all_segments(
    segments: &[([f64; 2], [f64; 2])],
    snap: f64,
) -> (Vec<[f64; 2]>, Vec<(u32, u32)>, Vec<u32>) {
    use std::collections::{HashMap, HashSet};

    let n = segments.len();
    let inv_snap = 1.0 / snap;
    let snap_sq = snap * snap;

    // Quantise a point to a (i64,i64) grid key.
    let key = |pt: [f64; 2]| -> (i64, i64) {
        (
            (pt[0] * inv_snap).round() as i64,
            (pt[1] * inv_snap).round() as i64,
        )
    };

    // Collect split parameters per segment.
    let mut split_ts: Vec<Vec<f64>> = vec![Vec::new(); n];

    // Pairwise crossings — O(n²) but unchanged for now (Phase 2 will replace
    // with a sweep-line if profiling shows this dominates).
    for i in 0..n {
        for j in (i + 1)..n {
            if let Some((_pt, ti, tj)) = segment_intersection_2d(
                &segments[i].0,
                &segments[i].1,
                &segments[j].0,
                &segments[j].1,
            ) {
                split_ts[i].push(ti);
                split_ts[j].push(tj);
            }
        }
    }

    // T-junctions: endpoint of one segment on interior of another.
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            for endpoint in &[segments[j].0, segments[j].1] {
                if let Some(t) =
                    point_on_segment_t(endpoint, &segments[i].0, &segments[i].1, snap)
                {
                    if t > 1e-6 && t < 1.0 - 1e-6 {
                        split_ts[i].push(t);
                    }
                }
            }
        }
    }

    // Snap-rounded vertex pool.
    let mut vertex_pool: Vec<[f64; 2]> = Vec::new();
    let mut vertex_index: HashMap<(i64, i64), u32> = HashMap::new();
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut edge_set: HashSet<(u32, u32)> = HashSet::new();
    let mut endpoint_map: Vec<u32> = Vec::with_capacity(2 * n);

    let mut find_or_insert = |pool: &mut Vec<[f64; 2]>,
                              idx: &mut HashMap<(i64, i64), u32>,
                              pt: [f64; 2]|
     -> u32 {
        let k = key(pt);
        if let Some(&i) = idx.get(&k) {
            return i;
        }
        // Snap the stored coordinate to the grid centre so all sub-segments
        // sharing this key are perfectly collinear at their joint.
        let snapped = [(k.0 as f64) * snap, (k.1 as f64) * snap];
        let i = pool.len() as u32;
        pool.push(snapped);
        idx.insert(k, i);
        i
    };

    for (seg_i, seg) in segments.iter().enumerate() {
        let (p0, p1) = (seg.0, seg.1);
        let ts = &mut split_ts[seg_i];

        let ep_start = find_or_insert(&mut vertex_pool, &mut vertex_index, p0);
        let ep_end = find_or_insert(&mut vertex_pool, &mut vertex_index, p1);
        endpoint_map.push(ep_start);
        endpoint_map.push(ep_end);

        ts.push(0.0);
        ts.push(1.0);
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ts.dedup_by(|a, b| (*a - *b).abs() < 1e-8);

        for w in ts.windows(2) {
            let t0 = w[0];
            let t1 = w[1];
            if (t1 - t0).abs() < 1e-10 {
                continue;
            }
            let dx = p1[0] - p0[0];
            let dy = p1[1] - p0[1];
            let pt_a = [p0[0] + t0 * dx, p0[1] + t0 * dy];
            let pt_b = [p0[0] + t1 * dx, p0[1] + t1 * dy];

            let seg_dx = pt_b[0] - pt_a[0];
            let seg_dy = pt_b[1] - pt_a[1];
            if seg_dx * seg_dx + seg_dy * seg_dy < snap_sq {
                continue;
            }

            let vi = find_or_insert(&mut vertex_pool, &mut vertex_index, pt_a);
            let vj = find_or_insert(&mut vertex_pool, &mut vertex_index, pt_b);
            if vi != vj {
                let edge = if vi < vj { (vi, vj) } else { (vj, vi) };
                if edge_set.insert(edge) {
                    edges.push(edge);
                }
            }
        }
    }

    (vertex_pool, edges, endpoint_map)
}

/// Project a point onto a segment and return the parameter t if within tolerance.
fn point_on_segment_t(pt: &[f64; 2], a: &[f64; 2], b: &[f64; 2], tolerance: f64) -> Option<f64> {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-20 {
        return None;
    }
    let t = ((pt[0] - a[0]) * dx + (pt[1] - a[1]) * dy) / len_sq;
    if t < -0.01 || t > 1.01 {
        return None;
    }
    // Check distance from point to projected position
    let proj_x = a[0] + t * dx;
    let proj_y = a[1] + t * dy;
    let dist_sq = (pt[0] - proj_x) * (pt[0] - proj_x) + (pt[1] - proj_y) * (pt[1] - proj_y);
    if dist_sq < tolerance * tolerance {
        Some(t.clamp(0.0, 1.0))
    } else {
        None
    }
}

// ───────────────────── Dangling edge pruning ─────────────────────

/// Remove degree-1 vertices and their edges iteratively.
/// After removal, relinks next/prev pointers at affected vertices.
fn prune_dangling(
    half_edges: &mut Vec<ArrHalfEdge>,
    outgoing: &mut [Vec<u32>],
    verts: &[[f64; 2]],
) {
    let mut changed = true;
    while changed {
        changed = false;
        for vi in 0..outgoing.len() {
            let live: Vec<u32> = outgoing[vi]
                .iter()
                .copied()
                .filter(|&h| half_edges[h as usize].origin != u32::MAX)
                .collect();

            if live.len() == 1 {
                // Degree 1: mark this edge and its twin as dead
                let h = live[0];
                let t = half_edges[h as usize].twin;
                half_edges[h as usize].origin = u32::MAX;
                half_edges[t as usize].origin = u32::MAX;
                outgoing[vi].clear();

                // Also clean the other vertex's outgoing list
                let other_v = half_edges[t as usize].origin;
                // origin already set to MAX, but we need the original value
                // The twin's origin was the other vertex before we cleared it
                // We can find it: t is the twin going from other_v to vi
                // But we already set it to MAX. Let's track differently.

                changed = true;
            }
        }
        // Refresh all outgoing lists
        for out in outgoing.iter_mut() {
            out.retain(|&h| half_edges[h as usize].origin != u32::MAX);
        }
    }

    // Relink next/prev at all vertices that still have edges
    // (since pruning may have invalidated some links)
    for (vi, out) in outgoing.iter_mut().enumerate() {
        if out.len() < 2 {
            continue;
        }
        let vx = verts[vi][0];
        let vy = verts[vi][1];

        out.sort_by(|&a, &b| {
            let dest_a = half_edges[half_edges[a as usize].twin as usize].origin;
            let dest_b = half_edges[half_edges[b as usize].twin as usize].origin;
            let angle_a = (verts[dest_a as usize][1] - vy)
                .atan2(verts[dest_a as usize][0] - vx);
            let angle_b = (verts[dest_b as usize][1] - vy)
                .atan2(verts[dest_b as usize][0] - vx);
            angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        let k = out.len();
        for i in 0..k {
            let arriving = half_edges[out[i] as usize].twin;
            let departing = out[(i + k - 1) % k];
            half_edges[arriving as usize].next = departing;
            half_edges[departing as usize].prev = arriving;
        }
    }
}

// ───────────────────── Face extraction ─────────────────────

/// Extract face cycles from the linked DCEL.
fn extract_faces(
    verts: &[[f64; 2]],
    half_edges: &mut Vec<ArrHalfEdge>,
) -> Vec<ArrFace> {
    let mut faces = Vec::new();
    let mut visited = vec![false; half_edges.len()];

    for start_he in 0..half_edges.len() {
        if visited[start_he] || half_edges[start_he].origin == u32::MAX {
            continue;
        }
        if half_edges[start_he].next == NO_EDGE {
            continue;
        }

        // Walk the cycle
        let mut cycle = Vec::new();
        let mut he = start_he as u32;
        let mut valid = true;
        loop {
            if visited[he as usize] || half_edges[he as usize].origin == u32::MAX {
                valid = false;
                break;
            }
            visited[he as usize] = true;
            cycle.push(he);
            he = half_edges[he as usize].next;
            if he == start_he as u32 {
                break;
            }
            if cycle.len() > half_edges.len() {
                valid = false;
                break;
            }
        }

        if !valid || cycle.len() < 3 {
            continue;
        }

        // Compute signed area and centroid via shoelace
        let mut area2 = 0.0;
        let mut cx = 0.0;
        let mut cy = 0.0;
        for &h in &cycle {
            let o = half_edges[h as usize].origin as usize;
            let next_h = half_edges[h as usize].next;
            let d = half_edges[next_h as usize].origin as usize;
            let cross = verts[o][0] * verts[d][1] - verts[d][0] * verts[o][1];
            area2 += cross;
            cx += (verts[o][0] + verts[d][0]) * cross;
            cy += (verts[o][1] + verts[d][1]) * cross;
        }

        let area = area2 * 0.5;

        // Only keep bounded interior faces (positive area = CCW winding).
        // The caller must ensure footprint exterior ring is CCW.
        if area <= 1e-10 {
            continue;
        }

        let inv6a = 1.0 / (3.0 * area2);
        cx *= inv6a;
        cy *= inv6a;

        let face_idx = faces.len() as u32;
        for &h in &cycle {
            half_edges[h as usize].face = face_idx;
        }

        faces.push(ArrFace {
            edge: cycle[0],
            area: area.abs(),
            cx,
            cy,
        });
    }

    faces
}

// ───────────────────── Merged faces (Phase 5 dissolver) ─────────────────────

/// A merged group of arrangement faces sharing a single cluster id.
/// The boundary is given as one outer ring (CCW) plus zero or more hole
/// rings (CW), each a list of arrangement-vertex indices.  `members`
/// lists the source `Arrangement` face indices that compose the cluster.
#[derive(Debug, Clone)]
pub struct MergedFace {
    pub members: Vec<u32>,
    pub outer: Vec<u32>,
    pub holes: Vec<Vec<u32>>,
}

impl Arrangement {
    /// Merge groups of interior faces that share the same `cluster_id`
    /// across their twin half-edges into single polygons.  Faces whose
    /// id is absent from `cluster_id` are treated as outside the cluster
    /// set (i.e. their twin counts as a boundary edge for any cluster).
    ///
    /// This is the roofer `arr_dissolve_seg_edges` equivalent: it removes
    /// shared edges between same-label adjacent faces without mutating
    /// the underlying DCEL.  The caller decides the cluster equivalence
    /// (typically: same graph-cut label, plus optional same-plane Z
    /// agreement at the shared edge).
    pub fn build_merged_faces(
        &self,
        cluster_id: &std::collections::HashMap<u32, u32>,
    ) -> Vec<MergedFace> {
        use std::collections::{HashMap, HashSet};

        // Group source face indices by cluster.
        let mut by_cluster: HashMap<u32, Vec<u32>> = HashMap::new();
        for (&fi, &c) in cluster_id {
            by_cluster.entry(c).or_default().push(fi);
        }

        let mut out: Vec<MergedFace> = Vec::with_capacity(by_cluster.len());

        for (_cid, members) in by_cluster {
            let member_set: HashSet<u32> = members.iter().copied().collect();

            // Collect boundary half-edges of this cluster: ones whose
            // twin's face is NOT in the cluster.
            let mut boundary_he: Vec<u32> = Vec::new();
            for &fi in &members {
                let start = self.faces[fi as usize].edge;
                let mut h = start;
                loop {
                    let he = &self.half_edges[h as usize];
                    let twin_face = self.half_edges[he.twin as usize].face;
                    if !member_set.contains(&twin_face) {
                        boundary_he.push(h);
                    }
                    h = he.next;
                    if h == start || h == NO_EDGE {
                        break;
                    }
                }
            }
            if boundary_he.is_empty() {
                continue;
            }

            // Walk boundary cycles.  From a boundary half-edge `h` (with
            // the cluster on the LEFT), the next boundary half-edge is
            // obtained by following `h.next` and then radially crossing
            // any half-edge whose twin still points into the cluster.
            let mut visited: HashSet<u32> = HashSet::with_capacity(boundary_he.len());
            let mut cycles: Vec<Vec<u32>> = Vec::new();
            for &start_h in &boundary_he {
                if visited.contains(&start_h) {
                    continue;
                }
                let mut cycle: Vec<u32> = Vec::new();
                let mut h = start_h;
                let safety_cap = self.half_edges.len() + 4;
                let mut safety = safety_cap;
                loop {
                    if !visited.insert(h) {
                        break;
                    }
                    cycle.push(self.half_edges[h as usize].origin);
                    // Advance to the next boundary half-edge in the same cluster.
                    let mut cur = self.half_edges[h as usize].next;
                    let mut inner = self.half_edges.len();
                    while inner > 0 && cur != NO_EDGE {
                        let twin = self.half_edges[cur as usize].twin;
                        let tf = self.half_edges[twin as usize].face;
                        if !member_set.contains(&tf) {
                            break;
                        }
                        cur = self.half_edges[twin as usize].next;
                        inner -= 1;
                    }
                    if cur == NO_EDGE {
                        cycle.clear();
                        break;
                    }
                    h = cur;
                    if h == start_h {
                        break;
                    }
                    if safety == 0 {
                        cycle.clear();
                        break;
                    }
                    safety -= 1;
                }
                if cycle.len() >= 3 {
                    cycles.push(cycle);
                }
            }

            // Step E (2026-05-25): a cluster can produce MULTIPLE
            // disconnected positive cycles when its faces are not
            // 4-connected through the arrangement (e.g. two roof patches
            // pinched at a single vertex, or split by another label).
            // Emit ONE MergedFace per positive cycle, attaching each
            // negative cycle (hole) to whichever outer contains its
            // first vertex.
            let mut classified: Vec<(f64, Vec<u32>)> = cycles
                .into_iter()
                .map(|c| (self.signed_ring_area(&c), c))
                .collect();

            // Partition into outers (a > 0) and holes (a < 0).
            let mut outers: Vec<Vec<u32>> = Vec::new();
            let mut hole_rings: Vec<Vec<u32>> = Vec::new();
            for (area, ring) in classified.drain(..) {
                if area > 0.0 {
                    outers.push(ring);
                } else if area < 0.0 {
                    hole_rings.push(ring);
                }
            }
            if outers.is_empty() {
                continue;
            }

            // Assign each hole to the outer that geometrically contains
            // a probe point (the hole's first vertex).  If multiple
            // outers contain it, pick the smallest-area one (tightest
            // enclosure); if none, drop the hole defensively.
            let mut holes_per_outer: Vec<Vec<Vec<u32>>> =
                (0..outers.len()).map(|_| Vec::new()).collect();
            // Pre-compute outer areas (for tightest-enclosure tie-break).
            let outer_areas: Vec<f64> = outers
                .iter()
                .map(|r| self.signed_ring_area(r).abs())
                .collect();
            for hole in hole_rings {
                let probe_vi = hole[0] as usize;
                let probe = (self.vertices[probe_vi].x, self.vertices[probe_vi].y);
                let mut best: Option<usize> = None;
                for (oi, outer) in outers.iter().enumerate() {
                    if Self::point_in_ring(outer, &self.vertices, probe) {
                        best = match best {
                            None => Some(oi),
                            Some(prev) if outer_areas[oi] < outer_areas[prev] => Some(oi),
                            other => other,
                        };
                    }
                }
                if let Some(oi) = best {
                    holes_per_outer[oi].push(hole);
                }
            }

            for (oi, outer) in outers.into_iter().enumerate() {
                out.push(MergedFace {
                    members: members.clone(),
                    outer,
                    holes: std::mem::take(&mut holes_per_outer[oi]),
                });
            }
        }

        out
    }

    fn signed_ring_area(&self, ring: &[u32]) -> f64 {
        let mut a = 0.0;
        for i in 0..ring.len() {
            let j = (i + 1) % ring.len();
            let p = &self.vertices[ring[i] as usize];
            let q = &self.vertices[ring[j] as usize];
            a += p.x * q.y - q.x * p.y;
        }
        a * 0.5
    }

    /// Point-in-polygon (ray-casting) for a ring of vertex indices.
    /// Used by `build_merged_faces` (Step E) to assign each hole to its
    /// enclosing outer when a cluster has disconnected components.
    fn point_in_ring(
        ring: &[u32],
        vertices: &[ArrVertex],
        (px, py): (f64, f64),
    ) -> bool {
        let n = ring.len();
        if n < 3 {
            return false;
        }
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let vi = &vertices[ring[i] as usize];
            let vj = &vertices[ring[j] as usize];
            let intersect = ((vi.y > py) != (vj.y > py))
                && (px < (vj.x - vi.x) * (py - vi.y) / (vj.y - vi.y + f64::EPSILON) + vi.x);
            if intersect {
                inside = !inside;
            }
            j = i;
        }
        inside
    }
}

impl MergedFace {
    /// Triangulate this merged polygon (outer ring + holes) using earcutr,
    /// returning arrangement-vertex index triples.  Vertex indices in the
    /// output reference the same `Arrangement::vertices` array that the
    /// caller used to build this `MergedFace`.
    pub fn triangulate(&self, arr: &Arrangement) -> Vec<[u32; 3]> {
        let total = self.outer.len() + self.holes.iter().map(|h| h.len()).sum::<usize>();
        if self.outer.len() < 3 {
            return Vec::new();
        }
        let mut coords: Vec<f64> = Vec::with_capacity(2 * total);
        let mut vids: Vec<u32> = Vec::with_capacity(total);
        for &vi in &self.outer {
            let v = &arr.vertices[vi as usize];
            coords.push(v.x);
            coords.push(v.y);
            vids.push(vi);
        }
        let mut hole_offsets: Vec<usize> = Vec::with_capacity(self.holes.len());
        for hole in &self.holes {
            hole_offsets.push(vids.len());
            for &vi in hole {
                let v = &arr.vertices[vi as usize];
                coords.push(v.x);
                coords.push(v.y);
                vids.push(vi);
            }
        }
        let tri = earcutr::earcut(&coords, &hole_offsets, 2).unwrap_or_default();
        let mut out = Vec::with_capacity(tri.len() / 3);
        for ch in tri.chunks_exact(3) {
            out.push([vids[ch[0]], vids[ch[1]], vids[ch[2]]]);
        }
        out
    }
}

// ───────────────────── Tests ─────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_no_ridges() {
        // Simple square: 4 edges forming a closed polygon
        let segments = vec![
            ([0.0, 0.0], [10.0, 0.0]),
            ([10.0, 0.0], [10.0, 10.0]),
            ([10.0, 10.0], [0.0, 10.0]),
            ([0.0, 10.0], [0.0, 0.0]),
        ];
        let (arr, _) = Arrangement::build(&segments, 0.01);
        // Should have 1 interior face (the square)
        assert_eq!(arr.faces.len(), 1, "square should have 1 interior face");
        let verts = arr.face_vertex_indices(0);
        assert_eq!(verts.len(), 4, "square face should have 4 vertices");
        assert!((arr.faces[0].area - 100.0).abs() < 1.0, "area should be ~100");
    }

    #[test]
    fn test_square_with_diagonal() {
        // Square + one diagonal ridge → 2 triangular faces
        let segments = vec![
            ([0.0, 0.0], [10.0, 0.0]),
            ([10.0, 0.0], [10.0, 10.0]),
            ([10.0, 10.0], [0.0, 10.0]),
            ([0.0, 10.0], [0.0, 0.0]),
            ([0.0, 0.0], [10.0, 10.0]), // diagonal
        ];
        let (arr, _) = Arrangement::build(&segments, 0.01);
        assert_eq!(arr.faces.len(), 2, "square+diagonal should have 2 faces");
        let total_area: f64 = arr.faces.iter().map(|f| f.area).sum();
        assert!((total_area - 100.0).abs() < 1.0, "total area should be ~100");
    }

    #[test]
    fn test_square_with_midline() {
        // Square + horizontal midline → 2 rectangular faces
        let segments = vec![
            ([0.0, 0.0], [10.0, 0.0]),
            ([10.0, 0.0], [10.0, 10.0]),
            ([10.0, 10.0], [0.0, 10.0]),
            ([0.0, 10.0], [0.0, 0.0]),
            ([0.0, 5.0], [10.0, 5.0]), // midline
        ];
        let (arr, _) = Arrangement::build(&segments, 0.01);
        assert_eq!(arr.faces.len(), 2, "square+midline should have 2 faces");
        for f in &arr.faces {
            assert!((f.area - 50.0).abs() < 1.0, "each half should be ~50 sq units");
        }
    }

    #[test]
    fn test_crossing_ridges() {
        // Square + two crossing ridges → 4 faces
        let segments = vec![
            ([0.0, 0.0], [10.0, 0.0]),
            ([10.0, 0.0], [10.0, 10.0]),
            ([10.0, 10.0], [0.0, 10.0]),
            ([0.0, 10.0], [0.0, 0.0]),
            ([0.0, 5.0], [10.0, 5.0]),  // horizontal midline
            ([5.0, 0.0], [5.0, 10.0]),  // vertical midline
        ];
        let (arr, _) = Arrangement::build(&segments, 0.01);
        assert_eq!(arr.faces.len(), 4, "crossing ridges should create 4 faces");
        let total_area: f64 = arr.faces.iter().map(|f| f.area).sum();
        assert!((total_area - 100.0).abs() < 1.0, "total area should be ~100");
    }

    #[test]
    fn test_triangulation() {
        let segments = vec![
            ([0.0, 0.0], [10.0, 0.0]),
            ([10.0, 0.0], [10.0, 10.0]),
            ([10.0, 10.0], [0.0, 10.0]),
            ([0.0, 10.0], [0.0, 0.0]),
        ];
        let (arr, _) = Arrangement::build(&segments, 0.01);
        let tris = arr.triangulate_face(0);
        assert_eq!(tris.len(), 2, "square should triangulate into 2 triangles");
    }

    #[test]
    fn test_dangling_ridge() {
        // Square + ridge that stops in the middle → should be pruned
        let segments = vec![
            ([0.0, 0.0], [10.0, 0.0]),
            ([10.0, 0.0], [10.0, 10.0]),
            ([10.0, 10.0], [0.0, 10.0]),
            ([0.0, 10.0], [0.0, 0.0]),
            ([5.0, 0.0], [5.0, 5.0]), // dangling: ends at (5,5) interior
        ];
        let (arr, _) = Arrangement::build(&segments, 0.01);
        // After pruning, should still have 1 face (the square) since dangling edge removed
        assert_eq!(arr.faces.len(), 1, "dangling ridge should be pruned");
    }

    #[test]
    fn test_neighbors() {
        // Square + midline → 2 faces that are neighbors
        let segments = vec![
            ([0.0, 0.0], [10.0, 0.0]),
            ([10.0, 0.0], [10.0, 10.0]),
            ([10.0, 10.0], [0.0, 10.0]),
            ([0.0, 10.0], [0.0, 0.0]),
            ([0.0, 5.0], [10.0, 5.0]),
        ];
        let (arr, _) = Arrangement::build(&segments, 0.01);
        assert_eq!(arr.faces.len(), 2);
        let n0 = arr.face_neighbors(0);
        let n1 = arr.face_neighbors(1);
        assert!(n0.contains(&1), "face 0 should neighbor face 1");
        assert!(n1.contains(&0), "face 1 should neighbor face 0");
    }
}
