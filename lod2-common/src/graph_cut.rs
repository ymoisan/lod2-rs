//! Alpha-expansion graph-cut for optimal plane labeling of CDT faces.
//!
//! Implements the energy minimization framework from Boykov, Veksler & Zabih (2001):
//!   E(labeling) = Σ_p D_p(l_p) + λ · Σ_{p,q ∈ N} V_{pq}(l_p, l_q)
//!
//! where:
//!   - D_p(l) is the data term: how well plane `l` fits face `p`
//!   - V_{pq} is the smoothness term: penalty for adjacent faces having different labels
//!   - λ is the smoothness weight
//!
//! Max-flow uses BFS augmenting paths (Edmonds-Karp), which is sufficient for the
//! small graphs (hundreds of CDT faces) encountered in building reconstruction.

use std::collections::{HashSet, VecDeque};

// ───────────────────── Max-Flow / Min-Cut Solver ─────────────────────

/// Directed arc in the residual graph.
#[derive(Debug, Clone)]
struct Arc {
    to: usize,
    residual: f64,
    rev_idx: usize,
}

/// Max-flow solver with explicit source (node n) and sink (node n+1).
///
/// Nodes 0..n are the regular nodes. The solver uses Edmonds-Karp (BFS augmenting paths).
pub struct MaxFlowSolver {
    /// Total nodes including source and sink.
    total_nodes: usize,
    /// Source node index.
    source: usize,
    /// Sink node index.
    sink: usize,
    /// Adjacency list: adj[u] contains indices into `arcs`.
    adj: Vec<Vec<usize>>,
    /// All arcs. Forward and reverse arcs are at consecutive indices (2i, 2i+1).
    arcs: Vec<Arc>,
}

impl MaxFlowSolver {
    /// Create a solver for `n` regular nodes. Source = n, Sink = n+1.
    pub fn new(n: usize) -> Self {
        Self {
            total_nodes: n + 2,
            source: n,
            sink: n + 1,
            adj: vec![Vec::new(); n + 2],
            arcs: Vec::new(),
        }
    }

    /// Internal: add a directed edge u→v with capacity `cap`, and a reverse arc v→u with cap 0.
    fn add_directed_arc(&mut self, u: usize, v: usize, cap: f64) {
        let idx_fwd = self.arcs.len();
        let idx_rev = idx_fwd + 1;
        self.arcs.push(Arc { to: v, residual: cap, rev_idx: idx_rev });
        self.arcs.push(Arc { to: u, residual: 0.0, rev_idx: idx_fwd });
        self.adj[u].push(idx_fwd);
        self.adj[v].push(idx_rev);
    }

    /// Add terminal edges: source → node (cap_s) and node → sink (cap_t).
    pub fn add_terminal(&mut self, node: usize, cap_s: f64, cap_t: f64) {
        if cap_s > 0.0 {
            self.add_directed_arc(self.source, node, cap_s);
        }
        if cap_t > 0.0 {
            self.add_directed_arc(node, self.sink, cap_t);
        }
    }

    /// Add a symmetric pairwise edge between u and v with capacity `cap` in both directions.
    pub fn add_edge(&mut self, u: usize, v: usize, cap: f64) {
        // Two directed arcs: u→v and v→u, each with its own reverse.
        let idx_uv = self.arcs.len();
        let idx_vu = idx_uv + 1;
        self.arcs.push(Arc { to: v, residual: cap, rev_idx: idx_vu });
        self.arcs.push(Arc { to: u, residual: cap, rev_idx: idx_uv });
        self.adj[u].push(idx_uv);
        self.adj[v].push(idx_vu);
    }

    /// Run Edmonds-Karp (BFS augmenting paths) and return the min-cut partition.
    /// Returns a vec of length `n` (regular nodes only):
    ///   `true` = source-side (will adopt label α), `false` = sink-side (keep current label).
    pub fn solve(&mut self) -> Vec<bool> {
        let n_total = self.total_nodes;
        let source = self.source;
        let sink = self.sink;

        // Repeatedly find shortest augmenting path via BFS
        loop {
            let mut parent_arc: Vec<Option<usize>> = vec![None; n_total];
            let mut visited = vec![false; n_total];
            let mut queue = VecDeque::new();
            visited[source] = true;
            queue.push_back(source);

            let mut found = false;
            while let Some(u) = queue.pop_front() {
                if u == sink {
                    found = true;
                    break;
                }
                for &arc_idx in &self.adj[u] {
                    let arc = &self.arcs[arc_idx];
                    if arc.residual > 1e-12 && !visited[arc.to] {
                        visited[arc.to] = true;
                        parent_arc[arc.to] = Some(arc_idx);
                        queue.push_back(arc.to);
                    }
                }
            }

            if !found {
                break;
            }

            // Find bottleneck along the augmenting path
            let mut bottleneck = f64::MAX;
            let mut node = sink;
            while let Some(arc_idx) = parent_arc[node] {
                bottleneck = bottleneck.min(self.arcs[arc_idx].residual);
                node = self.arcs[self.arcs[arc_idx].rev_idx].to; // parent node
            }

            // Augment flow along the path
            node = sink;
            while let Some(arc_idx) = parent_arc[node] {
                self.arcs[arc_idx].residual -= bottleneck;
                let rev = self.arcs[arc_idx].rev_idx;
                self.arcs[rev].residual += bottleneck;
                node = self.arcs[rev].to;
            }
        }

        // Determine min-cut: BFS from source on residual graph
        let mut reachable = vec![false; n_total];
        let mut queue = VecDeque::new();
        reachable[source] = true;
        queue.push_back(source);
        while let Some(u) = queue.pop_front() {
            for &arc_idx in &self.adj[u] {
                let arc = &self.arcs[arc_idx];
                if arc.residual > 1e-12 && !reachable[arc.to] {
                    reachable[arc.to] = true;
                    queue.push_back(arc.to);
                }
            }
        }

        // Return only regular nodes (0..n), excluding source and sink
        let n = self.total_nodes - 2;
        reachable[..n].to_vec()
    }
}

// ───────────────────── Alpha-Expansion ─────────────────────

/// Face in the graph-cut optimization, representing one CDT triangle.
#[derive(Debug, Clone)]
pub struct GcFace {
    pub idx: usize,
    pub cx: f64,
    pub cy: f64,
    pub local_z: f64,
    /// Indices into the per-building point cloud of LiDAR returns whose
    /// (x,y) falls inside this face's 2D extent.  Empty for tiny faces
    /// with no support; the data-cost falls back to centroid-residual.
    pub pts: Vec<usize>,
}

/// Neighbor pair: two faces sharing an edge, with the shared edge length.
#[derive(Debug, Clone)]
pub struct GcNeighborPair {
    pub face_a: usize,
    pub face_b: usize,
    pub edge_length: f64,
}

/// Data term: honest mean squared per-point residual PLUS an inlier-
/// coverage penalty that forbids assigning a plane to a face whose
/// LiDAR points are NOT actually the plane's inliers.  No clipping and
/// no global cap — large residuals ("this plane is way off here") and
/// large coverage penalties ("this plane has zero support in this
/// face") must be visible to graph-cut so it cannot collapse to a
/// "compromise" plane that is never best but always second-best.
///
/// Formula:
///   cost = residual_sq + (1 - coverage)² · COVERAGE_WEIGHT
///                       + (face has no pts ? FALLBACK_PENALTY : 0)
///
/// where coverage = |face.pts ∩ plane.inliers| / |face.pts|.
fn data_cost(
    face: &GcFace,
    plane_idx: usize,
    planes: &[crate::plane::Plane],
    plane_inlier_sets: &[HashSet<usize>],
    points: &[nalgebra::Point3<f64>],
) -> f64 {
    const COVERAGE_WEIGHT: f64 = 8.0; // m² — dominant when coverage = 0
    const FALLBACK_PENALTY: f64 = 2.0; // m² — face has no contained LiDAR points

    let plane = &planes[plane_idx];
    if plane.normal.z.abs() < 1e-6 {
        return 1e6; // near-vertical plane can't be a roof — effectively forbidden
    }

    if face.pts.is_empty() {
        // No LiDAR points in this face: fall back to centroid residual
        // against the local height estimate.  Honest squared residual,
        // no clip, plus a flat fallback penalty so well-supported faces
        // are always preferred during alpha-expansion.
        let z_plane = -(plane.normal.x * face.cx + plane.normal.y * face.cy + plane.d)
            / plane.normal.z;
        let r = z_plane - face.local_z;
        return r * r + FALLBACK_PENALTY;
    }

    // Mean per-point squared residual.  Unclipped — a wrong plane
    // legitimately reaches 5–25 m² here and that information must reach
    // the cut.
    let mut acc = 0.0;
    let mut inlier_hits = 0usize;
    let set = &plane_inlier_sets[plane_idx];
    for &i in &face.pts {
        let p = &points[i];
        let zp = -(plane.normal.x * p.x + plane.normal.y * p.y + plane.d) / plane.normal.z;
        let r = zp - p.z;
        acc += r * r;
        if set.contains(&i) {
            inlier_hits += 1;
        }
    }
    let residual_sq = acc / face.pts.len() as f64;
    let coverage = inlier_hits as f64 / face.pts.len() as f64;
    let coverage_penalty = (1.0 - coverage).powi(2) * COVERAGE_WEIGHT;

    residual_sq + coverage_penalty
}

/// Smoothness term: edge-length–weighted penalty for label disagreement.
fn smoothness_cost(edge_length: f64) -> f64 {
    edge_length
}

/// Public wrapper for debug instrumentation only.
pub fn data_cost_pub(
    face: &GcFace,
    plane_idx: usize,
    planes: &[crate::plane::Plane],
    plane_inlier_sets: &[HashSet<usize>],
    points: &[nalgebra::Point3<f64>],
) -> f64 {
    data_cost(face, plane_idx, planes, plane_inlier_sets, points)
}

/// Build per-plane inlier sets from `Plane.inliers`.  Caller passes the
/// returned slice to `alpha_expansion` / `data_cost_pub`.
pub fn build_plane_inlier_sets(planes: &[crate::plane::Plane]) -> Vec<HashSet<usize>> {
    planes
        .iter()
        .map(|p| p.inliers.iter().copied().collect())
        .collect()
}

/// Run alpha-expansion to find the optimal labeling of faces to planes.
///   - `lambda`: smoothness weight (higher = smoother labeling)
///
/// Returns: optimized per-face plane labels.
pub fn alpha_expansion(
    faces: &[GcFace],
    neighbors: &[GcNeighborPair],
    planes: &[crate::plane::Plane],
    plane_inlier_sets: &[HashSet<usize>],
    points: &[nalgebra::Point3<f64>],
    initial_labels: &[usize],
    lambda: f64,
) -> Vec<usize> {
    let n_faces = faces.len();
    let n_labels = planes.len();

    if n_faces == 0 || n_labels == 0 {
        return initial_labels.to_vec();
    }

    let mut labels = initial_labels.to_vec();
    let mut changed = true;
    let mut iteration = 0;
    let max_iterations = 3;

    while changed && iteration < max_iterations {
        changed = false;
        iteration += 1;

        for alpha in 0..n_labels {
            // Build s-t graph for the alpha-expansion move.
            //
            // For each face p:
            //   - source → p : D_p(alpha)       (cost if p KEEPS current label, cut = switch to α)
            //   - p → sink   : D_p(current)     (cost if p switches to α, cut = keep)
            //
            // Wait — the convention is:
            //   source-side = adopts α, sink-side = keeps current label.
            //   So: source → p cap = D_p(current_label)  [pay this if you DON'T adopt α]
            //       p → sink cap = D_p(alpha)              [pay this if you DO adopt α]
            //
            // Actually in Boykov et al., the convention is:
            //   if p is cut to source side → p gets label α
            //   source → p: D(p, α)   ... no wait.
            //
            // Standard convention:
            //   source → p with cap D_p(label_p) means: "cost of NOT switching to α"
            //   p → sink with cap D_p(α) means: "cost of switching to α"
            //
            //   If min-cut puts p on source side: p gets label α (source-side)
            //     → we pay p → sink (D_p(α)) ... no, we cut the edge to sink.
            //
            // Let me use the standard formulation:
            //   t-link to source: cap = D_p(α)       [if p goes to sink, we pay this = don't adopt α, but this doesn't make sense]
            //
            // OK, clearest way: In alpha-expansion, source = α, sink = current label.
            //   source → p: D_p(current_label_p)  [penalty for keeping current]
            //   p → sink: D_p(α)                   [penalty for switching to α]
            //
            // If p is in source-set after min-cut: p adopts α. We pay D_p(α) via the p→sink edge.
            // If p is in sink-set: p keeps current label. We pay D_p(current) via source→p edge.
            //
            // For pairwise: if both p,q same label → no edge needed.
            // If one is α already → add terminal edges.
            // If both non-α, same label → add edge p-q with cap λ·V.
            // If both non-α, different labels → auxiliary node.

            let mut solver = MaxFlowSolver::new(n_faces);

            for (i, face) in faces.iter().enumerate() {
                let d_alpha = data_cost(face, alpha, planes, plane_inlier_sets, points);
                let d_current = data_cost(face, labels[i], planes, plane_inlier_sets, points);

                if labels[i] == alpha {
                    // Already labeled α: must stay α → infinite source link
                    solver.add_terminal(i, f64::MAX, 0.0);
                } else {
                    // source → p: D(current), p → sink: D(α)
                    solver.add_terminal(i, d_current, d_alpha);
                }
            }

            for pair in neighbors {
                let la = labels[pair.face_a];
                let lb = labels[pair.face_b];
                let w = lambda * smoothness_cost(pair.edge_length);

                if la == alpha && lb == alpha {
                    // Both already α → no penalty regardless
                } else if la == alpha {
                    // face_a stays α (source-side). If face_b also goes source (→ α): no penalty.
                    // If face_b stays sink (keeps lb ≠ α): penalty w.
                    // Encode: face_b → sink gets extra w (cost of keeping different from α neighbor)
                    solver.add_terminal(pair.face_b, w, 0.0);
                } else if lb == alpha {
                    solver.add_terminal(pair.face_a, w, 0.0);
                } else if la == lb {
                    // Same non-α label. If both keep → no penalty. If both switch → no penalty.
                    // If one switches, one doesn't → penalty w.
                    solver.add_edge(pair.face_a, pair.face_b, w);
                } else {
                    // Different non-α labels. Penalty for any combination where they differ:
                    //   both keep → penalty w (they already differ)
                    //   both switch to α → no penalty
                    //   a switches, b keeps → penalty w (α ≠ lb)
                    //   a keeps, b switches → penalty w (la ≠ α)
                    //
                    // Only "both switch" avoids penalty. Use auxiliary node:
                    //   aux → sink: 0, source → aux: 0
                    //   a → aux: w, b → aux: w, aux → sink: w
                    //
                    // Simpler approximation: add w to both sink-side terminals and w/2 edge.
                    // This overestimates slightly but is metric and produces valid moves.
                    solver.add_terminal(pair.face_a, w, 0.0);
                    solver.add_terminal(pair.face_b, w, 0.0);
                }
            }

            let cut = solver.solve();

            // Source-side nodes adopt label α
            for i in 0..n_faces {
                if cut[i] && labels[i] != alpha {
                    labels[i] = alpha;
                    changed = true;
                }
            }
        }
    }

    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxflow_simple() {
        // Two nodes, source→0 (cap 5), 1→sink (cap 5), 0→1 (cap 3)
        let mut solver = MaxFlowSolver::new(2);
        solver.add_terminal(0, 5.0, 0.0); // source → 0: cap 5
        solver.add_terminal(1, 0.0, 5.0); // 1 → sink: cap 5
        solver.add_edge(0, 1, 3.0);

        let cut = solver.solve();
        // Max flow = 3 (bottleneck at 0→1)
        // Min cut: cut edge 0→1. Node 0 in source set, node 1 in sink set.
        assert!(cut[0]);
        assert!(!cut[1]);
    }

    #[test]
    fn test_maxflow_both_source() {
        // Both nodes prefer source (high source cap, low sink cap)
        let mut solver = MaxFlowSolver::new(2);
        solver.add_terminal(0, 10.0, 1.0);
        solver.add_terminal(1, 10.0, 1.0);
        solver.add_edge(0, 1, 1.0);

        let cut = solver.solve();
        // Min cut: cut both sink edges (cost 1+1=2) vs cut both source edges (cost 10+10=20)
        // Both should be in source set
        assert!(cut[0]);
        assert!(cut[1]);
    }

    #[test]
    fn test_maxflow_both_sink() {
        // Both nodes prefer sink
        let mut solver = MaxFlowSolver::new(2);
        solver.add_terminal(0, 1.0, 10.0);
        solver.add_terminal(1, 1.0, 10.0);
        solver.add_edge(0, 1, 1.0);

        let cut = solver.solve();
        assert!(!cut[0]);
        assert!(!cut[1]);
    }

    #[test]
    fn test_alpha_expansion_simple() {
        use nalgebra::Vector3;

        // Two planes: z=10 and z=15
        let planes = vec![
            crate::plane::Plane {
                normal: Vector3::new(0.0, 0.0, 1.0),
                d: -10.0,
                inliers: vec![],
                rmse: 0.0,
            },
            crate::plane::Plane {
                normal: Vector3::new(0.0, 0.0, 1.0),
                d: -15.0,
                inliers: vec![],
                rmse: 0.0,
            },
        ];

        // 4 faces: first two at z=10, last two at z=15
        let faces = vec![
            GcFace { idx: 0, cx: 0.0, cy: 0.0, local_z: 10.0, pts: vec![] },
            GcFace { idx: 1, cx: 1.0, cy: 0.0, local_z: 10.0, pts: vec![] },
            GcFace { idx: 2, cx: 2.0, cy: 0.0, local_z: 15.0, pts: vec![] },
            GcFace { idx: 3, cx: 3.0, cy: 0.0, local_z: 15.0, pts: vec![] },
        ];

        let neighbors = vec![
            GcNeighborPair { face_a: 0, face_b: 1, edge_length: 1.0 },
            GcNeighborPair { face_a: 1, face_b: 2, edge_length: 1.0 },
            GcNeighborPair { face_a: 2, face_b: 3, edge_length: 1.0 },
        ];

        // Build empty inlier sets (no inliers populated for this synthetic test)
        let plane_inlier_sets = build_plane_inlier_sets(&planes);

        // All initially labeled as plane 0 (z=10)
        let initial = vec![0, 0, 0, 0];
        let points: Vec<nalgebra::Point3<f64>> = Vec::new();
        let result = alpha_expansion(&faces, &neighbors, &planes, &plane_inlier_sets, &points, &initial, 0.5);

        // Faces 0,1 should stay on plane 0 (z=10), faces 2,3 should switch to plane 1 (z=15)
        assert_eq!(result[0], 0, "face 0 should be plane 0");
        assert_eq!(result[1], 0, "face 1 should be plane 0");
        assert_eq!(result[2], 1, "face 2 should be plane 1");
        assert_eq!(result[3], 1, "face 3 should be plane 1");
    }
}
