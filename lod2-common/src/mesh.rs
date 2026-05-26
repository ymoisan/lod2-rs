use crate::polygon::AttributeMap;
use nalgebra::{Point3, Vector3};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceType {
    GroundSurface,
    WallSurface,
    RoofSurface,
    ClosureSurface,
}

#[derive(Debug, Clone)]
pub struct SemanticSurface {
    pub surface_type: SurfaceType,
    pub on_footprint_edge: Option<bool>,
    pub azimuth: Option<f64>,
    pub slope: Option<f64>,
    pub h_roof_50p: Option<f64>,
    pub h_roof_70p: Option<f64>,
    pub h_roof_min: Option<f64>,
    pub h_roof_max: Option<f64>,
}

impl SemanticSurface {
    pub fn ground() -> Self {
        Self {
            surface_type: SurfaceType::GroundSurface,
            on_footprint_edge: None,
            azimuth: None, slope: None,
            h_roof_50p: None, h_roof_70p: None, h_roof_min: None, h_roof_max: None,
        }
    }

    pub fn wall(on_edge: bool) -> Self {
        Self {
            surface_type: SurfaceType::WallSurface,
            on_footprint_edge: Some(on_edge),
            azimuth: None, slope: None,
            h_roof_50p: None, h_roof_70p: None, h_roof_min: None, h_roof_max: None,
        }
    }

    pub fn roof() -> Self {
        Self {
            surface_type: SurfaceType::RoofSurface,
            on_footprint_edge: None,
            azimuth: None, slope: None,
            h_roof_50p: None, h_roof_70p: None, h_roof_min: None, h_roof_max: None,
        }
    }

    pub fn roof_with_stats(slope: f64, azimuth: f64) -> Self {
        Self {
            surface_type: SurfaceType::RoofSurface,
            on_footprint_edge: None,
            azimuth: Some(azimuth),
            slope: Some(slope),
            h_roof_50p: None, h_roof_70p: None, h_roof_min: None, h_roof_max: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Face {
    pub indices: Vec<u32>,
    pub holes: Vec<Vec<u32>>,
    pub semantic_index: Option<usize>,
}

impl Face {
    pub fn new(indices: Vec<u32>) -> Self {
        Self { indices, holes: Vec::new(), semantic_index: None }
    }

    pub fn with_semantic(mut self, idx: usize) -> Self {
        self.semantic_index = Some(idx);
        self
    }

    pub fn with_hole(mut self, hole: Vec<u32>) -> Self {
        self.holes.push(hole);
        self
    }
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Point3<f64>>,
    pub faces: Vec<Face>,
    pub semantics: Vec<SemanticSurface>,
}

impl Mesh {
    pub fn new() -> Self {
        Self { vertices: Vec::new(), faces: Vec::new(), semantics: Vec::new() }
    }

    pub fn add_vertex(&mut self, v: Point3<f64>) -> u32 {
        let idx = self.vertices.len() as u32;
        self.vertices.push(v);
        idx
    }

    pub fn add_face(&mut self, face: Face) {
        self.faces.push(face);
    }

    pub fn add_semantic(&mut self, semantic: SemanticSurface) -> usize {
        let idx = self.semantics.len();
        self.semantics.push(semantic);
        idx
    }

    pub fn face_normal(&self, face_idx: usize) -> Option<Vector3<f64>> {
        let face = &self.faces[face_idx];
        if face.indices.len() < 3 {
            return None;
        }
        let p0 = &self.vertices[face.indices[0] as usize];
        let p1 = &self.vertices[face.indices[1] as usize];
        let p2 = &self.vertices[face.indices[2] as usize];
        let v1 = p1 - p0;
        let v2 = p2 - p0;
        let n = v1.cross(&v2);
        let len = n.norm();
        if len < 1e-15 {
            return None;
        }
        Some(n / len)
    }

    pub fn compute_volume(&self) -> f64 {
        let mut volume = 0.0;
        for face in &self.faces {
            if face.indices.len() < 3 {
                continue;
            }
            let p0 = &self.vertices[face.indices[0] as usize];
            for i in 1..(face.indices.len() - 1) {
                let p1 = &self.vertices[face.indices[i] as usize];
                let p2 = &self.vertices[face.indices[i + 1] as usize];
                volume += p0.coords.dot(&p1.coords.cross(&p2.coords));
            }
        }
        (volume / 6.0).abs()
    }

    /// Signed tetra-sum volume (origin reference).  Positive means outward
    /// face normals; negative means inward (face winding is reversed).
    pub fn signed_volume(&self) -> f64 {
        let mut volume = 0.0;
        for face in &self.faces {
            if face.indices.len() < 3 {
                continue;
            }
            let p0 = &self.vertices[face.indices[0] as usize];
            for i in 1..(face.indices.len() - 1) {
                let p1 = &self.vertices[face.indices[i] as usize];
                let p2 = &self.vertices[face.indices[i + 1] as usize];
                volume += p0.coords.dot(&p1.coords.cross(&p2.coords));
            }
        }
        volume / 6.0
    }

    /// Reverse the winding of every face (so face normals flip).  Holes are
    /// also reversed so the relative orientation is preserved.
    pub fn reverse_winding(&mut self) {
        for face in &mut self.faces {
            face.indices.reverse();
            for hole in &mut face.holes {
                hole.reverse();
            }
        }
    }

    /// Normalise face winding so the mesh has outward-pointing normals.
    /// Uses `signed_volume`; if negative, reverses all faces.  No-op on
    /// open meshes whose signed volume is near zero.
    pub fn normalise_winding(&mut self) {
        let v = self.signed_volume();
        if v < 0.0 {
            self.reverse_winding();
        }
    }

    /// Seal small boundary holes by fan-triangulating each boundary cycle.
    ///
    /// Boundary edges (used by exactly one face when considered directed)
    /// are grouped into closed cycles via an outgoing-edge map.  Each
    /// cycle of length ≥ 3 and ≤ `max_ring` is closed by adding a fan of
    /// triangles from `cycle[0]` to every other edge.  Larger cycles
    /// (likely real holes such as courtyards) are left untouched.
    ///
    /// Returns the number of triangles emitted.
    pub fn seal_holes(&mut self, max_ring: usize) -> usize {
        // Collect directed edges from every face boundary (skip holes since
        // they are part of the same face).  Triangulate explicit holes only
        // when fan-walking, not when looking at boundary edges.
        let mut all_edges: HashSet<(u32, u32)> = HashSet::new();
        for face in &self.faces {
            let inds = &face.indices;
            let n = inds.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                all_edges.insert((inds[i], inds[(i + 1) % n]));
            }
            // Hole boundaries: opposite winding.
            for hole in &face.holes {
                let h = hole.len();
                if h < 3 {
                    continue;
                }
                for i in 0..h {
                    all_edges.insert((hole[i], hole[(i + 1) % h]));
                }
            }
        }

        // Boundary directed edges: those whose reverse is absent.
        let mut boundary: Vec<(u32, u32)> = Vec::new();
        for &(a, b) in &all_edges {
            if !all_edges.contains(&(b, a)) {
                boundary.push((a, b));
            }
        }
        if boundary.is_empty() {
            return 0;
        }

        // Outgoing map.
        let mut outgoing: HashMap<u32, Vec<u32>> = HashMap::new();
        for &(a, b) in &boundary {
            outgoing.entry(a).or_default().push(b);
        }

        // Walk into cycles.  Pop one outgoing at a time; when we hit a
        // vertex with no outgoing remaining we abandon the cycle.
        let mut emitted = 0usize;
        let total_boundary = boundary.len();
        let mut consumed = 0usize;
        let mut cycles: Vec<Vec<u32>> = Vec::new();
        let mut seen_start: HashSet<u32> = HashSet::new();
        for &(start, _) in &boundary {
            if seen_start.contains(&start) {
                continue;
            }
            if outgoing.get(&start).map_or(true, |v| v.is_empty()) {
                continue;
            }
            let mut cycle: Vec<u32> = Vec::new();
            let mut cur = start;
            let mut safety = total_boundary.saturating_mul(2) + 4;
            loop {
                cycle.push(cur);
                seen_start.insert(cur);
                let nexts = match outgoing.get_mut(&cur) {
                    Some(v) if !v.is_empty() => v,
                    _ => {
                        cycle.clear();
                        break;
                    }
                };
                let next = nexts.swap_remove(0);
                consumed += 1;
                if next == start {
                    break;
                }
                cur = next;
                if safety == 0 {
                    cycle.clear();
                    break;
                }
                safety -= 1;
            }
            if cycle.len() >= 3 {
                cycles.push(cycle);
            }
            if consumed >= total_boundary {
                break;
            }
        }

        // Seal each cycle.  Skip rings outside the size window so we do not
        // close real courtyards or worse, slice across the interior.
        for cycle in cycles {
            if cycle.len() < 3 || cycle.len() > max_ring {
                continue;
            }
            // Skip cycles that touch the same vertex more than once: a fan
            // through such a cycle would produce duplicate or zero-area
            // triangles.  Splitting them safely is left for a future pass.
            let mut sorted = cycle.clone();
            sorted.sort_unstable();
            let unique = sorted.windows(2).filter(|w| w[0] != w[1]).count() + 1;
            if unique != cycle.len() {
                continue;
            }
            // Manifold-preserving fan in REVERSE winding.  The boundary
            // cycle walks directed edges v_i→v_{i+1} that are currently
            // use=1 (their reverse is absent).  A sealing triangle must
            // contribute the REVERSE edge v_{i+1}→v_i, bumping each to
            // use=2.  Internal fan edges (a↔v_k for 1<k<n-1) appear in one
            // direction per triangle; the adjacent fan triangle adds the
            // opposite direction, so they pair up across the fan.
            // Safety: every new directed edge must be absent from
            // `all_edges` at the moment of insertion.
            let a = cycle[0];
            for k in 1..(cycle.len() - 1) {
                let b = cycle[k];
                let c = cycle[k + 1];
                if a == b || b == c || a == c {
                    continue;
                }
                // Reversed-winding triangle (a, c, b) ⇒ directed edges
                // a→c, c→b, b→a.
                let new_edges = [(a, c), (c, b), (b, a)];
                if new_edges.iter().any(|e| all_edges.contains(e)) {
                    continue;
                }
                for e in &new_edges {
                    all_edges.insert(*e);
                }
                self.faces.push(Face::new(vec![a, c, b]));
                emitted += 1;
            }
        }
        emitted
    }

    /// Merge adjacent roof triangles that share the same semantic surface into larger polygons.
    /// Ground, wall, and closure faces are left untouched.
    pub fn merge_coplanar_roof_faces(&mut self) {
        // Partition faces into roof vs non-roof
        let mut roof_by_semantic: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut non_roof_faces: Vec<Face> = Vec::new();

        for (fi, face) in self.faces.iter().enumerate() {
            let is_roof = face.semantic_index
                .and_then(|si| self.semantics.get(si))
                .map(|s| s.surface_type == SurfaceType::RoofSurface)
                .unwrap_or(false);
            if is_roof {
                roof_by_semantic
                    .entry(face.semantic_index.unwrap())
                    .or_default()
                    .push(fi);
            } else {
                non_roof_faces.push(face.clone());
            }
        }

        // Phase 1: Extract all component boundaries without simplification.
        // Each entry: (sem_idx, outer_ring, holes)
        let mut raw_components: Vec<(usize, Vec<u32>, Vec<Vec<u32>>)> = Vec::new();
        let mut fallback_faces: Vec<Face> = Vec::new();

        for (sem_idx, face_indices) in &roof_by_semantic {
            // Build directed half-edge → face index map
            let mut edge_to_face: HashMap<(u32, u32), usize> = HashMap::new();
            for &fi in face_indices {
                let inds = &self.faces[fi].indices;
                let n = inds.len();
                for i in 0..n {
                    let a = inds[i];
                    let b = inds[(i + 1) % n];
                    edge_to_face.insert((a, b), fi);
                }
            }

            // Build face adjacency
            let fi_to_local: HashMap<usize, usize> = face_indices.iter().enumerate().map(|(i, &fi)| (fi, i)).collect();
            let mut adj: Vec<Vec<usize>> = vec![Vec::new(); face_indices.len()];
            for &fi in face_indices {
                let inds = &self.faces[fi].indices;
                let n = inds.len();
                for i in 0..n {
                    let a = inds[i];
                    let b = inds[(i + 1) % n];
                    if let Some(&neighbor_fi) = edge_to_face.get(&(b, a)) {
                        if neighbor_fi != fi {
                            if let (Some(&li), Some(&ln)) = (fi_to_local.get(&fi), fi_to_local.get(&neighbor_fi)) {
                                adj[li].push(ln);
                            }
                        }
                    }
                }
            }

            // BFS connected components
            let mut visited = vec![false; face_indices.len()];
            for start in 0..face_indices.len() {
                if visited[start] { continue; }
                let mut component: Vec<usize> = Vec::new();
                let mut queue = VecDeque::new();
                visited[start] = true;
                queue.push_back(start);
                while let Some(li) = queue.pop_front() {
                    component.push(li);
                    for &nb in &adj[li] {
                        if !visited[nb] { visited[nb] = true; queue.push_back(nb); }
                    }
                }

                let comp_faces: Vec<usize> = component.iter().map(|&li| face_indices[li]).collect();
                if let Some((outer, holes)) = extract_boundary_rings(&self.faces, &self.vertices, &comp_faces) {
                    raw_components.push((*sem_idx, outer, holes));
                } else {
                    for &fi in &comp_faces {
                        fallback_faces.push(self.faces[fi].clone());
                    }
                }
            }
        }

        // Phase 2: Collect all boundary edges to find shared-edge vertices.
        // A vertex is "protected" if it's an endpoint of an edge shared between
        // two different components (i.e., the edge appears in both directions).
        let mut all_boundary_edges: HashMap<(u32, u32), usize> = HashMap::new(); // edge → component index
        for (ci, (_, outer, holes)) in raw_components.iter().enumerate() {
            let rings = std::iter::once(outer.as_slice()).chain(holes.iter().map(|h| h.as_slice()));
            for ring in rings {
                let rn = ring.len();
                for i in 0..rn {
                    let a = ring[i];
                    let b = ring[(i + 1) % rn];
                    all_boundary_edges.insert((a, b), ci);
                }
            }
        }

        let mut protected: HashSet<u32> = HashSet::new();
        for &(a, b) in all_boundary_edges.keys() {
            if let Some(&ci_rev) = all_boundary_edges.get(&(b, a)) {
                if let Some(&ci_fwd) = all_boundary_edges.get(&(a, b)) {
                    if ci_fwd != ci_rev {
                        // This edge is shared between two different components
                        protected.insert(a);
                        protected.insert(b);
                    }
                }
            }
        }

        // Phase 3: Simplify each component's rings, protecting shared-edge vertices.
        let mut merged_faces: Vec<Face> = Vec::new();
        for (sem_idx, outer, holes) in raw_components {
            let outer = simplify_ring_protected(&self.vertices, &outer, 0.05, &protected);
            let mut face = Face::new(outer).with_semantic(sem_idx);
            for hole in holes {
                let hole = simplify_ring_protected(&self.vertices, &hole, 0.05, &protected);
                if hole.len() >= 3 {
                    face.holes.push(hole);
                }
            }
            merged_faces.push(face);
        }

        self.faces = non_roof_faces;
        self.faces.append(&mut merged_faces);
        self.faces.append(&mut fallback_faces);

        // Remove unreferenced vertices after merging
        self.compact_vertices();
    }

    /// Remove vertices not referenced by any face and remap indices.
    pub fn compact_vertices(&mut self) {
        let mut used = vec![false; self.vertices.len()];
        for face in &self.faces {
            for &idx in &face.indices {
                used[idx as usize] = true;
            }
            for hole in &face.holes {
                for &idx in hole {
                    used[idx as usize] = true;
                }
            }
        }

        let mut remap = vec![0u32; self.vertices.len()];
        let mut new_vertices: Vec<Point3<f64>> = Vec::new();
        for (i, &is_used) in used.iter().enumerate() {
            if is_used {
                remap[i] = new_vertices.len() as u32;
                new_vertices.push(self.vertices[i]);
            }
        }

        if new_vertices.len() == self.vertices.len() {
            return; // Nothing to compact
        }

        for face in &mut self.faces {
            for idx in &mut face.indices {
                *idx = remap[*idx as usize];
            }
            for hole in &mut face.holes {
                for idx in hole {
                    *idx = remap[*idx as usize];
                }
            }
        }
        self.vertices = new_vertices;
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract boundary rings from a set of triangular faces.
/// Returns (outer_ring, holes) where outer_ring is the cycle with largest
/// absolute area and holes are the remaining cycles.  Each ring is a list
/// of vertex indices forming a closed polygon (without repeating the first
/// vertex).
fn extract_boundary_rings(faces: &[Face], vertices: &[Point3<f64>], face_indices: &[usize]) -> Option<(Vec<u32>, Vec<Vec<u32>>)> {
    // Collect all directed half-edges
    let mut all_edges: HashSet<(u32, u32)> = HashSet::new();
    for &fi in face_indices {
        let inds = &faces[fi].indices;
        let n = inds.len();
        for i in 0..n {
            all_edges.insert((inds[i], inds[(i + 1) % n]));
        }
    }

    // Boundary edges: those whose reverse is absent
    let mut boundary: HashSet<(u32, u32)> = HashSet::new();
    for &(a, b) in &all_edges {
        if !all_edges.contains(&(b, a)) {
            boundary.insert((a, b));
        }
    }

    if boundary.is_empty() {
        return None;
    }

    // Build outgoing-edge map
    let mut outgoing: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(a, b) in &boundary {
        outgoing.entry(a).or_default().push(b);
    }

    // Walk boundary cycles
    let mut used_edges: HashSet<(u32, u32)> = HashSet::new();
    let mut cycles: Vec<Vec<u32>> = Vec::new();

    for &(start_a, start_b) in &boundary {
        if used_edges.contains(&(start_a, start_b)) {
            continue;
        }

        let mut ring: Vec<u32> = vec![start_a];
        let mut prev = start_a;
        let mut curr = start_b;
        used_edges.insert((start_a, start_b));

        loop {
            ring.push(curr);
            if curr == start_a {
                ring.pop(); // Remove the closing duplicate
                break;
            }

            let candidates = match outgoing.get(&curr) {
                Some(c) => c,
                None => break,
            };

            // Pick next: if only one candidate (excluding where we came from), use it.
            // If multiple, use leftmost-turn heuristic.
            let next = if candidates.len() == 1 {
                candidates[0]
            } else {
                pick_leftmost_turn(vertices, prev, curr, candidates)
            };

            if used_edges.contains(&(curr, next)) {
                break; // Avoid infinite loop
            }
            used_edges.insert((curr, next));
            prev = curr;
            curr = next;
        }

        if ring.len() >= 3 {
            cycles.push(ring);
        }
    }

    if cycles.is_empty() {
        return None;
    }

    if cycles.len() == 1 {
        let outer = cycles.into_iter().next().unwrap();
        return Some((outer, Vec::new()));
    }

    // Multiple cycles: largest absolute area is the outer ring, rest are holes
    let mut best_idx = 0;
    let mut best_area = 0.0_f64;
    for (i, ring) in cycles.iter().enumerate() {
        let area = signed_area_2d(vertices, ring).abs();
        if area > best_area {
            best_area = area;
            best_idx = i;
        }
    }

    let outer = cycles.swap_remove(best_idx);
    let holes = cycles;

    Some((outer, holes))
}

/// Remove collinear vertices from a polygon ring.
/// Uses 2D (XY) projection for collinearity test, since roof plane vertices
/// are collinear in XY when they lie on a straight boundary edge.
/// `tolerance` is the max perpendicular distance (in CRS units) from the line.
fn simplify_ring(vertices: &[Point3<f64>], ring: &[u32], tolerance: f64) -> Vec<u32> {
    simplify_ring_protected(vertices, ring, tolerance, &HashSet::new())
}

/// Like simplify_ring, but vertices in `protected` are never removed.
/// Used to preserve vertices shared between adjacent merged components.
fn simplify_ring_protected(vertices: &[Point3<f64>], ring: &[u32], tolerance: f64, protected: &HashSet<u32>) -> Vec<u32> {
    let n = ring.len();
    if n < 4 {
        return ring.to_vec();
    }
    let mut keep = Vec::with_capacity(n);
    for i in 0..n {
        if protected.contains(&ring[i]) {
            keep.push(ring[i]);
            continue;
        }
        let prev = vertices[ring[(i + n - 1) % n] as usize];
        let curr = vertices[ring[i] as usize];
        let next = vertices[ring[(i + 1) % n] as usize];
        let dx1 = curr.x - prev.x;
        let dy1 = curr.y - prev.y;
        let dx2 = next.x - curr.x;
        let dy2 = next.y - curr.y;
        let cross_2d = (dx1 * dy2 - dy1 * dx2).abs();
        let edge_len = (dx2 * dx2 + dy2 * dy2).sqrt();
        if edge_len < 1e-12 || cross_2d / edge_len > tolerance {
            keep.push(ring[i]);
        }
    }
    if keep.len() < 3 {
        return ring.to_vec();
    }
    keep
}

/// Leftmost-turn heuristic: given incoming edge prev→curr, pick the outgoing
/// edge curr→next that makes the smallest CCW angle (outermost boundary).
fn pick_leftmost_turn(vertices: &[Point3<f64>], prev: u32, curr: u32, candidates: &[u32]) -> u32 {
    let pc = &vertices[curr as usize];
    let pp = &vertices[prev as usize];
    let in_dx = pc.x - pp.x;
    let in_dy = pc.y - pp.y;
    let in_angle = in_dy.atan2(in_dx);

    let mut best = candidates[0];
    let mut best_turn = f64::MAX;

    for &next in candidates {
        let pn = &vertices[next as usize];
        let out_dx = pn.x - pc.x;
        let out_dy = pn.y - pc.y;
        let out_angle = out_dy.atan2(out_dx);

        // Relative turn: how much we turn CCW from incoming to outgoing
        let mut turn = out_angle - in_angle;
        // Normalize to (0, 2π] — we want the smallest positive CCW turn
        if turn <= 0.0 {
            turn += std::f64::consts::TAU;
        }

        if turn < best_turn {
            best_turn = turn;
            best = next;
        }
    }

    best
}

fn signed_area_2d(vertices: &[Point3<f64>], ring: &[u32]) -> f64 {
    let n = ring.len();
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let vi = &vertices[ring[i] as usize];
        let vj = &vertices[ring[j] as usize];
        area += vi.x * vj.y - vj.x * vi.y;
    }
    area * 0.5
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoofReason {
    Reconstructed,
    FlatByAttribute,
    FallbackNoPlanes,
    FallbackCdtPanic,
    ReconstructedMixture,
    SlopedExtrusion,
}

impl RoofReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Reconstructed => "reconstructed",
            Self::FlatByAttribute => "flat_by_attribute",
            Self::FallbackNoPlanes => "fallback_no_planes",
            Self::FallbackCdtPanic => "fallback_cdt_panic",
            Self::ReconstructedMixture => "reconstructed_mixture",
            Self::SlopedExtrusion => "sloped_extrusion",
        }
    }
}

#[derive(Debug, Clone)]
pub struct BuildingGeometry {
    pub id: String,
    pub lod0: Option<Mesh>,
    pub lod12: Option<Mesh>,
    pub lod22: Option<Mesh>,
    pub h_ground: f64,
    pub attributes: AttributeMap,
    pub roof_reason: Option<RoofReason>,
}

impl BuildingGeometry {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            lod0: None,
            lod12: None,
            lod22: None,
            h_ground: 0.0,
            attributes: AttributeMap::new(),
            roof_reason: None,
        }
    }

    pub fn best_lod(&self) -> Option<&Mesh> {
        self.lod22.as_ref().or(self.lod12.as_ref())
    }

    pub fn geographic_extent(&self) -> Option<[f64; 6]> {
        let mesh = self.best_lod()?;
        if mesh.vertices.is_empty() {
            return None;
        }
        let mut min = [f64::MAX; 3];
        let mut max = [f64::MIN; 3];
        for v in &mesh.vertices {
            min[0] = min[0].min(v.x);
            min[1] = min[1].min(v.y);
            min[2] = min[2].min(v.z);
            max[0] = max[0].max(v.x);
            max[1] = max[1].max(v.y);
            max[2] = max[2].max(v.z);
        }
        Some([min[0], min[1], min[2], max[0], max[1], max[2]])
    }
}

#[cfg(test)]
mod merge_tests {
    use super::*;
    use nalgebra::Point3;

    fn p(x: f64, y: f64) -> Point3<f64> {
        Point3::new(x, y, 0.0)
    }

    #[test]
    fn two_adjacent_triangles_merge_into_quad() {
        let mut mesh = Mesh::new();
        let roof_idx = mesh.add_semantic(SemanticSurface::roof());
        let ground_idx = mesh.add_semantic(SemanticSurface::ground());

        // Square: 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1)
        mesh.add_vertex(p(0.0, 0.0));
        mesh.add_vertex(p(1.0, 0.0));
        mesh.add_vertex(p(1.0, 1.0));
        mesh.add_vertex(p(0.0, 1.0));

        // Two triangles sharing edge 0-2 (CCW winding)
        mesh.add_face(Face::new(vec![0, 1, 2]).with_semantic(roof_idx));
        mesh.add_face(Face::new(vec![0, 2, 3]).with_semantic(roof_idx));
        // One ground face (should be untouched)
        mesh.add_face(Face::new(vec![3, 2, 1, 0]).with_semantic(ground_idx));

        mesh.merge_coplanar_roof_faces();

        let roof_faces: Vec<_> = mesh.faces.iter()
            .filter(|f| f.semantic_index == Some(roof_idx))
            .collect();
        let ground_faces: Vec<_> = mesh.faces.iter()
            .filter(|f| f.semantic_index == Some(ground_idx))
            .collect();

        assert_eq!(roof_faces.len(), 1, "two triangles should merge into one polygon");
        assert_eq!(roof_faces[0].indices.len(), 4, "merged polygon should be a quad");
        assert_eq!(ground_faces.len(), 1, "ground face untouched");
    }

    #[test]
    fn disconnected_same_label_stay_separate() {
        let mut mesh = Mesh::new();
        let roof_idx = mesh.add_semantic(SemanticSurface::roof());

        // Triangle 1
        mesh.add_vertex(p(0.0, 0.0));
        mesh.add_vertex(p(1.0, 0.0));
        mesh.add_vertex(p(0.5, 1.0));
        // Triangle 2 (no shared vertices)
        mesh.add_vertex(p(5.0, 5.0));
        mesh.add_vertex(p(6.0, 5.0));
        mesh.add_vertex(p(5.5, 6.0));

        mesh.add_face(Face::new(vec![0, 1, 2]).with_semantic(roof_idx));
        mesh.add_face(Face::new(vec![3, 4, 5]).with_semantic(roof_idx));

        mesh.merge_coplanar_roof_faces();

        let roof_faces: Vec<_> = mesh.faces.iter()
            .filter(|f| f.semantic_index == Some(roof_idx))
            .collect();
        assert_eq!(roof_faces.len(), 2, "disconnected triangles should stay separate");
    }

    #[test]
    fn different_labels_not_merged() {
        let mut mesh = Mesh::new();
        let roof_a = mesh.add_semantic(SemanticSurface::roof());
        let roof_b = mesh.add_semantic(SemanticSurface::roof());

        mesh.add_vertex(p(0.0, 0.0));
        mesh.add_vertex(p(1.0, 0.0));
        mesh.add_vertex(p(1.0, 1.0));
        mesh.add_vertex(p(0.0, 1.0));

        mesh.add_face(Face::new(vec![0, 1, 2]).with_semantic(roof_a));
        mesh.add_face(Face::new(vec![0, 2, 3]).with_semantic(roof_b));

        mesh.merge_coplanar_roof_faces();

        let roof_faces: Vec<_> = mesh.faces.iter()
            .filter(|f| matches!(f.semantic_index, Some(i) if mesh.semantics[i].surface_type == SurfaceType::RoofSurface))
            .collect();
        assert_eq!(roof_faces.len(), 2, "different semantic indices should not merge");
    }

    #[test]
    fn wall_faces_untouched() {
        let mut mesh = Mesh::new();
        let wall_idx = mesh.add_semantic(SemanticSurface::wall(true));

        mesh.add_vertex(p(0.0, 0.0));
        mesh.add_vertex(p(1.0, 0.0));
        mesh.add_vertex(p(1.0, 1.0));
        mesh.add_vertex(p(0.0, 1.0));

        mesh.add_face(Face::new(vec![0, 1, 2]).with_semantic(wall_idx));
        mesh.add_face(Face::new(vec![0, 2, 3]).with_semantic(wall_idx));

        mesh.merge_coplanar_roof_faces();

        assert_eq!(mesh.faces.len(), 2, "wall faces should not be merged");
    }

    #[test]
    fn strip_of_three_triangles() {
        let mut mesh = Mesh::new();
        let roof_idx = mesh.add_semantic(SemanticSurface::roof());

        // Strip: 5 vertices, 3 triangles
        //  3---4
        //  |\ /|
        //  | 2 |
        //  |/ \|
        //  0---1
        mesh.add_vertex(p(0.0, 0.0)); // 0
        mesh.add_vertex(p(2.0, 0.0)); // 1
        mesh.add_vertex(p(1.0, 1.0)); // 2
        mesh.add_vertex(p(0.0, 2.0)); // 3
        mesh.add_vertex(p(2.0, 2.0)); // 4

        mesh.add_face(Face::new(vec![0, 1, 2]).with_semantic(roof_idx));
        mesh.add_face(Face::new(vec![0, 2, 3]).with_semantic(roof_idx));
        mesh.add_face(Face::new(vec![2, 4, 3]).with_semantic(roof_idx));

        mesh.merge_coplanar_roof_faces();

        let roof_faces: Vec<_> = mesh.faces.iter()
            .filter(|f| f.semantic_index == Some(roof_idx))
            .collect();
        assert_eq!(roof_faces.len(), 1, "strip of 3 triangles should merge into one polygon");
        assert!(roof_faces[0].indices.len() >= 4, "merged polygon should have at least 4 boundary vertices");
    }
}
