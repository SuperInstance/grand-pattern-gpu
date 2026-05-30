//! Parallel topology construction.
//!
//! Builds adjacency structures for the graph from edge lists,
//! preparing data for both CPU and GPU (Vulkan) execution paths.

use std::sync::{Arc, Mutex};
use std::thread;

/// Adjacency list representation for the graph.
#[derive(Debug, Clone)]
pub struct GraphTopology {
    /// Number of rooms (nodes).
    pub n_rooms: usize,
    /// For each room, list of (neighbor, weight) pairs.
    pub adjacency: Vec<Vec<(usize, f64)>>,
    /// Flat edge data for GPU upload: (from, to, weight).
    pub edges: Vec<(usize, usize, f64)>,
}

impl GraphTopology {
    /// Build from an edge list.
    pub fn from_edges(n_rooms: usize, edges: &[(usize, usize, f64)]) -> Self {
        let mut adjacency = vec![vec![]; n_rooms];
        for &(from, to, weight) in edges {
            if from < n_rooms && to < n_rooms {
                adjacency[from].push((to, weight));
            }
        }
        Self {
            n_rooms,
            adjacency,
            edges: edges.to_vec(),
        }
    }

    /// Build from edges using parallel construction.
    pub fn from_edges_parallel(n_rooms: usize, edges: &[(usize, usize, f64)], n_threads: usize) -> Self {
        if n_rooms == 0 {
            return Self {
                n_rooms: 0,
                adjacency: vec![],
                edges: edges.to_vec(),
            };
        }

        let threads_to_use = n_threads.min(n_rooms);
        let chunk_size = (n_rooms + threads_to_use - 1) / threads_to_use;

        // Pre-sort edges by target room for parallel processing
        let adj: Arc<Mutex<Vec<Vec<(usize, f64)>>>> = Arc::new(Mutex::new(vec![vec![]; n_rooms]));
        let mut handles = Vec::new();

        for t in 0..threads_to_use {
            let start = t * chunk_size;
            let end = ((t + 1) * chunk_size).min(n_rooms);
            if start >= end {
                continue;
            }
            let edges = edges.to_vec();
            let adj = Arc::clone(&adj);
            handles.push(thread::spawn(move || {
                let mut local = vec![vec![]; end - start];
                for &(from, to, weight) in &edges {
                    if from >= start && from < end && to < n_rooms {
                        local[from - start].push((to, weight));
                    }
                }
                let mut guard = adj.lock().unwrap();
                for (idx, i) in (start..end).enumerate() {
                    guard[i] = local[idx].clone();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let adjacency = Arc::try_unwrap(adj).unwrap().into_inner().unwrap();
        Self {
            n_rooms,
            adjacency,
            edges: edges.to_vec(),
        }
    }

    /// Generate GPU-friendly flat buffers for Vulkan upload.
    /// Returns (offsets, counts, neighbors, weights) as flat arrays.
    pub fn gpu_buffers(&self) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<f32>) {
        let mut offsets = Vec::with_capacity(self.n_rooms);
        let mut counts = Vec::with_capacity(self.n_rooms);
        let mut neighbors = Vec::new();
        let mut weights = Vec::new();

        let mut offset = 0u32;
        for adj in &self.adjacency {
            offsets.push(offset);
            counts.push(adj.len() as u32);
            for &(neighbor, weight) in adj {
                neighbors.push(neighbor as u32);
                weights.push(weight as f32);
            }
            offset += adj.len() as u32;
        }

        (offsets, counts, neighbors, weights)
    }

    /// Generate a ring graph topology (each room connected to neighbors).
    pub fn ring(n_rooms: usize, weight: f64) -> Self {
        let mut edges = Vec::new();
        for i in 0..n_rooms {
            let next = (i + 1) % n_rooms;
            let prev = (i + n_rooms - 1) % n_rooms;
            edges.push((i, next, weight));
            edges.push((i, prev, weight));
        }
        Self::from_edges(n_rooms, &edges)
    }

    /// Generate a fully connected graph topology.
    pub fn fully_connected(n_rooms: usize, weight: f64) -> Self {
        let mut edges = Vec::new();
        for i in 0..n_rooms {
            for j in 0..n_rooms {
                if i != j {
                    edges.push((i, j, weight));
                }
            }
        }
        Self::from_edges(n_rooms, &edges)
    }

    /// Generate a random graph topology.
    pub fn random(n_rooms: usize, edge_probability: f64, weight: f64, seed: u64) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut rng_state = seed;
        let mut next_rand = || -> u64 {
            let mut hasher = DefaultHasher::new();
            rng_state.hash(&mut hasher);
            rng_state = hasher.finish();
            rng_state
        };

        let mut edges = Vec::new();
        for i in 0..n_rooms {
            for j in 0..n_rooms {
                if i != j {
                    let r = (next_rand() % 10000) as f64 / 10000.0;
                    if r < edge_probability {
                        edges.push((i, j, weight));
                    }
                }
            }
        }
        Self::from_edges(n_rooms, &edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_from_edges() {
        let edges = vec![(0, 1, 1.0), (1, 0, 1.0), (1, 2, 0.5)];
        let topo = GraphTopology::from_edges(3, &edges);
        assert_eq!(topo.adjacency[0], vec![(1, 1.0)]);
        assert_eq!(topo.adjacency[1], vec![(0, 1.0), (2, 0.5)]);
        assert_eq!(topo.adjacency[2].len(), 0);
    }

    #[test]
    fn test_topology_parallel_matches_serial() {
        let edges = vec![
            (0, 1, 1.0), (1, 0, 1.0),
            (1, 2, 1.0), (2, 1, 1.0),
            (2, 3, 1.0), (3, 2, 1.0),
            (3, 4, 1.0), (4, 3, 1.0),
        ];
        let serial = GraphTopology::from_edges(5, &edges);
        let parallel = GraphTopology::from_edges_parallel(5, &edges, 4);
        assert_eq!(serial.adjacency, parallel.adjacency);
    }

    #[test]
    fn test_ring_topology() {
        let topo = GraphTopology::ring(4, 1.0);
        // Room 0 should connect to 1 and 3
        assert!(topo.adjacency[0].contains(&(1, 1.0)));
        assert!(topo.adjacency[0].contains(&(3, 1.0)));
    }

    #[test]
    fn test_gpu_buffers() {
        let topo = GraphTopology::ring(3, 0.5);
        let (offsets, counts, neighbors, weights) = topo.gpu_buffers();
        assert_eq!(offsets.len(), 3);
        assert_eq!(counts.len(), 3);
        // Each room has 2 edges
        assert!(counts.iter().all(|&c| c == 2));
        assert_eq!(neighbors.len(), 6);
        assert!(weights.iter().all(|&w| (w - 0.5f32).abs() < 1e-6));
    }

    #[test]
    fn test_fully_connected() {
        let topo = GraphTopology::fully_connected(4, 1.0);
        // Each room connects to 3 others
        for adj in &topo.adjacency {
            assert_eq!(adj.len(), 3);
        }
    }
}
