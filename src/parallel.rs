//! CPU parallel implementations using std::thread (zero deps).

use std::sync::{Arc, Mutex};
use std::thread;

/// Parallel graph diffusion.
///
/// Splits the delta computation across threads. Each thread computes
/// deltas for a chunk of rooms, then we apply them.
pub fn diffuse_parallel(rooms: &mut [f64], edges: &[(usize, usize, f64)], rate: f64, n_threads: usize) {
    let n = rooms.len();
    if n == 0 {
        return;
    }
    let threads_to_use = n_threads.min(n);

    // Build adjacency: for each room, which edges contribute to it?
    let mut incoming: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
    for &(from, to, weight) in edges {
        if from < n && to < n {
            incoming[from].push((to, weight));
        }
    }

    let incoming = Arc::new(incoming);
    let rooms_snapshot = Arc::new(rooms.to_vec());
    let deltas: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(vec![0.0; n]));

    let chunk_size = (n + threads_to_use - 1) / threads_to_use;
    let mut handles = Vec::new();

    for t in 0..threads_to_use {
        let start = t * chunk_size;
        let end = ((t + 1) * chunk_size).min(n);
        if start >= end {
            continue;
        }
        let inc = Arc::clone(&incoming);
        let snap = Arc::clone(&rooms_snapshot);
        let del = Arc::clone(&deltas);
        handles.push(thread::spawn(move || {
            let mut local_deltas = vec![0.0f64; end - start];
            for (idx, i) in (start..end).enumerate() {
                for &(j, w) in &inc[i] {
                    local_deltas[idx] += w * (snap[j] - snap[i]);
                }
            }
            let mut guard = del.lock().unwrap();
            for (idx, i) in (start..end).enumerate() {
                guard[i] = local_deltas[idx];
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let deltas = Arc::try_unwrap(deltas).unwrap().into_inner().unwrap();
    for (i, room) in rooms.iter_mut().enumerate() {
        *room += rate * deltas[i];
    }
}

/// Parallel JEPA prediction.
pub fn jepa_predict_parallel(rooms: &[f64], weights: &[Vec<(usize, f64)>], n_threads: usize) -> Vec<f64> {
    let n = rooms.len();
    if n == 0 {
        return vec![];
    }
    let threads_to_use = n_threads.min(n);
    let chunk_size = (n + threads_to_use - 1) / threads_to_use;

    let results: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(vec![0.0; n]));
    let mut handles = Vec::new();

    for t in 0..threads_to_use {
        let start = t * chunk_size;
        let end = ((t + 1) * chunk_size).min(n);
        if start >= end {
            continue;
        }
        let rooms = rooms.to_vec();
        let weights: Vec<Vec<(usize, f64)>> = weights.to_vec();
        let res = Arc::clone(&results);
        handles.push(thread::spawn(move || {
            let mut local = vec![0.0f64; end - start];
            for (idx, i) in (start..end).enumerate() {
                let mut sum = 0.0;
                let mut wtotal = 0.0;
                if i < weights.len() {
                    for &(j, w) in &weights[i] {
                        if j < n {
                            sum += w * rooms[j];
                            wtotal += w;
                        }
                    }
                }
                local[idx] = if wtotal > 0.0 { sum / wtotal } else { rooms[i] };
            }
            let mut guard = res.lock().unwrap();
            for (idx, i) in (start..end).enumerate() {
                guard[i] = local[idx];
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    Arc::try_unwrap(results).unwrap().into_inner().unwrap()
}

/// Parallel surprise computation.
pub fn surprise_parallel(actual: &[f64], predicted: &[f64], n_threads: usize) -> Vec<f64> {
    let n = actual.len();
    if n == 0 {
        return vec![];
    }
    let threads_to_use = n_threads.min(n);
    let chunk_size = (n + threads_to_use - 1) / threads_to_use;

    let results: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(vec![0.0; n]));
    let mut handles = Vec::new();

    for t in 0..threads_to_use {
        let start = t * chunk_size;
        let end = ((t + 1) * chunk_size).min(n);
        if start >= end {
            continue;
        }
        let actual = actual.to_vec();
        let predicted = predicted.to_vec();
        let res = Arc::clone(&results);
        handles.push(thread::spawn(move || {
            let mut local = vec![0.0f64; end - start];
            for (idx, i) in (start..end).enumerate() {
                local[idx] = (actual[i] - predicted[i]).abs();
            }
            let mut guard = res.lock().unwrap();
            for (idx, i) in (start..end).enumerate() {
                guard[i] = local[idx];
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    Arc::try_unwrap(results).unwrap().into_inner().unwrap()
}

/// Parallel fleet-wide stats via reduction.
pub fn fleet_reduce(rooms: &[f64], surprises: &[f64], n_threads: usize) -> (f64, f64) {
    let n = rooms.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let threads_to_use = n_threads.min(n);
    let chunk_size = (n + threads_to_use - 1) / threads_to_use;

    let partials: Arc<Mutex<Vec<(f64, f64)>>> = Arc::new(Mutex::new(vec![(0.0, 0.0); threads_to_use]));
    let mut handles = Vec::new();

    for t in 0..threads_to_use {
        let start = t * chunk_size;
        let end = ((t + 1) * chunk_size).min(n);
        if start >= end {
            continue;
        }
        let rooms = rooms.to_vec();
        let surprises = surprises.to_vec();
        let p = Arc::clone(&partials);
        handles.push(thread::spawn(move || {
            let mut vibe = 0.0f64;
            let mut surp = 0.0f64;
            for i in start..end {
                vibe += rooms[i];
                surp += surprises[i];
            }
            let mut guard = p.lock().unwrap();
            guard[t] = (vibe, surp);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let partials = Arc::try_unwrap(partials).unwrap().into_inner().unwrap();
    let total_vibe: f64 = partials.iter().map(|(v, _)| v).sum();
    let total_surprise: f64 = partials.iter().map(|(_, s)| s).sum();
    (total_vibe, total_surprise)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu;

    #[test]
    fn test_parallel_diffuse_matches_serial() {
        let mut serial = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut parallel = serial.clone();
        let edges = vec![
            (0, 1, 1.0), (1, 0, 1.0),
            (1, 2, 1.0), (2, 1, 1.0),
            (2, 3, 1.0), (3, 2, 1.0),
            (3, 4, 1.0), (4, 3, 1.0),
        ];
        for _ in 0..50 {
            cpu::diffuse(&mut serial, &edges, 0.2);
            diffuse_parallel(&mut parallel, &edges, 0.2, 4);
        }
        for i in 0..serial.len() {
            assert!((serial[i] - parallel[i]).abs() < 1e-10,
                "Mismatch at {i}: serial={}, parallel={}", serial[i], parallel[i]);
        }
    }

    #[test]
    fn test_parallel_conservation() {
        let mut rooms = vec![1.0, 2.0, 3.0, 4.0];
        let edges = vec![
            (0, 1, 0.5), (1, 0, 0.5),
            (1, 2, 0.5), (2, 1, 0.5),
            (2, 3, 0.5), (3, 2, 0.5),
        ];
        let total_before: f64 = rooms.iter().sum();
        for _ in 0..100 {
            diffuse_parallel(&mut rooms, &edges, 0.5, 2);
        }
        let total_after: f64 = rooms.iter().sum();
        assert!((total_before - total_after).abs() < 1e-6);
    }

    #[test]
    fn test_parallel_surprise_matches_serial() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![1.1, 1.9, 3.2, 3.8, 5.3];
        let serial = cpu::surprise(&actual, &predicted);
        let parallel = surprise_parallel(&actual, &predicted, 4);
        assert_eq!(serial, parallel);
    }

    #[test]
    fn test_parallel_fleet_stats() {
        let rooms = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let surprises = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let (v, s) = fleet_reduce(&rooms, &surprises, 4);
        assert!((v - 15.0).abs() < 1e-10);
        assert!((s - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_predict() {
        let rooms = vec![1.0, 2.0, 3.0];
        let weights = vec![
            vec![(1, 1.0), (2, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 1.0), (1, 1.0)],
        ];
        let serial = cpu::jepa_predict(&rooms, &weights);
        let parallel = jepa_predict_parallel(&rooms, &weights, 3);
        for i in 0..serial.len() {
            assert!((serial[i] - parallel[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_parallel_empty() {
        let mut rooms: Vec<f64> = vec![];
        diffuse_parallel(&mut rooms, &[], 0.1, 4);
        assert!(rooms.is_empty());
    }

    #[test]
    fn test_1m_rooms() {
        let n = 1_000_000;
        let mut rooms = vec![1.0; n];
        // Ring graph
        let mut edges = Vec::new();
        for i in 0..n {
            edges.push((i, (i + 1) % n, 0.01));
            edges.push((i, (i + n - 1) % n, 0.01));
        }
        // Just ensure it completes without panic
        diffuse_parallel(&mut rooms, &edges, 0.1, 8);
        // All rooms should still be very close to 1.0 (started uniform, only 1 step)
        for r in &rooms {
            assert!((r - 1.0).abs() < 1e-6);
        }
    }
}
