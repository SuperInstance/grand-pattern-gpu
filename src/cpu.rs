//! CPU serial reference implementations.

/// Graph diffusion: propagate vibe along edges.
///
/// Each room receives weighted contributions from its neighbors.
/// `edges` is a list of `(from, to, weight)` tuples.
pub fn diffuse(rooms: &mut [f64], edges: &[(usize, usize, f64)], rate: f64) {
    let mut deltas = vec![0.0f64; rooms.len()];
    for &(from, to, weight) in edges {
        if from < rooms.len() && to < rooms.len() {
            deltas[from] += weight * (rooms[to] - rooms[from]);
        }
    }
    for (i, room) in rooms.iter_mut().enumerate() {
        *room += rate * deltas[i];
    }
}

/// Weighted average prediction across all rooms.
///
/// Returns a prediction for each room as the weighted average of all other rooms.
/// `weights[i][j]` is the weight room `i` assigns to room `j`'s value.
pub fn jepa_predict(rooms: &[f64], weights: &[Vec<(usize, f64)>]) -> Vec<f64> {
    let n = rooms.len();
    let mut predictions = vec![0.0f64; n];
    for i in 0..n {
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
        predictions[i] = if wtotal > 0.0 { sum / wtotal } else { rooms[i] };
    }
    predictions
}

/// Update prediction weights from errors (JEPA-style learning).
///
/// Adjusts weights to reduce prediction error for next round.
pub fn jepa_learn(
    rooms: &[f64],
    predictions: &[f64],
    weights: &mut [Vec<(usize, f64)>],
    lr: f64,
) {
    for i in 0..rooms.len().min(weights.len()) {
        let error = rooms[i] - predictions[i];
        for (j, w) in weights[i].iter_mut() {
            if *j < rooms.len() {
                // Move weight toward rooms that were unexpectedly high if we underpredicted
                let grad = error * rooms[*j];
                *w += lr * grad;
                // Keep weights positive
                if *w < 0.0 {
                    *w = 0.0;
                }
            }
        }
    }
}

/// Compute surprise for all rooms: |predicted - actual|.
pub fn surprise(actual: &[f64], predicted: &[f64]) -> Vec<f64> {
    actual
        .iter()
        .zip(predicted.iter())
        .map(|(&a, &p)| (a - p).abs())
        .collect()
}

/// Fleet-wide stats: total vibe and total surprise via reduction.
pub fn fleet_stats(rooms: &[f64], surprises: &[f64]) -> (f64, f64) {
    let total_vibe: f64 = rooms.iter().copied().sum();
    let total_surprise: f64 = surprises.iter().copied().sum();
    (total_vibe, total_surprise)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let mut rooms: Vec<f64> = vec![];
        diffuse(&mut rooms, &[], 0.1);
        assert_eq!(rooms.len(), 0);
    }

    #[test]
    fn test_single_room() {
        let mut rooms = vec![1.0];
        diffuse(&mut rooms, &[], 0.1);
        assert_eq!(rooms[0], 1.0);
    }

    #[test]
    fn test_disconnected_graph() {
        let mut rooms = vec![1.0, 2.0, 3.0];
        diffuse(&mut rooms, &[], 0.5);
        assert_eq!(rooms, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_diffuse_basic() {
        let mut rooms = vec![0.0, 1.0];
        // Edge from room 0 to room 1 with weight 1.0
        diffuse(&mut rooms, &[(0, 1, 1.0)], 1.0);
        // delta[0] = 1.0 * (1.0 - 0.0) = 1.0
        // rooms[0] += 1.0 * 1.0 = 1.0
        assert!((rooms[0] - 1.0).abs() < 1e-10);
        assert!((rooms[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_conservation() {
        let mut rooms = vec![1.0, 2.0, 3.0, 4.0];
        let edges = vec![
            (0, 1, 0.5),
            (1, 0, 0.5),
            (1, 2, 0.5),
            (2, 1, 0.5),
            (2, 3, 0.5),
            (3, 2, 0.5),
        ];
        let total_before: f64 = rooms.iter().sum();
        for _ in 0..100 {
            diffuse(&mut rooms, &edges, 0.5);
        }
        let total_after: f64 = rooms.iter().sum();
        assert!((total_before - total_after).abs() < 1e-6, "Total before: {total_before}, after: {total_after}");
    }

    #[test]
    fn test_convergence() {
        let mut rooms = vec![0.0, 1.0, 0.0, 1.0];
        let edges = vec![
            (0, 1, 1.0), (1, 0, 1.0),
            (1, 2, 1.0), (2, 1, 1.0),
            (2, 3, 1.0), (3, 2, 1.0),
        ];
        for _ in 0..1000 {
            diffuse(&mut rooms, &edges, 0.1);
        }
        // Should converge to ~0.5 each
        for r in &rooms {
            assert!((r - 0.5).abs() < 0.01, "Room value: {r}");
        }
    }

    #[test]
    fn test_weighted_edges() {
        let mut rooms = vec![0.0, 10.0];
        // Weight 2.0 should pull harder
        diffuse(&mut rooms, &[(0, 1, 2.0)], 1.0);
        // delta[0] = 2.0 * (10.0 - 0.0) = 20.0
        assert!((rooms[0] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_jepa_predict() {
        let rooms = vec![1.0, 2.0, 3.0];
        let weights = vec![
            vec![(1, 1.0), (2, 1.0)], // room 0 predicts from rooms 1 and 2
            vec![(0, 1.0), (2, 1.0)], // room 1 predicts from rooms 0 and 2
            vec![(0, 1.0), (1, 1.0)], // room 2 predicts from rooms 0 and 1
        ];
        let preds = jepa_predict(&rooms, &weights);
        assert!((preds[0] - 2.5).abs() < 1e-10);
        assert!((preds[1] - 2.0).abs() < 1e-10);
        assert!((preds[2] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_jepa_learn() {
        let rooms = vec![2.0, 4.0];
        let predictions = vec![1.0, 3.0]; // underpredicted both
        let mut weights = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
        jepa_learn(&rooms, &predictions, &mut weights, 0.1);
        // weight[0][0].1 should increase (error * rooms[j] > 0)
        assert!(weights[0][0].1 > 1.0);
    }

    #[test]
    fn test_surprise() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.1, 1.8, 3.5];
        let s = surprise(&actual, &predicted);
        assert!((s[0] - 0.1).abs() < 1e-10);
        assert!((s[1] - 0.2).abs() < 1e-10);
        assert!((s[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fleet_stats() {
        let rooms = vec![1.0, 2.0, 3.0];
        let surprises = vec![0.1, 0.2, 0.3];
        let (vibe, surp) = fleet_stats(&rooms, &surprises);
        assert!((vibe - 6.0).abs() < 1e-10);
        assert!((surp - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_determinism() {
        let mut r1 = vec![1.0, 2.0, 3.0];
        let mut r2 = vec![1.0, 2.0, 3.0];
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        for _ in 0..100 {
            diffuse(&mut r1, &edges, 0.1);
            diffuse(&mut r2, &edges, 0.1);
        }
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_large_edge_count() {
        let n = 1000;
        let mut rooms = vec![1.0; n];
        let mut edges = Vec::new();
        for i in 0..n {
            for j in 0..3 {
                let neighbor = (i + j + 1) % n;
                edges.push((i, neighbor, 0.1));
            }
        }
        diffuse(&mut rooms, &edges, 0.5);
        // Should still be close to 1.0 since all start equal
        for r in &rooms {
            assert!((r - 1.0).abs() < 1e-10);
        }
    }
}
