//! Benchmark harness: CPU serial vs CPU parallel for various graph sizes.
//!
//! Run with: `cargo bench`

use std::time::Instant;

fn main() {
    let thread_counts = [1, 2, 4, 8];
    let sizes = [1_000usize, 10_000, 100_000];
    let iterations = 10;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          Grand Pattern GPU — Benchmark Report                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    for &n in &sizes {
        println!("┌─────────────────────────────────────────────┐");
        println!("│ Graph: {n:>6} rooms (ring, 2 edges each)       │");
        println!("├─────────────────────────────────────────────┤");

        // Build ring graph
        let mut edges = Vec::new();
        for i in 0..n {
            edges.push((i, (i + 1) % n, 0.1));
            edges.push((i, (i + n - 1) % n, 0.1));
        }

        let mut rooms_serial = vec![1.0f64; n];
        rooms_serial[0] = 2.0;

        // Serial baseline
        let start = Instant::now();
        for _ in 0..iterations {
            grand_pattern_gpu::diffuse(&mut rooms_serial, &edges, 0.1);
        }
        let serial_time = start.elapsed();
        println!("│ Serial:   {:>8.2} ms ({iterations} iterations)    │", serial_time.as_secs_f64() * 1000.0);

        let mut rooms_parallel = vec![1.0f64; n];
        rooms_parallel[0] = 2.0;

        // Parallel variants
        for &threads in &thread_counts {
            let mut rooms = vec![1.0f64; n];
            rooms[0] = 2.0;

            let start = Instant::now();
            for _ in 0..iterations {
                grand_pattern_gpu::diffuse_parallel(&mut rooms, &edges, 0.1, threads);
            }
            let parallel_time = start.elapsed();

            let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();
            println!("│ Parallel ({threads}T): {:>7.2} ms  →  {speedup:>5.2}x speedup    │",
                parallel_time.as_secs_f64() * 1000.0);

            // Keep last parallel run for comparison
            if threads == thread_counts[thread_counts.len() - 1] {
                rooms_parallel = rooms;
            }
        }

        // Verify results match
        let max_diff = rooms_serial.iter()
            .zip(rooms_parallel.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        println!("│ Max diff serial vs parallel: {max_diff:.2e}          │");

        println!("└─────────────────────────────────────────────┘");
        println!();
    }

    // Surprise benchmark
    println!("┌─────────────────────────────────────────────┐");
    println!("│ Surprise computation benchmark               │");
    println!("├─────────────────────────────────────────────┤");

    for &n in &sizes {
        let actual: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let predicted: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();

        let start = Instant::now();
        let _serial = grand_pattern_gpu::surprise(&actual, &predicted);
        let serial_time = start.elapsed();

        let start = Instant::now();
        let _parallel = grand_pattern_gpu::surprise_parallel(&actual, &predicted, 4);
        let parallel_time = start.elapsed();

        let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();
        println!("│ {n:>6} rooms: serial {:>7.2}ms, parallel {:>7.2}ms ({speedup:.2}x) │",
            serial_time.as_secs_f64() * 1000.0, parallel_time.as_secs_f64() * 1000.0);
    }

    println!("└─────────────────────────────────────────────┘");
    println!();

    // Fleet stats benchmark
    println!("┌─────────────────────────────────────────────┐");
    println!("│ Fleet reduction benchmark                    │");
    println!("├─────────────────────────────────────────────┤");

    for &n in &sizes {
        let rooms: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let surprises: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();

        let start = Instant::now();
        let _serial = grand_pattern_gpu::fleet_stats(&rooms, &surprises);
        let serial_time = start.elapsed();

        let start = Instant::now();
        let _parallel = grand_pattern_gpu::fleet_reduce(&rooms, &surprises, 4);
        let parallel_time = start.elapsed();

        let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();
        println!("│ {n:>6} rooms: serial {:>7.2}ms, parallel {:>7.2}ms ({speedup:.2}x) │",
            serial_time.as_secs_f64() * 1000.0, parallel_time.as_secs_f64() * 1000.0);
    }

    println!("└─────────────────────────────────────────────┘");
    println!();

    // Shader validation
    println!("┌─────────────────────────────────────────────┐");
    println!("│ Shader validation                            │");
    println!("├─────────────────────────────────────────────┤");
    let results = grand_pattern_gpu::validate_all_shaders();
    for (i, result) in results.iter().enumerate() {
        let names = ["diffuse", "jepa_predict", "jepa_learn", "surprise", "fleet_stats"];
        match result {
            Ok(()) => println!("│ {} ... ✓ valid                            │", names[i]),
            Err(e) => println!("│ {} ... ✗ {e:<30}│", names[i]),
        }
    }
    if grand_pattern_gpu::glslang_available() {
        println!("│ (glslangValidator available)                 │");
    } else {
        println!("│ (glslangValidator not found — soft pass)     │");
    }
    println!("└─────────────────────────────────────────────┘");
}
