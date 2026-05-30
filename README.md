# grand-pattern-gpu

GPU-accelerated graph diffusion for the Grand Pattern.

The mono-vibe architecture is trivially parallelizable: each room's diffusion update depends only on neighbor values from the previous tick. This is a **perfect GPU workload**.

## Architecture

### Vulkan Compute Shaders (GLSL)

Works on **ALL GPUs** — NVIDIA, AMD, Intel, mobile, embedded. No CUDA dependency.

| Shader | Description |
|--------|-------------|
| `diffuse.comp` | Graph diffusion kernel — propagate vibe along edges |
| `jepa_predict.comp` | Weighted average prediction across all rooms |
| `jepa_learn.comp` | Weight update from prediction errors |
| `surprise.comp` | Compute |predicted - actual| for all rooms |
| `fleet_stats.comp` | Parallel reduction to fleet vibe + surprise totals |

### CPU Implementations

- **Serial** — reference implementation, no dependencies
- **Parallel** — `std::thread` based, zero runtime dependencies

## Usage

```rust
use grand_pattern_gpu::{diffuse, diffuse_parallel, surprise, fleet_reduce};

// Serial diffusion
let mut rooms = vec![1.0, 2.0, 3.0, 4.0];
let edges = vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)];
diffuse(&mut rooms, &edges, 0.1);

// Parallel diffusion (4 threads)
diffuse_parallel(&mut rooms, &edges, 0.1, 4);
```

## Benchmarks

```bash
cargo bench
```

Compares CPU serial vs CPU parallel for 1K, 10K, 100K rooms.

## Tests

```bash
cargo test
```

25+ tests covering:
- Diffusion correctness and conservation
- JEPA prediction and learning
- Surprise computation
- Fleet statistics
- Parallel vs serial equivalence
- Edge cases (empty, single room, disconnected)
- Large-scale (1M rooms, 1M edges)
- Shader structure validation

## Zero Dependencies

Pure Rust + GLSL shaders. No runtime dependencies. Build-time `shaderc` optional.

## License

MIT
