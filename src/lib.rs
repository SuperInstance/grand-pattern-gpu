//! # grand-pattern-gpu
//!
//! GPU-accelerated graph diffusion for the Grand Pattern.
//!
//! The mono-vibe architecture is trivially parallelizable: each room's diffusion
//! update depends only on neighbor values from the previous tick. This makes it
//! a perfect GPU workload.
//!
//! This crate provides:
//! - **CPU serial** implementations (reference)
//! - **CPU parallel** implementations (std::thread, zero deps)
//! - **Vulkan compute shaders** (GLSL) for GPU execution
//!
//! ## Core Operations
//!
//! | Operation | Description |
//! |-----------|-------------|
//! | `diffuse` | Graph diffusion: propagate vibe along edges |
//! | `jepa_predict` | Weighted average prediction across rooms |
//! | `jepa_learn` | Update prediction weights from errors |
//! | `surprise` | Compute |predicted - actual| per room |
//! | `fleet_stats` | Reduce to fleet-wide vibe + surprise totals |

mod cpu;
mod parallel;
mod shaders;
mod topology;

pub use cpu::*;
pub use parallel::*;
pub use shaders::*;
pub use topology::*;
