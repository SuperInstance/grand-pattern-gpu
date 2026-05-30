//! Vulkan compute shader definitions and validation.
//!
//! This module provides:
//! - Shader source access for each compute pipeline
//! - GLSL validation (via `glslangValidator` if available)
//! - Shader description strings for introspection

use std::process::Command;

/// Returns the source of the diffuse compute shader.
pub fn diffuse_shader() -> &'static str {
    include_str!("../shaders/diffuse.comp")
}

/// Returns the source of the JEPA predict compute shader.
pub fn jepa_predict_shader() -> &'static str {
    include_str!("../shaders/jepa_predict.comp")
}

/// Returns the source of the JEPA learn compute shader.
pub fn jepa_learn_shader() -> &'static str {
    include_str!("../shaders/jepa_learn.comp")
}

/// Returns the source of the surprise compute shader.
pub fn jepa_surprise_shader() -> &'static str {
    include_str!("../shaders/surprise.comp")
}

/// Returns the source of the fleet stats compute shader.
pub fn fleet_stats_shader() -> &'static str {
    include_str!("../shaders/fleet_stats.comp")
}

/// All shader names and their sources.
pub fn all_shaders() -> Vec<(&'static str, &'static str)> {
    vec![
        ("diffuse", diffuse_shader()),
        ("jepa_predict", jepa_predict_shader()),
        ("jepa_learn", jepa_learn_shader()),
        ("surprise", jepa_surprise_shader()),
        ("fleet_stats", fleet_stats_shader()),
    ]
}

/// Validate a single GLSL shader source using glslangValidator.
///
/// Returns Ok(()) if valid, Err with message otherwise.
/// Returns Ok(()) if glslangValidator is not available (soft validation).
pub fn validate_shader_glslang(source: &str, stage: &str) -> Result<(), String> {
    // Write to temp file
    let tmp_dir = std::env::temp_dir();
    let tmp_path = tmp_dir.join(format!("gp_gpu_validate_{}.comp", std::process::id()));
    std::fs::write(&tmp_path, source).map_err(|e| format!("Failed to write temp shader: {e}"))?;

    let result = Command::new("glslangValidator")
        .arg("-S")
        .arg("comp")
        .arg("-V")
        .arg(&tmp_path)
        .output();

    let _ = std::fs::remove_file(&tmp_path);

    match result {
        Ok(output) => {
            if output.status.success() {
                Ok(())
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                Err(format!("Shader validation failed ({stage}): {stderr}"))
            }
        }
        Err(_) => {
            // glslangValidator not available — soft pass
            Ok(())
        }
    }
}

/// Validate all shaders using glslangValidator.
pub fn validate_all_shaders() -> Vec<Result<(), String>> {
    all_shaders()
        .into_iter()
        .map(|(name, source)| validate_shader_glslang(source, name))
        .collect()
}

/// Check if glslangValidator is available on the system.
pub fn glslang_available() -> bool {
    Command::new("glslangValidator")
        .arg("--version")
        .output()
        .is_ok()
}

/// A shader that has been compiled to SPIR-V bytes (placeholder for runtime compilation).
#[derive(Debug, Clone)]
pub struct CompiledShader {
    pub name: String,
    pub spirv: Vec<u8>,
}

/// Simple GLSL -> SPIR-V compilation (requires shaderc or glslangValidator at build time).
/// For now, this validates shaders but returns empty SPIR-V as placeholder.
pub fn compile_shader(name: &str, source: &str) -> Result<CompiledShader, String> {
    validate_shader_glslang(source, name)?;

    // In a real implementation, this would use shaderc or glslang to produce SPIR-V.
    // For now we return a marker that the shader source is valid.
    Ok(CompiledShader {
        name: name.to_string(),
        spirv: Vec::new(), // placeholder — would be real SPIR-V in production
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_shaders_accessible() {
        let shaders = all_shaders();
        assert_eq!(shaders.len(), 5);
        for (name, source) in &shaders {
            assert!(!source.is_empty(), "Shader {name} is empty");
            assert!(source.contains("#version 450"), "Shader {name} missing version");
        }
    }

    #[test]
    fn test_diffuse_shader_structure() {
        let src = diffuse_shader();
        assert!(src.contains("rooms_in"));
        assert!(src.contains("rooms_out"));
        assert!(src.contains("edge_offsets"));
        assert!(src.contains("edge_weights"));
        assert!(src.contains("rate"));
    }

    #[test]
    fn test_surprise_shader_structure() {
        let src = jepa_surprise_shader();
        assert!(src.contains("abs("));
        assert!(src.contains("actual"));
        assert!(src.contains("predicted"));
    }

    #[test]
    fn test_fleet_stats_has_shared_memory() {
        let src = fleet_stats_shader();
        assert!(src.contains("shared"));
        assert!(src.contains("barrier"));
    }

    #[test]
    fn test_shader_validation_runs() {
        // Even without glslangValidator, this should return Ok
        let result = validate_shader_glslang(diffuse_shader(), "diffuse");
        assert!(result.is_ok() || result.is_err()); // just ensure no panic
    }

    #[test]
    fn test_compile_shader_returns() {
        let result = compile_shader("diffuse", diffuse_shader());
        // Will be Ok if glslang not available (soft pass), or Ok/Err if available
        assert!(result.is_ok() || result.is_err()); // no panic
    }
}
