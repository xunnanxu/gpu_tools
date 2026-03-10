pub mod util;

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

/// A single event from the Chrome trace format.
#[derive(Debug, Deserialize)]
pub struct TraceEvent {
    pub ph: Option<String>,
    pub cat: Option<String>,
    pub name: Option<String>,
    #[allow(dead_code)]
    pub ts: Option<f64>,
    pub dur: Option<f64>,
    #[allow(dead_code)]
    pub pid: Option<serde_json::Value>,
    #[allow(dead_code)]
    pub tid: Option<serde_json::Value>,
    pub args: Option<serde_json::Value>,
}

/// Top-level Chrome trace format.
#[derive(Debug, Deserialize)]
pub struct ChromeTrace {
    #[serde(rename = "traceEvents")]
    pub trace_events: Vec<TraceEvent>,
}

/// Unique identifier for a GPU kernel invocation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KernelId {
    pub name: String,
    pub grid: String,
    pub input_shapes: String,
}

impl std::fmt::Display for KernelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} | {} | {}", self.name, self.grid, self.input_shapes)
    }
}

/// Parse a chrome trace file and extract GPU kernel events.
///
/// Returns a map from `KernelId` to a list of execution durations (ms).
///
/// Input parameter shapes live on CPU-side operator events and are resolved
/// via the ac2g correlation chain:
///   GPU kernel `args.correlation` → cuda_runtime `args.correlation`
///   → cuda_runtime `args.External id` → cpu_op `args.External id` → `Input Dims`
pub fn parse_trace(path: &Path) -> Result<BTreeMap<KernelId, Vec<f64>>> {
    let content = util::read_maybe_gzipped(path)?;
    let trace: ChromeTrace = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse JSON from {}", path.display()))?;

    // Step 1: Build map from correlation id → CPU External id via cuda_runtime events.
    // cuda_runtime events carry both `correlation` (matching ac2g/kernel) and
    // `External id` (matching the launching cpu_op).
    let mut correlation_to_ext_id: HashMap<u64, u64> = HashMap::new();
    for event in &trace.trace_events {
        let cat = event.cat.as_deref().unwrap_or("");
        if cat != "cuda_runtime" && cat != "cuda_driver" {
            continue;
        }
        if let Some(args) = &event.args
            && let Some(correlation) = args.get("correlation").and_then(|v| v.as_u64())
            && let Some(ext_id) = args.get("External id").and_then(|v| v.as_u64())
        {
            correlation_to_ext_id.insert(correlation, ext_id);
        }
    }

    // Step 2: Build map from External id → Input Dims string from cpu_op events.
    let mut cpu_input_shapes: HashMap<u64, String> = HashMap::new();
    for event in &trace.trace_events {
        let ph = event.ph.as_deref().unwrap_or("");
        if ph != "X" && ph != "B" {
            continue;
        }
        if let Some(args) = &event.args
            && let Some(ext_id) = args.get("External id").and_then(|v| v.as_u64())
        {
            let dims = args.get("Input Dims").or_else(|| args.get("Input dims"));
            if let Some(dims) = dims {
                let formatted = format_input_dims(dims);
                if !formatted.is_empty() {
                    cpu_input_shapes.insert(ext_id, formatted);
                }
            }
        }
    }

    // Step 3: Collect GPU kernel events and resolve input shapes via correlation chain.
    let mut kernels: BTreeMap<KernelId, Vec<f64>> = BTreeMap::new();
    for event in &trace.trace_events {
        let cat = event.cat.as_deref().unwrap_or("");
        let ph = event.ph.as_deref().unwrap_or("");
        if cat != "kernel" || ph != "X" {
            continue;
        }

        let name = event.name.as_deref().unwrap_or("unknown").to_string();
        let dur_ms = event.dur.unwrap_or(0.0) / 1000.0;

        let args = event.args.as_ref();
        let grid = args
            .and_then(|a| a.get("grid"))
            .map(format_value)
            .unwrap_or_default();

        // Follow correlation chain: kernel.correlation → cuda_runtime.External id → cpu_op.Input Dims
        let correlation = args
            .and_then(|a| a.get("correlation"))
            .and_then(|v| v.as_u64());
        let cpu_ext_id = correlation.and_then(|c| correlation_to_ext_id.get(&c));
        let input_shapes = cpu_ext_id
            .and_then(|id| cpu_input_shapes.get(id))
            .cloned()
            .unwrap_or_default();

        let kernel_id = KernelId {
            name,
            grid,
            input_shapes,
        };
        kernels.entry(kernel_id).or_default().push(dur_ms);
    }

    Ok(kernels)
}

/// Generate a markdown comparison table across multiple traces.
pub fn generate_comparison_table(traces: &[(String, BTreeMap<KernelId, Vec<f64>>)]) -> String {
    let all_kernel_ids: BTreeSet<&KernelId> = traces
        .iter()
        .flat_map(|(_, kernels)| kernels.keys())
        .collect();

    if all_kernel_ids.is_empty() {
        return "No GPU kernel events found in any trace.\n".to_string();
    }

    let trace_names: Vec<&str> = traces.iter().map(|(name, _)| name.as_str()).collect();
    let mut lines = Vec::new();

    // Header
    let mut header = "| Kernel | Grid | Input Shapes |".to_string();
    for name in &trace_names {
        header.push_str(&format!(" {} (ms) |", name));
    }
    lines.push(header);

    // Separator
    let mut sep = "| --- | --- | --- |".to_string();
    for _ in &trace_names {
        sep.push_str(" ---: |");
    }
    lines.push(sep);

    // Data rows
    for kid in &all_kernel_ids {
        let mut row = format!(
            "| {} | {} | {} |",
            escape_md(&kid.name),
            escape_md(&kid.grid),
            escape_md(&kid.input_shapes)
        );
        for (_, kernels) in traces {
            if let Some(durations) = kernels.get(*kid) {
                let total: f64 = durations.iter().sum();
                row.push_str(&format!(" {:.3} |", total));
            } else {
                row.push_str(" |");
            }
        }
        lines.push(row);
    }

    lines.join("\n") + "\n"
}

/// Escape pipe characters in markdown table cells.
fn escape_md(s: &str) -> String {
    s.replace('|', "\\|")
}

/// Format Input Dims, converting empty sub-arrays (scalars) to `1`.
/// e.g. `[[1024, 128], [], [128, 256], []]` → `[[1024, 128], 1, [128, 256], 1]`
fn format_input_dims(v: &serde_json::Value) -> String {
    if let serde_json::Value::Array(arr) = v {
        let parts: Vec<String> = arr
            .iter()
            .map(|item| match item {
                serde_json::Value::Array(inner) if inner.is_empty() => "1".to_string(),
                other => format_value(other),
            })
            .collect();
        format!("[{}]", parts.join(", "))
    } else {
        format_value(v)
    }
}

/// Format a serde_json::Value into a readable string (for grids, shapes, etc.).
fn format_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Array(arr) => {
            let parts: Vec<String> = arr.iter().map(format_value).collect();
            format!("[{}]", parts.join(", "))
        }
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_value_array() {
        let v = serde_json::json!([384, 1, 1]);
        assert_eq!(format_value(&v), "[384, 1, 1]");
    }

    #[test]
    fn test_format_value_nested_array() {
        let v = serde_json::json!([[38400, 128], [128, 256]]);
        assert_eq!(format_value(&v), "[[38400, 128], [128, 256]]");
    }

    #[test]
    fn test_format_input_dims_scalars_become_1() {
        let v = serde_json::json!([[1024, 128], [], [128, 256], []]);
        assert_eq!(format_input_dims(&v), "[[1024, 128], 1, [128, 256], 1]");
    }

    #[test]
    fn test_format_input_dims_all_scalars() {
        let v = serde_json::json!([[], [], []]);
        assert_eq!(format_input_dims(&v), "[1, 1, 1]");
    }

    #[test]
    fn test_kernel_id_display() {
        let kid = KernelId {
            name: "my_kernel".to_string(),
            grid: "[384, 1, 1]".to_string(),
            input_shapes: "[38400, 128, 1]".to_string(),
        };
        assert_eq!(kid.to_string(), "my_kernel | [384, 1, 1] | [38400, 128, 1]");
    }

    #[test]
    fn test_parse_trace_correlation_chain() {
        // Tests the full correlation chain:
        // kernel.correlation(100) → cuda_runtime.correlation(100) + External id(27)
        //   → cpu_op.External id(27) → Input Dims
        let trace_json = serde_json::json!({
            "traceEvents": [
                {
                    "ph": "X", "cat": "kernel", "name": "my_kernel",
                    "ts": 1000, "dur": 500, "pid": 0, "tid": 7,
                    "args": { "External id": 33, "correlation": 100, "grid": [384, 1, 1] }
                },
                {
                    "ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
                    "ts": 900, "dur": 50, "pid": 0, "tid": 1,
                    "args": { "External id": 27, "correlation": 100 }
                },
                {
                    "ph": "X", "cat": "cpu_op", "name": "aten::mm",
                    "ts": 800, "dur": 200, "pid": 0, "tid": 1,
                    "args": { "External id": 27, "Input Dims": [[38400, 128], [128, 256]] }
                }
            ]
        });

        let tmp = std::env::temp_dir().join("trace_test_correlation.json");
        std::fs::write(&tmp, serde_json::to_string(&trace_json).unwrap()).unwrap();

        let result = parse_trace(&tmp).unwrap();
        assert_eq!(result.len(), 1);

        let (kid, durations) = result.iter().next().unwrap();
        assert_eq!(kid.name, "my_kernel");
        assert_eq!(kid.grid, "[384, 1, 1]");
        assert_eq!(kid.input_shapes, "[[38400, 128], [128, 256]]");
        assert_eq!(durations.len(), 1);
        assert!((durations[0] - 0.5).abs() < 0.001);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_parse_trace_scalar_dims_become_1() {
        // Empty arrays in Input Dims represent scalars and should become `1`.
        let trace_json = serde_json::json!({
            "traceEvents": [
                {
                    "ph": "X", "cat": "kernel", "name": "scale_kernel",
                    "ts": 1000, "dur": 200, "pid": 0, "tid": 7,
                    "args": { "External id": 33, "correlation": 100, "grid": [1, 1, 1] }
                },
                {
                    "ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
                    "ts": 900, "dur": 50, "pid": 0, "tid": 1,
                    "args": { "External id": 27, "correlation": 100 }
                },
                {
                    "ph": "X", "cat": "cpu_op", "name": "aten::mul",
                    "ts": 800, "dur": 200, "pid": 0, "tid": 1,
                    "args": { "External id": 27, "Input Dims": [[1024, 128], []] }
                }
            ]
        });

        let tmp = std::env::temp_dir().join("trace_test_scalar_dims.json");
        std::fs::write(&tmp, serde_json::to_string(&trace_json).unwrap()).unwrap();

        let result = parse_trace(&tmp).unwrap();
        let (kid, _) = result.iter().next().unwrap();
        assert_eq!(kid.input_shapes, "[[1024, 128], 1]");

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_parse_trace_no_correlation_leaves_shapes_empty() {
        let trace_json = serde_json::json!({
            "traceEvents": [
                {
                    "ph": "X", "cat": "kernel", "name": "orphan_kernel",
                    "ts": 1000, "dur": 200, "pid": 0, "tid": 7,
                    "args": { "External id": 99, "grid": [1, 1, 1] }
                }
            ]
        });

        let tmp = std::env::temp_dir().join("trace_test_no_corr.json");
        std::fs::write(&tmp, serde_json::to_string(&trace_json).unwrap()).unwrap();

        let result = parse_trace(&tmp).unwrap();
        let (kid, _) = result.iter().next().unwrap();
        assert_eq!(kid.input_shapes, "");

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_generate_comparison_table() {
        let kid = KernelId {
            name: "my_kernel".to_string(),
            grid: "[384, 1, 1]".to_string(),
            input_shapes: "[38400, 128, 1]".to_string(),
        };

        let mut trace1 = BTreeMap::new();
        trace1.insert(kid.clone(), vec![0.5]);

        let mut trace2 = BTreeMap::new();
        trace2.insert(kid.clone(), vec![0.7]);

        let traces = vec![
            ("trace1".to_string(), trace1),
            ("trace2".to_string(), trace2),
        ];

        let table = generate_comparison_table(&traces);
        assert!(table.contains("my_kernel"));
        assert!(table.contains("trace1 (ms)"));
        assert!(table.contains("trace2 (ms)"));
        assert!(table.contains("0.500"));
        assert!(table.contains("0.700"));
    }

    #[test]
    fn test_generate_comparison_table_missing_kernel() {
        let kid1 = KernelId {
            name: "kernel_a".to_string(),
            grid: "[1, 1, 1]".to_string(),
            input_shapes: String::new(),
        };
        let kid2 = KernelId {
            name: "kernel_b".to_string(),
            grid: "[2, 1, 1]".to_string(),
            input_shapes: String::new(),
        };

        let mut t1 = BTreeMap::new();
        t1.insert(kid1, vec![1.0]);

        let mut t2 = BTreeMap::new();
        t2.insert(kid2, vec![2.0]);

        let traces = vec![("t1".to_string(), t1), ("t2".to_string(), t2)];
        let table = generate_comparison_table(&traces);

        assert!(table.contains("kernel_a"));
        assert!(table.contains("kernel_b"));
        assert!(table.contains("1.000"));
        assert!(table.contains("2.000"));
    }

    #[test]
    fn test_duplicate_kernel_ids_summed() {
        let kid = KernelId {
            name: "repeated".to_string(),
            grid: "[1, 1, 1]".to_string(),
            input_shapes: String::new(),
        };

        let mut t1 = BTreeMap::new();
        t1.insert(kid, vec![1.0, 2.0, 3.0]);

        let traces = vec![("t1".to_string(), t1)];
        let table = generate_comparison_table(&traces);
        assert!(table.contains("6.000"));
    }
}
