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

/// Compute p50 (median) of a sorted slice.
fn percentile50(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Stats for a kernel in a single trace.
struct KernelStats {
    p50: f64,
    max: f64,
    total: f64,
}

fn compute_stats(durations: &[f64]) -> KernelStats {
    let mut sorted = durations.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    KernelStats {
        p50: percentile50(&sorted),
        max: sorted.last().copied().unwrap_or(0.0),
        total: sorted.iter().sum(),
    }
}

/// Generate a markdown comparison table across multiple traces.
///
/// Each trace gets three sub-columns: p50, max, total (all in ms).
/// Rows show `[num_instances] kernel_name` and are sorted by max execution time descending.
pub fn generate_comparison_table(traces: &[(String, BTreeMap<KernelId, Vec<f64>>)]) -> String {
    let all_kernel_ids: BTreeSet<&KernelId> = traces
        .iter()
        .flat_map(|(_, kernels)| kernels.keys())
        .collect();

    if all_kernel_ids.is_empty() {
        return "No GPU kernel events found in any trace.\n".to_string();
    }

    // Pre-compute stats for sorting.
    // Sort key: maximum "max" value across all traces for each kernel_id, descending.
    let mut kernel_list: Vec<&KernelId> = all_kernel_ids.into_iter().collect();
    kernel_list.sort_by(|a, b| {
        let max_a = traces
            .iter()
            .filter_map(|(_, k)| k.get(*a).map(|d| compute_stats(d).max))
            .fold(0.0_f64, f64::max);
        let max_b = traces
            .iter()
            .filter_map(|(_, k)| k.get(*b).map(|d| compute_stats(d).max))
            .fold(0.0_f64, f64::max);
        max_b
            .partial_cmp(&max_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let trace_names: Vec<&str> = traces.iter().map(|(name, _)| name.as_str()).collect();
    let mut lines = Vec::new();

    // Header
    let mut header = "| Kernel | Grid | Input Shapes |".to_string();
    for name in &trace_names {
        header.push_str(&format!(
            " {} p50 (ms) | {} max (ms) | {} total (ms) |",
            name, name, name
        ));
    }
    lines.push(header);

    // Separator
    let mut sep = "| --- | --- | --- |".to_string();
    for _ in &trace_names {
        sep.push_str(" ---: | ---: | ---: |");
    }
    lines.push(sep);

    // Data rows
    for kid in &kernel_list {
        // Determine instance count (max across traces for this kernel_id).
        let max_count = traces
            .iter()
            .filter_map(|(_, k)| k.get(*kid).map(|d| d.len()))
            .max()
            .unwrap_or(0);

        let mut row = format!(
            "| `[{}] {}` | `{}` | `{}` |",
            max_count,
            escape_md(&kid.name),
            escape_md(&kid.grid),
            escape_md(&kid.input_shapes)
        );
        for (_, kernels) in traces {
            if let Some(durations) = kernels.get(*kid) {
                let stats = compute_stats(durations);
                row.push_str(&format!(
                    " {:.3} | {:.3} | {:.3} |",
                    stats.p50, stats.max, stats.total
                ));
            } else {
                row.push_str(" | | |");
            }
        }
        lines.push(row);
    }

    lines.join("\n") + "\n"
}

/// Merge multiple trace files into a single Chrome trace JSON.
///
/// CPU processes are sorted before GPU processes using `process_sort_index`
/// metadata events. CPU processes are identified by having categories like
/// `cpu_op`/`cuda_runtime`; GPU processes have `kernel`/`gpu_memcpy`.
///
/// When merging multiple files, pids are offset to avoid collisions.
pub fn merge_traces(paths: &[std::path::PathBuf]) -> Result<serde_json::Value> {
    let mut all_events: Vec<serde_json::Value> = Vec::new();
    // Track which categories each pid uses, to classify CPU vs GPU.
    let mut cats_by_pid: HashMap<i64, std::collections::HashSet<String>> = HashMap::new();
    let pid_offset_step: i64 = 1_000_000;

    for (file_idx, path) in paths.iter().enumerate() {
        let content = util::read_maybe_gzipped(path)?;
        let raw: serde_json::Value = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse JSON from {}", path.display()))?;
        let events = raw
            .get("traceEvents")
            .and_then(|v| v.as_array())
            .with_context(|| format!("Missing traceEvents in {}", path.display()))?;

        let offset = if paths.len() > 1 {
            file_idx as i64 * pid_offset_step
        } else {
            0
        };

        for event in events {
            let mut event = event.clone();

            // Offset pid for multi-file merges to avoid collisions.
            if offset != 0
                && let Some(pid) = event.get("pid").and_then(|v| v.as_i64())
            {
                event["pid"] = serde_json::json!(pid + offset);
            }

            // Track categories per pid.
            if let Some(cat) = event.get("cat").and_then(|v| v.as_str())
                && !cat.is_empty()
            {
                let pid = event.get("pid").and_then(|v| v.as_i64()).unwrap_or(0);
                cats_by_pid.entry(pid).or_default().insert(cat.to_string());
            }

            // Strip existing process_sort_index events — we'll add our own.
            let is_sort_index = event.get("ph").and_then(|v| v.as_str()) == Some("M")
                && event.get("name").and_then(|v| v.as_str()) == Some("process_sort_index");
            if is_sort_index {
                continue;
            }

            all_events.push(event);
        }
    }

    // Add process_sort_index metadata events.
    // CPU processes (with cpu_op/cuda_runtime categories) get sort_index 0.
    // GPU processes (with kernel/gpu_memcpy categories) get sort_index 1.
    // Other processes get sort_index 2.
    let gpu_cats: std::collections::HashSet<&str> =
        ["kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"]
            .into_iter()
            .collect();
    let cpu_cats: std::collections::HashSet<&str> = ["cpu_op", "cuda_runtime", "cuda_driver"]
        .into_iter()
        .collect();

    for (pid, cats) in &cats_by_pid {
        let has_cpu = cats.iter().any(|c| cpu_cats.contains(c.as_str()));
        let has_gpu = cats.iter().any(|c| gpu_cats.contains(c.as_str()));
        let sort_index = if has_cpu {
            0
        } else if has_gpu {
            1
        } else {
            2
        };
        all_events.push(serde_json::json!({
            "ph": "M",
            "pid": pid,
            "tid": 0,
            "name": "process_sort_index",
            "args": {"sort_index": sort_index}
        }));
    }

    Ok(serde_json::json!({ "traceEvents": all_events }))
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
    fn test_generate_table_p50_max_total() {
        let kid = KernelId {
            name: "my_kernel".to_string(),
            grid: "[384, 1, 1]".to_string(),
            input_shapes: "[38400, 128, 1]".to_string(),
        };

        let mut trace1 = BTreeMap::new();
        trace1.insert(kid.clone(), vec![1.0, 2.0, 3.0]);

        let traces = vec![("t1".to_string(), trace1)];
        let table = generate_comparison_table(&traces);

        // Header should have p50, max, total sub-columns
        assert!(table.contains("t1 p50 (ms)"));
        assert!(table.contains("t1 max (ms)"));
        assert!(table.contains("t1 total (ms)"));
        // [3] instances
        assert!(table.contains("[3] my_kernel"));
        // p50=2.0, max=3.0, total=6.0
        assert!(table.contains("2.000"));
        assert!(table.contains("3.000"));
        assert!(table.contains("6.000"));
    }

    #[test]
    fn test_generate_table_sorted_by_max_descending() {
        let kid_fast = KernelId {
            name: "fast_kernel".to_string(),
            grid: "[1, 1, 1]".to_string(),
            input_shapes: String::new(),
        };
        let kid_slow = KernelId {
            name: "slow_kernel".to_string(),
            grid: "[1, 1, 1]".to_string(),
            input_shapes: String::new(),
        };

        let mut t1 = BTreeMap::new();
        t1.insert(kid_fast, vec![0.1]);
        t1.insert(kid_slow, vec![10.0]);

        let traces = vec![("t1".to_string(), t1)];
        let table = generate_comparison_table(&traces);

        // slow_kernel (max=10.0) should appear before fast_kernel (max=0.1)
        let slow_pos = table.find("slow_kernel").unwrap();
        let fast_pos = table.find("fast_kernel").unwrap();
        assert!(slow_pos < fast_pos);
    }

    #[test]
    fn test_generate_table_missing_kernel() {
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
        // Missing kernel should have empty p50/max/total cells
        assert!(table.contains("| | | |"));
    }

    #[test]
    fn test_percentile50() {
        assert!((percentile50(&[1.0, 2.0, 3.0]) - 2.0).abs() < 0.001);
        assert!((percentile50(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 0.001);
        assert!((percentile50(&[5.0]) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_merge_single_file_adds_sort_index() {
        let trace_json = serde_json::json!({
            "traceEvents": [
                {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 100, "dur": 50, "pid": 1000, "tid": 1},
                {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 200, "dur": 100, "pid": 0, "tid": 7}
            ]
        });

        let tmp = std::env::temp_dir().join("trace_test_merge_single.json");
        std::fs::write(&tmp, serde_json::to_string(&trace_json).unwrap()).unwrap();

        let result = merge_traces(&[tmp.clone()]).unwrap();
        let events = result["traceEvents"].as_array().unwrap();

        // Original 2 events + 2 sort_index metadata events
        assert_eq!(events.len(), 4);

        // Find sort_index events
        let sort_events: Vec<&serde_json::Value> = events
            .iter()
            .filter(|e| e.get("name").and_then(|v| v.as_str()) == Some("process_sort_index"))
            .collect();
        assert_eq!(sort_events.len(), 2);

        // CPU pid (1000) should have sort_index 0
        let cpu_sort = sort_events
            .iter()
            .find(|e| e["pid"].as_i64() == Some(1000))
            .unwrap();
        assert_eq!(cpu_sort["args"]["sort_index"].as_i64(), Some(0));

        // GPU pid (0) should have sort_index 1
        let gpu_sort = sort_events
            .iter()
            .find(|e| e["pid"].as_i64() == Some(0))
            .unwrap();
        assert_eq!(gpu_sort["args"]["sort_index"].as_i64(), Some(1));

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_merge_multi_file_offsets_pids() {
        let trace1 = serde_json::json!({
            "traceEvents": [
                {"ph": "X", "cat": "cpu_op", "name": "op1", "ts": 100, "dur": 50, "pid": 100, "tid": 1}
            ]
        });
        let trace2 = serde_json::json!({
            "traceEvents": [
                {"ph": "X", "cat": "cpu_op", "name": "op2", "ts": 200, "dur": 50, "pid": 100, "tid": 1}
            ]
        });

        let tmp1 = std::env::temp_dir().join("trace_test_merge_m1.json");
        let tmp2 = std::env::temp_dir().join("trace_test_merge_m2.json");
        std::fs::write(&tmp1, serde_json::to_string(&trace1).unwrap()).unwrap();
        std::fs::write(&tmp2, serde_json::to_string(&trace2).unwrap()).unwrap();

        let result = merge_traces(&[tmp1.clone(), tmp2.clone()]).unwrap();
        let events = result["traceEvents"].as_array().unwrap();

        // File 1 pid stays 100, file 2 pid becomes 100 + 1_000_000
        let pids: Vec<i64> = events
            .iter()
            .filter(|e| e.get("ph").and_then(|v| v.as_str()) == Some("X"))
            .filter_map(|e| e["pid"].as_i64())
            .collect();
        assert!(pids.contains(&100));
        assert!(pids.contains(&1_000_100));

        std::fs::remove_file(&tmp1).ok();
        std::fs::remove_file(&tmp2).ok();
    }
}
