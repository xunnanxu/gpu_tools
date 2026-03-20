//! Merge multiple Chrome trace JSON files into one.

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use tracing::info;

/// Open a file as a streaming reader, transparently decompressing gzip if needed.
///
/// Peeks at the first two bytes to detect the gzip magic (`0x1f 0x8b`) without
/// loading the whole file; chains those bytes back so no data is lost.
fn open_reader(path: &Path) -> Result<Box<dyn Read>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut buf = std::io::BufReader::new(file);

    let mut magic = [0u8; 2];
    let n = buf.read(&mut magic)?;
    // Reconstruct the full stream by prepending the bytes we already consumed.
    let full = std::io::Cursor::new(magic[..n].to_vec()).chain(buf);

    if n == 2 && magic[0] == 0x1f && magic[1] == 0x8b {
        Ok(Box::new(std::io::BufReader::new(GzDecoder::new(full))))
    } else {
        Ok(Box::new(full))
    }
}

/// Merge multiple trace files into a single Chrome trace JSON.
///
/// CPU processes are sorted before GPU processes using `process_sort_index`
/// metadata events. CPU processes are identified by having categories like
/// `cpu_op`/`cuda_runtime`; GPU processes have `kernel`/`gpu_memcpy`.
///
/// When merging multiple files, pids are offset to avoid collisions.
///
/// Timestamps are aligned across files: the file with the smallest earliest
/// `ts` is used as the reference; all other files are shifted backward so
/// their first event lines up with that reference, eliminating blank gaps
/// between traces in the viewer.
pub fn merge_traces(paths: &[PathBuf], mp: &MultiProgress) -> Result<serde_json::Value> {
    let pb = mp.add(ProgressBar::new(paths.len() as u64));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {pos}/{len} ({percent}%) [{elapsed}<{eta}] {wide_bar}")
            .expect("valid template")
            .progress_chars("=> "),
    );
    pb.set_message("Merging traces");

    let mut all_events: Vec<serde_json::Value> = Vec::new();
    // Track which categories each pid uses, to classify CPU vs GPU.
    let mut cats_by_pid: HashMap<i64, HashSet<String>> = HashMap::new();
    let pid_offset_step: i64 = 1_000_000;
    let mut processed: u64 = 0;
    let mut loaded_total: u64 = 0;

    // Per-file bookkeeping for timestamp alignment.
    let mut file_event_ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(paths.len());
    let mut file_min_ts: Vec<Option<f64>> = Vec::with_capacity(paths.len());

    for (file_idx, path) in paths.iter().enumerate() {
        info!("Loading {}", path.display());
        // Stream-parse: no need to hold the full file text and parsed tree simultaneously.
        let reader = open_reader(path)?;
        let mut raw: serde_json::Value = serde_json::from_reader(reader)
            .with_context(|| format!("Failed to parse JSON from {}", path.display()))?;

        // Move the events array out of `raw` to avoid cloning each element.
        let events = match raw["traceEvents"].take() {
            serde_json::Value::Array(arr) => arr,
            _ => anyhow::bail!("Missing or invalid traceEvents in {}", path.display()),
        };

        loaded_total += events.len() as u64;
        info!("Loaded {} ({} events)", path.display(), events.len());

        let offset = if paths.len() > 1 {
            file_idx as i64 * pid_offset_step
        } else {
            0
        };

        let start_idx = all_events.len();
        let mut min_ts: Option<f64> = None;

        for mut event in events {
            // Offset pid for multi-file merges to avoid collisions.
            if offset != 0
                && let Some(pid) = event.get("pid").and_then(|v| v.as_i64())
            {
                event["pid"] = serde_json::json!(pid + offset);
            }

            // Track per-file minimum timestamp for alignment.
            if let Some(ts) = event.get("ts").and_then(|v| v.as_f64()) {
                min_ts = Some(min_ts.map_or(ts, |m: f64| m.min(ts)));
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
            processed += 1;
            if processed.is_multiple_of(10_000) {
                pb.set_message(format!("Merging traces ({processed} / {loaded_total})"));
            }
        }

        file_event_ranges.push(start_idx..all_events.len());
        file_min_ts.push(min_ts);
        pb.inc(1);
    }

    // Align timestamps across files: shift each file's events so the earliest
    // event across all files becomes the common time origin.
    if paths.len() > 1 {
        let global_min = file_min_ts
            .iter()
            .filter_map(|&ts| ts)
            .fold(f64::INFINITY, f64::min);
        let global_min = if global_min.is_infinite() {
            0.0
        } else {
            global_min
        };
        for (range, min_ts) in file_event_ranges.iter().zip(file_min_ts.iter()) {
            let ts_shift = global_min - min_ts.unwrap_or(global_min);
            if ts_shift == 0.0 {
                continue;
            }
            for event in &mut all_events[range.clone()] {
                if let Some(ts) = event.get("ts").and_then(|v| v.as_f64()) {
                    event["ts"] = serde_json::json!(ts + ts_shift);
                }
            }
        }
    }

    // Add process_sort_index metadata events.
    // CPU processes (with cpu_op/cuda_runtime categories) get sort_index 0.
    // GPU processes (with kernel/gpu_memcpy categories) get sort_index 1.
    // Other processes get sort_index 2.
    let gpu_cats: HashSet<&str> = ["kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"]
        .into_iter()
        .collect();
    let cpu_cats: HashSet<&str> = ["cpu_op", "cuda_runtime", "cuda_driver"]
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

    pb.finish_with_message(format!("Merged {} files ({processed} events)", paths.len()));
    Ok(serde_json::json!({ "traceEvents": all_events }))
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let result = merge_traces(&[tmp.clone()], &MultiProgress::new()).unwrap();
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

        let result = merge_traces(&[tmp1.clone(), tmp2.clone()], &MultiProgress::new()).unwrap();
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

    #[test]
    fn test_merge_timestamps_aligned_to_global_minimum() {
        // file1 earliest ts = 100, file2 earliest ts = 300
        // file2 should be shifted by -200 so both start at 100.
        let trace1 = serde_json::json!({
            "traceEvents": [
                {"ph": "X", "cat": "cpu_op", "name": "op1", "ts": 100, "dur": 10, "pid": 1, "tid": 1},
                {"ph": "X", "cat": "cpu_op", "name": "op2", "ts": 150, "dur": 10, "pid": 1, "tid": 1}
            ]
        });
        let trace2 = serde_json::json!({
            "traceEvents": [
                {"ph": "X", "cat": "cpu_op", "name": "op3", "ts": 300, "dur": 10, "pid": 1, "tid": 1},
                {"ph": "X", "cat": "cpu_op", "name": "op4", "ts": 400, "dur": 10, "pid": 1, "tid": 1}
            ]
        });

        let tmp1 = std::env::temp_dir().join("trace_test_ts_align1.json");
        let tmp2 = std::env::temp_dir().join("trace_test_ts_align2.json");
        std::fs::write(&tmp1, serde_json::to_string(&trace1).unwrap()).unwrap();
        std::fs::write(&tmp2, serde_json::to_string(&trace2).unwrap()).unwrap();

        let result = merge_traces(&[tmp1.clone(), tmp2.clone()], &MultiProgress::new()).unwrap();
        let events = result["traceEvents"].as_array().unwrap();

        let ts_by_name: std::collections::HashMap<&str, f64> = events
            .iter()
            .filter_map(|e| {
                let name = e.get("name")?.as_str()?;
                let ts = e.get("ts")?.as_f64()?;
                Some((name, ts))
            })
            .collect();

        // file1 unchanged (already at global min)
        assert!((ts_by_name["op1"] - 100.0).abs() < 0.001);
        assert!((ts_by_name["op2"] - 150.0).abs() < 0.001);
        // file2 shifted back by 200
        assert!((ts_by_name["op3"] - 100.0).abs() < 0.001);
        assert!((ts_by_name["op4"] - 200.0).abs() < 0.001);

        std::fs::remove_file(&tmp1).ok();
        std::fs::remove_file(&tmp2).ok();
    }

    #[test]
    fn test_merge_timestamps_aligned_with_float_ts() {
        // Real PyTorch profiler traces use float microsecond timestamps.
        let trace1 = serde_json::json!({
            "traceEvents": [
                {"ph": "X", "cat": "cpu_op", "name": "op1", "ts": 6795772743043.665, "dur": 10.5, "pid": 1, "tid": 1},
                {"ph": "X", "cat": "cpu_op", "name": "op2", "ts": 6795772743240.594, "dur": 10.5, "pid": 1, "tid": 1}
            ]
        });
        let trace2 = serde_json::json!({
            "traceEvents": [
                {"ph": "X", "cat": "cpu_op", "name": "op3", "ts": 6795773000000.0, "dur": 10.5, "pid": 1, "tid": 1},
                {"ph": "X", "cat": "cpu_op", "name": "op4", "ts": 6795773100000.0, "dur": 10.5, "pid": 1, "tid": 1}
            ]
        });

        let tmp1 = std::env::temp_dir().join("trace_test_ts_float1.json");
        let tmp2 = std::env::temp_dir().join("trace_test_ts_float2.json");
        std::fs::write(&tmp1, serde_json::to_string(&trace1).unwrap()).unwrap();
        std::fs::write(&tmp2, serde_json::to_string(&trace2).unwrap()).unwrap();

        let result = merge_traces(&[tmp1.clone(), tmp2.clone()], &MultiProgress::new()).unwrap();
        let events = result["traceEvents"].as_array().unwrap();

        let ts_by_name: std::collections::HashMap<&str, f64> = events
            .iter()
            .filter_map(|e| {
                let name = e.get("name")?.as_str()?;
                let ts = e.get("ts")?.as_f64()?;
                Some((name, ts))
            })
            .collect();

        // file1 has the global min, so unchanged
        assert!((ts_by_name["op1"] - 6795772743043.665).abs() < 1.0);
        assert!((ts_by_name["op2"] - 6795772743240.594).abs() < 1.0);
        // file2 shifted back by (6795773000000 - 6795772743043.665) ≈ 256956.335
        let shift = 6795773000000.0 - 6795772743043.665;
        assert!((ts_by_name["op3"] - (6795773000000.0 - shift)).abs() < 1.0);
        assert!((ts_by_name["op4"] - (6795773100000.0 - shift)).abs() < 1.0);

        std::fs::remove_file(&tmp1).ok();
        std::fs::remove_file(&tmp2).ok();
    }
}
