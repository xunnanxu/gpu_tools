//! Convert nsys-rep files to Chrome trace JSON format.

use anyhow::{Context, Result};
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::Path;

/// Convert an nsys-rep file to Chrome trace JSON.
///
/// The nsys-rep file is an SQLite database containing CUPTI activity records.
/// This reads the relevant tables and produces Chrome trace JSON viewable in
/// chrome://tracing or Perfetto.
pub fn nsys_to_chrome_trace(nsys_path: &Path) -> Result<serde_json::Value> {
    let conn = Connection::open_with_flags(
        nsys_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .with_context(|| format!("Failed to open nsys-rep file: {}", nsys_path.display()))?;

    let strings = load_strings(&conn)?;
    let min_ts = find_min_timestamp(&conn)?;

    let mut events: Vec<serde_json::Value> = Vec::new();

    log_skip(
        append_kernel_events(&conn, &strings, min_ts, &mut events),
        "kernel",
    );
    log_skip(
        append_runtime_events(&conn, &strings, min_ts, &mut events),
        "runtime",
    );
    log_skip(append_memcpy_events(&conn, min_ts, &mut events), "memcpy");
    log_skip(append_memset_events(&conn, min_ts, &mut events), "memset");
    log_skip(
        append_nvtx_events(&conn, &strings, min_ts, &mut events),
        "nvtx",
    );

    tracing::info!("Converted {} events", events.len());
    Ok(serde_json::json!({ "traceEvents": events }))
}

fn log_skip(result: Result<()>, kind: &str) {
    if let Err(e) = result {
        tracing::debug!("Skipping {kind} events: {e}");
    }
}

fn load_strings(conn: &Connection) -> Result<HashMap<i64, String>> {
    let mut stmt = conn.prepare("SELECT id, value FROM StringIds")?;
    let mut map = HashMap::new();
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        map.insert(row.get(0)?, row.get(1)?);
    }
    Ok(map)
}

fn find_min_timestamp(conn: &Connection) -> Result<i64> {
    let tables = [
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
        "NVTX_EVENTS",
    ];
    let mut min_ts = i64::MAX;
    for table in tables {
        let sql = format!("SELECT MIN(start) FROM \"{table}\"");
        match conn.query_row(&sql, [], |row| row.get::<_, Option<i64>>(0)) {
            Ok(Some(ts)) if ts < min_ts => min_ts = ts,
            _ => {}
        }
    }
    if min_ts == i64::MAX {
        min_ts = 0;
    }
    Ok(min_ts)
}

/// Convert nanoseconds to microseconds relative to min_ts.
fn ns_to_us(ns: i64, base: i64) -> f64 {
    (ns - base) as f64 / 1000.0
}

fn dur_us(start: i64, end: i64) -> f64 {
    (end - start) as f64 / 1000.0
}

fn resolve_name(strings: &HashMap<i64, String>, id: i64, fallback_prefix: &str) -> String {
    strings
        .get(&id)
        .cloned()
        .unwrap_or_else(|| format!("{fallback_prefix}_{id}"))
}

fn append_kernel_events(
    conn: &Connection,
    strings: &HashMap<i64, String>,
    min_ts: i64,
    events: &mut Vec<serde_json::Value>,
) -> Result<()> {
    let mut stmt = conn.prepare(
        "SELECT start, end, deviceId, streamId, correlationId, \
         demangledName, shortName, \
         gridX, gridY, gridZ, blockX, blockY, blockZ, \
         staticSharedMemory, dynamicSharedMemory, registersPerThread \
         FROM CUPTI_ACTIVITY_KIND_KERNEL",
    )?;

    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let start: i64 = row.get(0)?;
        let end: i64 = row.get(1)?;
        let device_id: i64 = row.get(2)?;
        let stream_id: i64 = row.get(3)?;
        let correlation_id: i64 = row.get(4)?;
        let demangled: i64 = row.get(5)?;
        let short: i64 = row.get(6)?;
        let grid_x: i64 = row.get(7)?;
        let grid_y: i64 = row.get(8)?;
        let grid_z: i64 = row.get(9)?;
        let block_x: i64 = row.get(10)?;
        let block_y: i64 = row.get(11)?;
        let block_z: i64 = row.get(12)?;
        let static_sm: i64 = row.get(13)?;
        let dynamic_sm: i64 = row.get(14)?;
        let regs: i64 = row.get(15)?;

        let name = strings
            .get(&demangled)
            .or_else(|| strings.get(&short))
            .cloned()
            .unwrap_or_else(|| format!("kernel_{demangled}"));

        events.push(serde_json::json!({
            "ph": "X",
            "cat": "kernel",
            "name": name,
            "ts": ns_to_us(start, min_ts),
            "dur": dur_us(start, end),
            "pid": device_id,
            "tid": stream_id,
            "args": {
                "correlation": correlation_id,
                "grid": [grid_x, grid_y, grid_z],
                "block": [block_x, block_y, block_z],
                "shared_memory": static_sm + dynamic_sm,
                "registers_per_thread": regs
            }
        }));
    }

    Ok(())
}

fn append_runtime_events(
    conn: &Connection,
    strings: &HashMap<i64, String>,
    min_ts: i64,
    events: &mut Vec<serde_json::Value>,
) -> Result<()> {
    let mut stmt = conn.prepare(
        "SELECT start, end, globalTid, correlationId, nameId \
         FROM CUPTI_ACTIVITY_KIND_RUNTIME",
    )?;

    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let start: i64 = row.get(0)?;
        let end: i64 = row.get(1)?;
        let global_tid: i64 = row.get(2)?;
        let correlation_id: i64 = row.get(3)?;
        let name_id: i64 = row.get(4)?;

        let name = resolve_name(strings, name_id, "runtime");
        // globalTid: upper 32 bits = process ID, lower 32 bits = thread ID.
        let pid = ((global_tid >> 32) & 0xFFFF_FFFF) as i32;
        let tid = (global_tid & 0xFFFF_FFFF) as i32;

        events.push(serde_json::json!({
            "ph": "X",
            "cat": "cuda_runtime",
            "name": name,
            "ts": ns_to_us(start, min_ts),
            "dur": dur_us(start, end),
            "pid": pid,
            "tid": tid,
            "args": {
                "correlation": correlation_id,
                "External id": correlation_id
            }
        }));
    }

    Ok(())
}

fn memcpy_kind_name(kind: i64) -> &'static str {
    match kind {
        1 => "HtoD",
        2 => "DtoH",
        3 => "HtoH",
        4 => "DtoD",
        8 => "PtoP",
        _ => "Unknown",
    }
}

fn append_memcpy_events(
    conn: &Connection,
    min_ts: i64,
    events: &mut Vec<serde_json::Value>,
) -> Result<()> {
    let mut stmt = conn.prepare(
        "SELECT start, end, deviceId, streamId, correlationId, bytes, copyKind \
         FROM CUPTI_ACTIVITY_KIND_MEMCPY",
    )?;

    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let start: i64 = row.get(0)?;
        let end: i64 = row.get(1)?;
        let device_id: i64 = row.get(2)?;
        let stream_id: i64 = row.get(3)?;
        let correlation_id: i64 = row.get(4)?;
        let bytes: i64 = row.get(5)?;
        let copy_kind: i64 = row.get(6)?;

        let name = format!("Memcpy {}", memcpy_kind_name(copy_kind));

        events.push(serde_json::json!({
            "ph": "X",
            "cat": "gpu_memcpy",
            "name": name,
            "ts": ns_to_us(start, min_ts),
            "dur": dur_us(start, end),
            "pid": device_id,
            "tid": stream_id,
            "args": {
                "correlation": correlation_id,
                "bytes": bytes,
                "copy_kind": memcpy_kind_name(copy_kind)
            }
        }));
    }

    Ok(())
}

fn append_memset_events(
    conn: &Connection,
    min_ts: i64,
    events: &mut Vec<serde_json::Value>,
) -> Result<()> {
    let mut stmt = conn.prepare(
        "SELECT start, end, deviceId, streamId, correlationId, bytes, value \
         FROM CUPTI_ACTIVITY_KIND_MEMSET",
    )?;

    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let start: i64 = row.get(0)?;
        let end: i64 = row.get(1)?;
        let device_id: i64 = row.get(2)?;
        let stream_id: i64 = row.get(3)?;
        let correlation_id: i64 = row.get(4)?;
        let bytes: i64 = row.get(5)?;
        let value: i64 = row.get(6)?;

        events.push(serde_json::json!({
            "ph": "X",
            "cat": "gpu_memset",
            "name": "Memset",
            "ts": ns_to_us(start, min_ts),
            "dur": dur_us(start, end),
            "pid": device_id,
            "tid": stream_id,
            "args": {
                "correlation": correlation_id,
                "bytes": bytes,
                "value": value
            }
        }));
    }

    Ok(())
}

fn append_nvtx_events(
    conn: &Connection,
    strings: &HashMap<i64, String>,
    min_ts: i64,
    events: &mut Vec<serde_json::Value>,
) -> Result<()> {
    let mut stmt = conn.prepare(
        "SELECT start, end, text, globalTid \
         FROM NVTX_EVENTS WHERE end IS NOT NULL",
    )?;

    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let start: i64 = row.get(0)?;
        let end: i64 = row.get(1)?;
        let text_id: Option<i64> = row.get(2)?;
        let global_tid: i64 = row.get(3)?;

        let name = text_id
            .and_then(|id| strings.get(&id).cloned())
            .unwrap_or_else(|| "nvtx".to_string());
        let pid = ((global_tid >> 32) & 0xFFFF_FFFF) as i32;
        let tid = (global_tid & 0xFFFF_FFFF) as i32;

        events.push(serde_json::json!({
            "ph": "X",
            "cat": "nvtx",
            "name": name,
            "ts": ns_to_us(start, min_ts),
            "dur": dur_us(start, end),
            "pid": pid,
            "tid": tid
        }));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_db(path: &Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
             INSERT INTO StringIds VALUES (1, 'my_kernel');
             INSERT INTO StringIds VALUES (2, 'cudaLaunchKernel');
             INSERT INTO StringIds VALUES (3, 'test_annotation');

             CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                 start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER,
                 correlationId INTEGER, demangledName INTEGER, shortName INTEGER,
                 gridX INTEGER, gridY INTEGER, gridZ INTEGER,
                 blockX INTEGER, blockY INTEGER, blockZ INTEGER,
                 staticSharedMemory INTEGER, dynamicSharedMemory INTEGER,
                 registersPerThread INTEGER
             );
             INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                 (1000000, 1500000, 0, 7, 100, 1, 1, 384, 1, 1, 128, 1, 1, 0, 1024, 32);

             CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                 start INTEGER, end INTEGER, globalTid INTEGER,
                 correlationId INTEGER, nameId INTEGER
             );
             INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
                 (900000, 950000, 4294967297, 100, 2);

             CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                 start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER,
                 correlationId INTEGER, bytes INTEGER, copyKind INTEGER
             );
             INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES
                 (2000000, 2100000, 0, 7, 101, 4096, 1);

             CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
                 start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER,
                 correlationId INTEGER, bytes INTEGER, value INTEGER
             );

             CREATE TABLE NVTX_EVENTS (
                 start INTEGER, end INTEGER, eventType INTEGER,
                 rangeId INTEGER, category INTEGER, color INTEGER,
                 text INTEGER, globalTid INTEGER, endGlobalTid INTEGER
             );
             INSERT INTO NVTX_EVENTS VALUES
                 (500000, 2500000, 59, 0, 0, 0, 3, 4294967297, 4294967297);",
        )
        .unwrap();
    }

    #[test]
    fn test_convert_nsys_to_chrome_trace() {
        let tmp = std::env::temp_dir().join("test_nsys_convert.sqlite");
        create_test_db(&tmp);

        let result = nsys_to_chrome_trace(&tmp).unwrap();
        let events = result["traceEvents"].as_array().unwrap();

        // 1 kernel + 1 runtime + 1 memcpy + 1 nvtx = 4 events
        assert_eq!(events.len(), 4);

        let kernel = events.iter().find(|e| e["cat"] == "kernel").unwrap();
        assert_eq!(kernel["name"], "my_kernel");
        assert_eq!(kernel["ph"], "X");
        assert_eq!(kernel["pid"], 0);
        assert_eq!(kernel["tid"], 7);
        assert_eq!(kernel["args"]["grid"], serde_json::json!([384, 1, 1]));
        assert_eq!(kernel["args"]["correlation"], 100);

        let runtime = events.iter().find(|e| e["cat"] == "cuda_runtime").unwrap();
        assert_eq!(runtime["name"], "cudaLaunchKernel");
        assert_eq!(runtime["args"]["correlation"], 100);

        let memcpy = events.iter().find(|e| e["cat"] == "gpu_memcpy").unwrap();
        assert!(memcpy["name"].as_str().unwrap().contains("HtoD"));
        assert_eq!(memcpy["args"]["bytes"], 4096);

        let nvtx = events.iter().find(|e| e["cat"] == "nvtx").unwrap();
        assert_eq!(nvtx["name"], "test_annotation");

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_convert_missing_tables() {
        let tmp = std::env::temp_dir().join("test_nsys_empty.sqlite");
        let conn = Connection::open(&tmp).unwrap();
        conn.execute_batch("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);")
            .unwrap();
        drop(conn);

        let result = nsys_to_chrome_trace(&tmp).unwrap();
        let events = result["traceEvents"].as_array().unwrap();
        assert!(events.is_empty());

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_timestamps_normalized() {
        let tmp = std::env::temp_dir().join("test_nsys_timestamps.sqlite");
        let conn = Connection::open(&tmp).unwrap();
        conn.execute_batch(
            "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
             INSERT INTO StringIds VALUES (1, 'kern');
             CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                 start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER,
                 correlationId INTEGER, demangledName INTEGER, shortName INTEGER,
                 gridX INTEGER, gridY INTEGER, gridZ INTEGER,
                 blockX INTEGER, blockY INTEGER, blockZ INTEGER,
                 staticSharedMemory INTEGER, dynamicSharedMemory INTEGER,
                 registersPerThread INTEGER
             );
             INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                 (10000000, 10500000, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0);",
        )
        .unwrap();
        drop(conn);

        let result = nsys_to_chrome_trace(&tmp).unwrap();
        let events = result["traceEvents"].as_array().unwrap();
        let kernel = &events[0];

        // start=10000000ns, min_ts=10000000ns -> ts=0.0us
        assert!((kernel["ts"].as_f64().unwrap()).abs() < 0.001);
        // dur = 500000ns = 500.0us
        assert!((kernel["dur"].as_f64().unwrap() - 500.0).abs() < 0.001);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_memcpy_kind_names() {
        assert_eq!(memcpy_kind_name(1), "HtoD");
        assert_eq!(memcpy_kind_name(2), "DtoH");
        assert_eq!(memcpy_kind_name(4), "DtoD");
        assert_eq!(memcpy_kind_name(99), "Unknown");
    }
}
