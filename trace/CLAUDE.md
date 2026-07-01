# Project: GPU Trace Analysis CLI

## File Structure
- `src/main.rs`: Handles the `clap` CLI entry point and argument parsing.
- `src/lib.rs`: Handles main logic like chrome trace JSON parsing, trace analysis.
- `src/util.rs`: Contains reusable utils like optional gzipped content handling, input validation.
- `src/remote.rs`: SSH/SCP remote operations for downloading and listing trace files.

## System Architecture & Tech Stack
- **Language:** Rust (Stable)
- **Toolchain:** Use standard `cargo` commands (`build`, `run`, `test`).
- **Preferred Crates:** - CLI Parsing: `clap` (with the `derive` feature)
  - Error Handling: `anyhow` and `thiserror`
  - Serialization: `serde` and `serde_json`
  - Logging/Output: `tracing` and `tracing-subscriber`

## CLI Commands

### `analyze`
- Usage: `trace analyze -t trace1.json -t trace2.json -o output_dir`
- Parses chrome trace JSON files produced by torch profiler.
- Focuses on **GPU kernel events only**. Input params come from CPU side via ac2g connection events.
- Each GPU event is keyed by a `kernel_id` tuple: (kernel name, launch grid, input parameter shape). The value is execution time in ms.
- Outputs a markdown comparison table: columns are trace names, rows are `kernel_id`s.
- Each kernel row shows `[num_instances] kernel_id` with p50, max, and total time listed separately.
- Missing `kernel_id` matches across traces leave the cell empty.
- Rows sorted by max execution time descending.

### `merge`
- Usage: `trace merge -t trace1.json -t trace2.json -o output.json`
- Merges multiple trace files into one.
- CPU processes (e.g. `python3 988976`) sorted before GPU processes (e.g. `python3.0` with `stream X`) using `process_sort_index`.
- **Timestamp alignment:** all files are shifted so the globally earliest `ts` across all files becomes the common origin. Each file's shift = `global_min_ts - file_min_ts`. This eliminates the blank gap between traces in the viewer when files were recorded at different wall-clock offsets.

### `download`
- Usage: `trace download ssh://host:/path/to/trace.json [more_sources...] -o local/path`
- Downloads trace files or directories from a remote host via SSH (`ssh://host:/path` format, honors `~/.ssh/config`).
- Accepts one or more positional source URLs. Each source is auto-classified as a file or directory on the remote (single `find -maxdepth 0 -printf '%y'` call per source, run up-front before any download begins).
- **Single file source:** output supports renaming (`ssh://host:/a.json -o b.json`) or keeping name (`ssh://host:/a.json -o dir/`).
- **Directory source or multiple sources:** output must be a directory. Directories are walked recursively, downloading only `*.json` and `*.json.gz` files and preserving relative structure below the source dir.
- **Gzip optimization (default on):** if a raw JSON file exceeds the threshold (default 500MB), gzips it on the remote host first (`gzip -kf`) then downloads the `.json.gz`. Disable with `--no-compress`. Threshold configurable via `--compress-threshold <MB>`.
- Logs each SSH/SCP operation via `info!`.

### `list`
- Usage: `trace list -t ssh://host:/path/to/trace.json`
- Usage: `trace list -r ssh://host:/path/to/dir`
- Lists trace files on a remote host via SSH, showing path, human-readable size, and modification date.
- `-t` for single file info, `-r` for recursive directory listing (mutually exclusive, one required).
- Recursive mode only lists `*.json` and `*.json.gz` files.
- Logs each SSH operation via `info!`.

## Coding Standards
- Strictly enforce `cargo fmt` for formatting.
- Ensure the code passes `cargo clippy -- -D warnings` without any errors.
- Separate core logic from CLI parsing: Keep `main.rs` extremely thin. Put the business logic in `lib.rs` or a dedicated `core` module.
- Write unit tests for all parsing logic.
- Code should mimic python click style (but using Rust native support when possible).

## Workflow Directive
- **CRITICAL:** At the start of every session, or when asking for the "next task", you MUST read `TODO.md` to understand the current project state.
- Update `TODO.md` automatically by checking off tasks (`[x]`) when they are fully tested and complete.

## Trace Parsing Details

### GPU -> CPU Trace Linking
GPU kernels link to CPU ops through a correlation chain, not direct External id matching:                       

GPU kernel args.correlation → matches cuda_runtime args.correlation → cuda_runtime args.External id → matches cpu_op args.External id → Input Dims

### Input Dim Handling

Scalars are represented as empty arrays.

```
[[1024, 128], [], [128, 256], []]
```

is actually

```
[[1024, 128], 1, [128, 256], 1]
```
