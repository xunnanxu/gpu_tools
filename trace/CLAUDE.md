# Project: GPU Trace Analysis CLI

## System Architecture & Tech Stack
- **Language:** Rust (Stable)
- **Toolchain:** Use standard `cargo` commands (`build`, `run`, `test`).
- **Preferred Crates:** - CLI Parsing: `clap` (with the `derive` feature)
  - Error Handling: `anyhow` and `thiserror`
  - Serialization: `serde` and `serde_json`
  - Logging/Output: `tracing` and `tracing-subscriber`

## Coding Standards
- Strictly enforce `cargo fmt` for formatting.
- Ensure the code passes `cargo clippy -- -D warnings` without any errors.
- Separate core logic from CLI parsing: Keep `main.rs` extremely thin. Put the business logic in `lib.rs` or a dedicated `core` module.
- Write unit tests for all parsing logic.
- Code should mimic python click style (but using Rust native support when possible).

## Workflow Directive
- **CRITICAL:** At the start of every session, or when asking for the "next task", you MUST read `TODO.md` to understand the current project state.
- Update `TODO.md` automatically by checking off tasks (`[x]`) when they are fully tested and complete.
- Whenever a task is completed, make a git commit but do not push the changes yet.

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