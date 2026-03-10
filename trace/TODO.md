## File structure
- `src/main.rs`: Handles the `clap` CLI entry point and argument parsing.
- `src/lib.rs`: Handles main logic like chrome trace JSON parsing, trace analysis
- `src/util.rs`: Contains reusable utils like optional gzipped content handling, input validation.

## Checklist

* [x] CLI takes multiple trace json files in chrome trace format (produced by torch profiler) that are intaken by -t trace1.json -t trace2.json ... each trace is tracked by its file name. The output directory is specified by `--output` or `-o`.
* [x] For each trace file, we care about **GPU kernel events ONLY**. However, the input params are recorded on CPU side via the special ac2g connection events. Each GPU event is keyed by the kernel name (e.g. `nvjet_tst_256x256_64x4_2x1_2cta_v_bz_bias_NNT`), the launch grid (e.g. `[384, 1, 1]`), and the input parameter shape (e.g. `[38400, 128, 1]`). This tuple will be later referred to as `kernel_id`. And the corresponding value is the execution time of the kernel in ms.
* [x] Create a table view at the end for comparison as markdown into the output directory. Each column is keyed by the trace name. Each row is the execution time keyed by that `kernel_id`. If a `kernel_id` does not have a corresponding match, leave that cell empty.
