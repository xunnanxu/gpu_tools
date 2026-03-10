use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

/// Compare GPU kernel execution times across multiple PyTorch profiler traces.
#[derive(Parser, Debug)]
#[command(name = "trace", version, about)]
struct Cli {
    /// Trace JSON files in Chrome trace format (repeatable).
    #[arg(short = 't', long = "trace", required = true)]
    traces: Vec<PathBuf>,

    /// Output path: a directory (writes comparison.md inside) or a .md file path.
    #[arg(short = 'o', long = "output", default_value = ".")]
    output: PathBuf,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    trace::util::validate_trace_files(&cli.traces)?;

    // If the output path looks like a file (has .md extension), use it directly.
    // Otherwise treat it as a directory and write comparison.md inside.
    let output_path = if cli.output.extension().is_some_and(|ext| ext == "md") {
        if let Some(parent) = cli.output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        cli.output.clone()
    } else {
        std::fs::create_dir_all(&cli.output)?;
        cli.output.join("comparison.md")
    };

    let mut all_traces = Vec::new();
    for path in &cli.traces {
        let trace_name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());

        tracing::info!("Parsing trace: {}", path.display());
        let kernels = trace::parse_trace(path)?;
        tracing::info!(
            "Found {} unique kernel IDs in {}",
            kernels.len(),
            trace_name
        );
        all_traces.push((trace_name, kernels));
    }

    let table = trace::generate_comparison_table(&all_traces);

    std::fs::write(&output_path, &table)?;
    tracing::info!("Wrote comparison table to {}", output_path.display());

    print!("{}", table);

    Ok(())
}
