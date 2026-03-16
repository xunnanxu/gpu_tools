use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// GPU trace analysis CLI for PyTorch profiler traces.
#[derive(Parser, Debug)]
#[command(name = "trace", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Analyze and compare GPU kernel execution times across traces.
    Analyze {
        /// Trace JSON files in Chrome trace format (repeatable).
        #[arg(short = 't', long = "trace", required = true)]
        traces: Vec<PathBuf>,

        /// Output path: a directory (writes comparison.md inside) or a .md file path.
        #[arg(short = 'o', long = "output", default_value = ".")]
        output: PathBuf,
    },

    /// Merge trace files into one, with CPU processes sorted before GPU.
    Merge {
        /// Trace JSON files to merge (repeatable).
        #[arg(short = 't', long = "trace", required = true)]
        traces: Vec<PathBuf>,

        /// Output JSON file path.
        #[arg(short = 'o', long = "output", required = true)]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Command::Analyze { traces, output } => run_analyze(&traces, &output),
        Command::Merge { traces, output } => run_merge(&traces, &output),
    }
}

fn run_analyze(traces: &[PathBuf], output: &PathBuf) -> Result<()> {
    trace::util::validate_trace_files(traces)?;

    let output_path = if output.extension().is_some_and(|ext| ext == "md") {
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        output.clone()
    } else {
        std::fs::create_dir_all(output)?;
        output.join("comparison.md")
    };

    let mut all_traces = Vec::new();
    for path in traces {
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

fn run_merge(traces: &[PathBuf], output: &PathBuf) -> Result<()> {
    trace::util::validate_trace_files(traces)?;

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let merged = trace::merge_traces(traces)?;

    let file = std::fs::File::create(output)?;
    let writer = std::io::BufWriter::new(file);
    serde_json::to_writer(writer, &merged)?;

    tracing::info!("Wrote merged trace to {}", output.display());

    Ok(())
}
