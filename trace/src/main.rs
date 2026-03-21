use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::MultiProgress;
use std::path::{Path, PathBuf};
use tracing_indicatif::IndicatifWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// GPU trace analysis CLI for PyTorch profiler traces.
#[derive(Parser, Debug)]
#[command(name = "trace", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Debug, ValueEnum)]
enum ConvertFrom {
    Nsys,
}

#[derive(Clone, Debug, ValueEnum)]
enum ConvertTo {
    Json,
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

    /// Download trace files from a remote host via SSH.
    #[command(group(clap::ArgGroup::new("source").required(true).multiple(false).args(["trace", "remote_dir"])))]
    Download {
        /// Single remote trace file: ssh://host:/path/to/trace.json
        #[arg(short = 't', long = "trace")]
        trace: Option<String>,

        /// Remote directory to download recursively (*.json, *.json.gz only):
        /// ssh://host:/path/to/dir
        #[arg(short = 'r', long = "remote-dir")]
        remote_dir: Option<String>,

        /// Output path (file or directory).
        #[arg(short = 'o', long = "output", required = true)]
        output: PathBuf,

        /// Disable gzip optimization for large files.
        #[arg(long = "no-gzip", default_value_t = false)]
        no_gzip: bool,
    },

    /// List trace files on a remote host via SSH.
    #[command(group(clap::ArgGroup::new("source").required(true).multiple(false).args(["trace", "remote_dir"])))]
    List {
        /// Single remote trace file: ssh://host:/path/to/trace.json
        #[arg(short = 't', long = "trace")]
        trace: Option<String>,

        /// Remote directory to list recursively (*.json, *.json.gz only):
        /// ssh://host:/path/to/dir
        #[arg(short = 'r', long = "remote-dir")]
        remote_dir: Option<String>,
    },

    /// Convert trace files between formats.
    Convert {
        /// Input format.
        #[arg(long, default_value = "nsys", value_enum)]
        from: ConvertFrom,

        /// Output format.
        #[arg(long, default_value = "json", value_enum)]
        to: ConvertTo,

        /// Input file (e.g. report.nsys-rep).
        input: PathBuf,

        /// Output file path (defaults to <input_stem>.json).
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let mp = MultiProgress::new();
    let indicatif_writer: IndicatifWriter = IndicatifWriter::new(mp.clone());
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(indicatif_writer))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Analyze { traces, output } => run_analyze(&traces, &output),
        Command::Merge { traces, output } => run_merge(&traces, &output, &mp),
        Command::Download {
            trace,
            remote_dir,
            output,
            no_gzip,
        } => run_download(trace.as_deref(), remote_dir.as_deref(), &output, no_gzip),
        Command::List { trace, remote_dir } => run_list(trace.as_deref(), remote_dir.as_deref()),
        Command::Convert {
            from,
            to,
            input,
            output,
        } => run_convert(from, to, &input, output.as_ref()),
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

fn run_merge(traces: &[PathBuf], output: &PathBuf, mp: &MultiProgress) -> Result<()> {
    trace::util::validate_trace_files(traces)?;

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let merged = trace::merge::merge_traces(traces, mp)?;

    let file = std::fs::File::create(output)?;
    let writer = std::io::BufWriter::new(file);
    serde_json::to_writer(writer, &merged)?;

    tracing::info!("Wrote merged trace to {}", output.display());

    Ok(())
}

fn run_convert(
    _from: ConvertFrom,
    _to: ConvertTo,
    input: &Path,
    output: Option<&PathBuf>,
) -> Result<()> {
    anyhow::ensure!(input.exists(), "Input file not found: {}", input.display());
    anyhow::ensure!(input.is_file(), "Not a file: {}", input.display());

    let output_path = match output {
        Some(p) => p.clone(),
        None => input.with_extension("json"),
    };

    let trace = trace::convert::nsys_to_chrome_trace(input)?;

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = std::fs::File::create(&output_path)?;
    let writer = std::io::BufWriter::new(file);
    serde_json::to_writer(writer, &trace)?;

    tracing::info!("Wrote Chrome trace JSON to {}", output_path.display());

    Ok(())
}

fn run_download(
    trace: Option<&str>,
    remote_dir: Option<&str>,
    output: &Path,
    no_gzip: bool,
) -> Result<()> {
    trace::remote::run_download(trace, remote_dir, output, no_gzip)
}

fn run_list(trace: Option<&str>, remote_dir: Option<&str>) -> Result<()> {
    trace::remote::run_list(trace, remote_dir)
}
