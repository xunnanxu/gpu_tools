use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tracing::info;

/// Parsed SSH URL of the form `ssh://host:/path/to/file`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SshUrl {
    pub host: String,
    pub remote_path: String,
}

impl SshUrl {
    /// Parse a URL like `ssh://myhost:/data/traces/trace.json`.
    pub fn parse(url: &str) -> Result<Self> {
        let rest = url
            .strip_prefix("ssh://")
            .ok_or_else(|| anyhow::anyhow!("URL must start with ssh:// (got: {url})"))?;

        let (host, path) = rest.split_once(':').ok_or_else(|| {
            anyhow::anyhow!("Invalid SSH URL format. Expected ssh://host:/path, got: {url}")
        })?;

        anyhow::ensure!(!host.is_empty(), "SSH host cannot be empty in URL: {url}");
        anyhow::ensure!(
            !path.is_empty(),
            "Remote path cannot be empty in URL: {url}"
        );

        Ok(SshUrl {
            host: host.to_string(),
            remote_path: path.to_string(),
        })
    }

    /// Format as `host:path` for use with scp.
    pub fn to_scp_source(&self) -> String {
        format!("{}:{}", self.host, self.remote_path)
    }
}

/// Information about a single remote file.
#[derive(Debug)]
struct RemoteFileInfo {
    path: String,
    size_bytes: u64,
    modified: String,
}

/// Run an ssh command and return stdout.
fn ssh_exec(host: &str, remote_command: &str) -> Result<String> {
    info!("ssh {host}: {remote_command}");
    let output = std::process::Command::new("ssh")
        .args([host, remote_command])
        .output()
        .with_context(|| format!("Failed to run ssh to {host}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ssh command failed on {host}: {stderr}");
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Run scp to download a file.
fn scp_download(scp_source: &str, local_dest: &Path) -> Result<()> {
    info!("scp {} -> {}", scp_source, local_dest.display());
    let status = std::process::Command::new("scp")
        .args([scp_source, &local_dest.to_string_lossy()])
        .status()
        .with_context(|| format!("Failed to run scp from {scp_source}"))?;

    if !status.success() {
        anyhow::bail!("scp failed: {scp_source} -> {}", local_dest.display());
    }
    Ok(())
}

/// Get file info for a single remote file using `find -printf` (handles symlinks).
fn stat_remote_file(host: &str, path: &str) -> Result<RemoteFileInfo> {
    // Use find -L -maxdepth 0 to stat a single file, following symlinks.
    // find -printf interprets \t as tab (unlike stat --format).
    let cmd = format!(r"find -L '{path}' -maxdepth 0 -printf '%s\t%TY-%Tm-%Td %TH:%TM\t%p\n'");
    let output = ssh_exec(host, &cmd)?;
    let mut files = parse_find_printf_output(&output)?;
    files
        .pop()
        .ok_or_else(|| anyhow::anyhow!("File not found on remote: {path}"))
}

/// List trace files recursively in a remote directory.
/// Uses `find -L` to follow symlinks (NFS safety) and `-printf` for reliable
/// tab-delimited output. Only includes *.json and *.json.gz files.
fn list_remote_dir(host: &str, dir: &str) -> Result<Vec<RemoteFileInfo>> {
    let cmd = format!(
        r"find -L '{dir}' -type f \( -name '*.json' -o -name '*.json.gz' \) -printf '%s\t%TY-%Tm-%Td %TH:%TM\t%p\n'"
    );
    let output = ssh_exec(host, &cmd)?;
    let mut files = parse_find_printf_output(&output)?;
    files.sort_by(|a, b| b.modified.cmp(&a.modified));
    Ok(files)
}

/// Parse output lines of `find -printf '%s\t%TY-%Tm-%Td %TH:%TM\t%p\n'`.
fn parse_find_printf_output(output: &str) -> Result<Vec<RemoteFileInfo>> {
    let mut files = Vec::new();
    for line in output.lines() {
        if line.trim().is_empty() {
            continue;
        }
        // Format: size\tmodified\tpath
        let (size_str, rest) = line
            .split_once('\t')
            .ok_or_else(|| anyhow::anyhow!("Unexpected find output line: {line}"))?;
        let (modified, path) = rest
            .split_once('\t')
            .ok_or_else(|| anyhow::anyhow!("Unexpected find output line: {line}"))?;
        files.push(RemoteFileInfo {
            size_bytes: size_str.parse()?,
            modified: modified.to_string(),
            path: path.to_string(),
        });
    }
    Ok(files)
}

/// Format bytes as human-readable size.
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Execute the `list` command.
pub fn run_list(trace: Option<&str>, remote_dir: Option<&str>) -> Result<()> {
    let url = SshUrl::parse(trace.or(remote_dir).unwrap())?;

    let files = if trace.is_some() {
        vec![stat_remote_file(&url.host, &url.remote_path)?]
    } else {
        list_remote_dir(&url.host, &url.remote_path)?
    };

    if files.is_empty() {
        info!("No trace files found.");
        return Ok(());
    }

    println!("{:<12} {:<20} Path", "Size", "Modified");
    println!("{}", "-".repeat(72));
    for f in &files {
        println!(
            "{:<12} {:<20} {}",
            format_size(f.size_bytes),
            f.modified,
            f.path
        );
    }
    info!("Found {} trace file(s)", files.len());
    Ok(())
}

/// Determine the local output path for a single-file download.
///
/// If `output` ends with `/` or is an existing directory, the remote filename
/// is placed inside it. Otherwise `output` is treated as the target filename.
pub fn resolve_single_output(remote_path: &str, output: &Path) -> PathBuf {
    let output_str = output.to_string_lossy();
    if output_str.ends_with('/') || output.is_dir() {
        let filename = Path::new(remote_path)
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("trace.json"));
        output.join(filename)
    } else {
        output.to_path_buf()
    }
}

const GZIP_THRESHOLD: u64 = 1_073_741_824; // 1 GB

/// Execute the `download` command.
pub fn run_download(
    trace: Option<&str>,
    remote_dir: Option<&str>,
    output: &Path,
    no_gzip: bool,
) -> Result<()> {
    let url = SshUrl::parse(trace.or(remote_dir).unwrap())?;

    if trace.is_some() {
        download_single(&url, output, no_gzip)
    } else {
        download_recursive(&url, output, no_gzip)
    }
}

fn download_single(url: &SshUrl, output: &Path, no_gzip: bool) -> Result<()> {
    let local_path = resolve_single_output(&url.remote_path, output);

    if let Some(parent) = local_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let (scp_remote_path, final_local_path) = if !no_gzip && !url.remote_path.ends_with(".gz") {
        let info = stat_remote_file(&url.host, &url.remote_path)?;
        if info.size_bytes > GZIP_THRESHOLD {
            info!(
                "File is {}, compressing on remote first",
                format_size(info.size_bytes)
            );
            ssh_exec(&url.host, &format!("gzip -kf '{}'", url.remote_path))?;
            let gz_remote = format!("{}.gz", url.remote_path);
            let gz_local = ensure_gz_extension(&local_path);
            (gz_remote, gz_local)
        } else {
            (url.remote_path.clone(), local_path)
        }
    } else {
        (url.remote_path.clone(), local_path)
    };

    let scp_source = format!("{}:{}", url.host, scp_remote_path);
    scp_download(&scp_source, &final_local_path)?;
    info!("Downloaded to {}", final_local_path.display());
    Ok(())
}

fn download_recursive(url: &SshUrl, output: &Path, no_gzip: bool) -> Result<()> {
    std::fs::create_dir_all(output)?;

    let files = list_remote_dir(&url.host, &url.remote_path)?;

    if files.is_empty() {
        info!("No trace files found in remote directory.");
        return Ok(());
    }

    info!("Found {} trace file(s) to download", files.len());

    for file_info in &files {
        let relative = file_info
            .path
            .strip_prefix(&url.remote_path)
            .unwrap_or(&file_info.path)
            .trim_start_matches('/');
        let local_path = output.join(relative);

        if let Some(parent) = local_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let (scp_remote, final_local) = if !no_gzip
            && !file_info.path.ends_with(".gz")
            && file_info.size_bytes > GZIP_THRESHOLD
        {
            info!(
                "File {} is {}, compressing on remote first",
                file_info.path,
                format_size(file_info.size_bytes)
            );
            ssh_exec(&url.host, &format!("gzip -kf '{}'", file_info.path))?;
            let gz_remote = format!("{}.gz", file_info.path);
            let gz_local = ensure_gz_extension(&local_path);
            (gz_remote, gz_local)
        } else {
            (file_info.path.clone(), local_path)
        };

        let scp_source = format!("{}:{}", url.host, scp_remote);
        scp_download(&scp_source, &final_local)?;
    }

    info!("Downloaded {} file(s) to {}", files.len(), output.display());
    Ok(())
}

/// Ensure a path ends with `.gz`, appending it only if not already present.
/// e.g. `foo.json` -> `foo.json.gz`, `foo.json.gz` -> `foo.json.gz`
fn ensure_gz_extension(path: &Path) -> PathBuf {
    if path.extension().is_some_and(|ext| ext == "gz") {
        path.to_path_buf()
    } else {
        let mut s = path.as_os_str().to_os_string();
        s.push(".gz");
        PathBuf::from(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssh_url_parse_valid() {
        let url = SshUrl::parse("ssh://myhost:/data/traces/trace.json").unwrap();
        assert_eq!(url.host, "myhost");
        assert_eq!(url.remote_path, "/data/traces/trace.json");
    }

    #[test]
    fn test_ssh_url_parse_host_alias() {
        let url = SshUrl::parse("ssh://runpod_b200:/home/user/trace.json").unwrap();
        assert_eq!(url.host, "runpod_b200");
        assert_eq!(url.remote_path, "/home/user/trace.json");
    }

    #[test]
    fn test_ssh_url_parse_missing_prefix() {
        assert!(SshUrl::parse("http://host:/path").is_err());
    }

    #[test]
    fn test_ssh_url_parse_missing_colon() {
        assert!(SshUrl::parse("ssh://host/path").is_err());
    }

    #[test]
    fn test_ssh_url_parse_empty_host() {
        assert!(SshUrl::parse("ssh://:/path").is_err());
    }

    #[test]
    fn test_ssh_url_parse_empty_path() {
        assert!(SshUrl::parse("ssh://host:").is_err());
    }

    #[test]
    fn test_ssh_url_to_scp_source() {
        let url = SshUrl {
            host: "myhost".to_string(),
            remote_path: "/data/trace.json".to_string(),
        };
        assert_eq!(url.to_scp_source(), "myhost:/data/trace.json");
    }

    #[test]
    fn test_resolve_single_output_rename() {
        let result = resolve_single_output("/remote/a.json", Path::new("local/b.json"));
        assert_eq!(result, PathBuf::from("local/b.json"));
    }

    #[test]
    fn test_resolve_single_output_trailing_slash() {
        let result = resolve_single_output("/remote/a.json", Path::new("local/dir/"));
        assert_eq!(result, PathBuf::from("local/dir/a.json"));
    }

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(500), "500 B");
    }

    #[test]
    fn test_format_size_kb() {
        assert_eq!(format_size(2048), "2.0 KB");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(format_size(1_500_000), "1.4 MB");
    }

    #[test]
    fn test_format_size_gb() {
        assert_eq!(format_size(2_147_483_648), "2.0 GB");
    }

    #[test]
    fn test_parse_find_printf_single() {
        let output = "1073741824\t2026-03-20 14:30\t/data/trace.json\n";
        let files = parse_find_printf_output(output).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "/data/trace.json");
        assert_eq!(files[0].size_bytes, 1_073_741_824);
        assert_eq!(files[0].modified, "2026-03-20 14:30");
    }

    #[test]
    fn test_parse_find_printf_output() {
        let output = "1024\t2026-03-20 14:30\t/data/trace1.json\n5678\t2026-03-19 10:00\t/data/trace2.json.gz\n";
        let files = parse_find_printf_output(output).unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].path, "/data/trace1.json");
        assert_eq!(files[0].size_bytes, 1024);
        assert_eq!(files[1].path, "/data/trace2.json.gz");
        assert_eq!(files[1].size_bytes, 5678);
    }

    #[test]
    fn test_parse_find_printf_output_empty() {
        let files = parse_find_printf_output("").unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn test_ensure_gz_extension() {
        assert_eq!(
            ensure_gz_extension(Path::new("trace.json")),
            PathBuf::from("trace.json.gz")
        );
        assert_eq!(
            ensure_gz_extension(Path::new("/data/dir/file.json")),
            PathBuf::from("/data/dir/file.json.gz")
        );
    }
}
