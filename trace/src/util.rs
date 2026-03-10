use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Read file content, automatically decompressing gzip if detected.
pub fn read_maybe_gzipped(path: &Path) -> Result<String> {
    let data = std::fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;

    // Check for gzip magic bytes
    if data.len() >= 2 && data[0] == 0x1f && data[1] == 0x8b {
        let mut decoder = GzDecoder::new(&data[..]);
        let mut content = String::new();
        decoder
            .read_to_string(&mut content)
            .with_context(|| format!("Failed to decompress gzip file {}", path.display()))?;
        Ok(content)
    } else {
        String::from_utf8(data)
            .with_context(|| format!("File {} is not valid UTF-8", path.display()))
    }
}

/// Validate that all input trace files exist and are files.
pub fn validate_trace_files(paths: &[PathBuf]) -> Result<()> {
    for path in paths {
        anyhow::ensure!(path.exists(), "Trace file not found: {}", path.display());
        anyhow::ensure!(path.is_file(), "Not a file: {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_read_plain_file() {
        let dir = std::env::temp_dir().join("trace_test_plain");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.json");
        std::fs::write(&path, r#"{"traceEvents":[]}"#).unwrap();

        let content = read_maybe_gzipped(&path).unwrap();
        assert_eq!(content, r#"{"traceEvents":[]}"#);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_read_gzipped_file() {
        let dir = std::env::temp_dir().join("trace_test_gz");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.json.gz");

        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(b"hello gzip").unwrap();
        let compressed = encoder.finish().unwrap();
        std::fs::write(&path, &compressed).unwrap();

        let content = read_maybe_gzipped(&path).unwrap();
        assert_eq!(content, "hello gzip");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_validate_missing_file() {
        let result = validate_trace_files(&[PathBuf::from("/nonexistent/file.json")]);
        assert!(result.is_err());
    }
}
