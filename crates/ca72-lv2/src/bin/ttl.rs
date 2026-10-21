//! Write the CA-72's LV2 bundle TTLs (`manifest.ttl`, `dsp.ttl`) into a directory, from the
//! port table: `ca72-ttl [DIR]` (default `CA-72.lv2`). The binary the manifest names is copied
//! there by hand (`install.sh`); its name is the constant [`ca72_lv2::ttl::BINARY`].

use std::path::PathBuf;
use std::process::ExitCode;

use ca72_lv2::ttl::{write_bundle, BINARY};

fn main() -> ExitCode {
    let dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("CA-72.lv2"));
    match write_bundle(&dir, BINARY) {
        Ok(()) => {
            eprintln!("wrote {} and {}", dir.join("manifest.ttl").display(), dir.join("dsp.ttl").display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("ca72-ttl: {e}");
            ExitCode::FAILURE
        }
    }
}
