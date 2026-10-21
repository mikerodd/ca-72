//! The CA-72's audio engine apart from any plug-in format: the circuit model's voice in its
//! real-time quality, played by MIDI, its EXTERNAL INPUT the side chain, with POLY, UNISON,
//! DOUBLE, SPREAD and INNER, DRIVE and AUTO GAIN. The nih-plug (VST3/CLAP) and LV2 plug-ins
//! each drive it through [`engine`]; nothing here names a host.

pub mod character;
pub mod drive;
pub mod engine;
pub mod helper;
pub mod pool;
