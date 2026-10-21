//! The bundle's `manifest.ttl` and `dsp.ttl`, generated from the port table ([`crate::PORTS`])
//! so the two cannot drift: the plug-in's `run` reads those ports by index, and the host learns
//! of them only from here. `bin/ttl.rs` writes the two files.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::Path;

use crate::params::{Kind, PortDef};
use crate::{PLUGIN_URI, PORT_CONTROL_BASE, PORTS};

/// The binary's file name inside the bundle (`manifest.ttl`'s `lv2:binary`).
pub const BINARY: &str = "ca72.so";

/// The bundle's `manifest.ttl`: the one plug-in, its binary and its `dsp.ttl`.
pub fn manifest_ttl(binary: &str) -> String {
    let mut s = String::new();
    s.push_str("@prefix lv2:  <http://lv2plug.in/ns/lv2core#> .\n");
    s.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\n");
    let _ = writeln!(s, "<{PLUGIN_URI}>");
    s.push_str("    a lv2:Plugin ;\n");
    let _ = writeln!(s, "    lv2:binary <{binary}> ;");
    s.push_str("    rdfs:seeAlso <dsp.ttl> .\n");
    s
}

/// The bundle's `dsp.ttl`: the plug-in's class, features and every port.
pub fn dsp_ttl() -> String {
    let mut s = String::new();
    s.push_str("@prefix atom:  <http://lv2plug.in/ns/ext/atom#> .\n");
    s.push_str("@prefix doap:  <http://usefulinc.com/ns/doap#> .\n");
    s.push_str("@prefix state: <http://lv2plug.in/ns/ext/state#> .\n");
    s.push_str("@prefix lv2:   <http://lv2plug.in/ns/lv2core#> .\n");
    s.push_str("@prefix midi:  <http://lv2plug.in/ns/ext/midi#> .\n");
    s.push_str("@prefix pg:    <http://lv2plug.in/ns/ext/port-groups#> .\n");
    s.push_str("@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n");
    s.push_str("@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .\n");
    s.push_str("@prefix units: <http://lv2plug.in/ns/ext/units#> .\n");
    s.push_str("@prefix urid:  <http://lv2plug.in/ns/ext/urid#> .\n");
    s.push('\n');

    let groups = groups();
    for (label, node) in &groups {
        let _ = writeln!(s, "_:{node} a pg:Group ; rdfs:label \"{label}\" .");
    }
    s.push('\n');

    let _ = writeln!(s, "<{PLUGIN_URI}>");
    s.push_str("    a lv2:Plugin , lv2:InstrumentPlugin ;\n");
    s.push_str("    doap:name \"CA-72\" ;\n");
    s.push_str("    doap:license <https://spdx.org/licenses/GPL-3.0-or-later.html> ;\n");
    s.push_str("    lv2:optionalFeature lv2:hardRTCapable ;\n");
    s.push_str("    lv2:requiredFeature urid:map ;\n");
    s.push_str("    lv2:extensionData state:interface ;\n");

    // The audio and MIDI ports.
    let _ = writeln!(
        s,
        "    lv2:port [\n        a lv2:InputPort , lv2:AudioPort ;\n        \
         lv2:index {} ;\n        lv2:symbol \"in_l\" ;\n        \
         lv2:name \"External Input L\" ;\n        \
         lv2:connectionOptional true ;\n        pg:group _:g_mixer ;\n    ] ,",
        crate::PORT_IN_L
    );
    let _ = writeln!(
        s,
        "    [\n        a lv2:InputPort , lv2:AudioPort ;\n        \
         lv2:index {} ;\n        lv2:symbol \"in_r\" ;\n        \
         lv2:name \"External Input R\" ;\n        \
         lv2:connectionOptional true ;\n        pg:group _:g_mixer ;\n    ] ,",
        crate::PORT_IN_R
    );
    let _ = writeln!(
        s,
        "    [\n        a lv2:OutputPort , lv2:AudioPort ;\n        \
         lv2:index {} ;\n        lv2:symbol \"out_l\" ;\n        \
         lv2:name \"Output L\" ;\n    ] ,",
        crate::PORT_OUT_L
    );
    let _ = writeln!(
        s,
        "    [\n        a lv2:OutputPort , lv2:AudioPort ;\n        \
         lv2:index {} ;\n        lv2:symbol \"out_r\" ;\n        \
         lv2:name \"Output R\" ;\n    ] ,",
        crate::PORT_OUT_R
    );
    let _ = writeln!(
        s,
        "    [\n        a lv2:InputPort , atom:AtomPort ;\n        \
         lv2:index {} ;\n        lv2:symbol \"midi\" ;\n        \
         lv2:name \"MIDI Input\" ;\n        lv2:designation lv2:control ;\n        \
         atom:bufferType atom:Sequence ;\n        \
         atom:supports midi:MidiEvent ;\n    ] ,",
        crate::PORT_MIDI
    );

    // The control ports.
    let last = PORTS.len() - 1;
    for (i, p) in PORTS.iter().enumerate() {
        let index = PORT_CONTROL_BASE as usize + i;
        let terminator = if i == last { " ." } else { " ," };
        let mut body = String::new();
        let _ = write!(
            body,
            "    [\n        a lv2:InputPort , lv2:ControlPort ;\n        \
             lv2:index {index} ;\n        lv2:symbol \"{}\" ;\n        \
             lv2:name \"{}\" ;\n        lv2:default {} ;\n        \
             lv2:minimum {} ;\n        lv2:maximum {} ;",
            p.symbol, p.name, p.default, p.min, p.max
        );
        if let Some(unit) = unit_of(p) {
            let _ = write!(body, "\n        units:unit {unit} ;");
        }
        match p.kind {
            Kind::Float => {}
            Kind::Integer => {
                let _ = write!(body, "\n        lv2:portProperty lv2:integer ;");
            }
            Kind::Toggle | Kind::Choice => {
                let _ = write!(
                    body,
                    "\n        lv2:portProperty lv2:integer , lv2:enumeration ;"
                );
            }
        }
        for (label, value) in p.points {
            let _ = write!(
                body,
                "\n        lv2:scalePoint [ rdfs:label \"{label}\" ; rdf:value {value} ] ;"
            );
        }
        let _ = write!(body, "\n        pg:group _:{} ;\n    ]{terminator}\n", node_of(p.group));
        s.push_str(&body);
    }

    s.push('\n');
    s
}

/// Write the bundle's two TTLs into `dir` (created if need be).
pub fn write_bundle(dir: &Path, binary: &str) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    fs::write(dir.join("manifest.ttl"), manifest_ttl(binary))?;
    fs::write(dir.join("dsp.ttl"), dsp_ttl())?;
    Ok(())
}

/// The port groups the table names, in the order first seen: (label, node).
fn groups() -> Vec<(&'static str, &'static str)> {
    let mut out: Vec<(&'static str, &'static str)> = Vec::new();
    for p in &PORTS {
        if !out.iter().any(|(label, _)| *label == p.group) {
            out.push((p.group, node_of(p.group)));
        }
    }
    out
}

/// A group's blank-node label: its name lower-cased, runs of non-alphanumerics an underscore.
fn node_of(group: &'static str) -> &'static str {
    match group {
        "Power" => "g_power",
        "Controllers" => "g_controllers",
        "Oscillator Bank" => "g_oscillator_bank",
        "Mixer" => "g_mixer",
        "Modifiers" => "g_modifiers",
        "Output" => "g_output",
        "Left Hand Controller" => "g_left_hand_controller",
        "MIDI" => "g_midi",
        "General" => "g_general",
        _ => "g_general",
    }
}

/// A port's unit, from its label, when the units extension names it.
fn unit_of(p: &PortDef) -> Option<&'static str> {
    match p.unit {
        " dB" => Some("units:db"),
        " %" => Some("units:pc"),
        " semitones" => Some("units:semitone12TET"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every port appears once, by index, and the table ends the plug-in's port list.
    #[test]
    fn every_port_is_declared_once() {
        let ttl = dsp_ttl();
        for i in 0..crate::PORT_COUNT {
            assert!(
                ttl.contains(&format!("lv2:index {i} ;")),
                "port {i} missing from dsp.ttl"
            );
        }
        assert!(ttl.trim_end().ends_with('.'), "the port list is not terminated");
    }

    /// The choices and switches carry their positions, the named groups a node.
    #[test]
    fn choices_and_groups_are_declared() {
        let ttl = dsp_ttl();
        assert!(ttl.contains("lv2:scalePoint [ rdfs:label \"Sawtooth\" ; rdf:value 2 ]"));
        assert!(ttl.contains("lv2:enumeration"));
        for (label, node) in groups() {
            assert!(
                ttl.contains(&format!("_:{node} a pg:Group ; rdfs:label \"{label}\"")),
                "group {label} missing"
            );
        }
    }

    /// The manifest names the binary and the data file.
    #[test]
    fn the_manifest_points_at_the_bundle() {
        let m = manifest_ttl("ca72.so");
        assert!(m.contains(&format!("<{PLUGIN_URI}>")));
        assert!(m.contains("lv2:binary <ca72.so>"));
        assert!(m.contains("rdfs:seeAlso <dsp.ttl>"));
    }

    /// The plug-in is a MIDI instrument with a name, so a host lists and plays it.
    #[test]
    fn the_plugin_is_a_named_instrument() {
        let ttl = dsp_ttl();
        assert!(ttl.contains("a lv2:Plugin , lv2:InstrumentPlugin ;"));
        assert!(ttl.contains("doap:name \"CA-72\""));
        assert!(ttl.contains("lv2:designation lv2:control"));
        assert!(ttl.contains("atom:supports midi:MidiEvent"));
    }
}
