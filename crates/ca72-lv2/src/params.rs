//! The LV2 control ports: the plug-in's parameters as one table ([`PORTS`]), which the
//! plug-in's `run` reads (by port index) and the bundle's `dsp.ttl` is generated from
//! (`bin/ttl.rs`). One table, so the two cannot drift.
//!
//! The symbols and the values are the nih-plug plug-in's own (`ca72-plugin`'s `params.rs`):
//! each port carries the parameter's *plain* value (its dial's units, a choice by its index,
//! a switch 0 or 1), so the factory presets (`ca72-plugin/sounds/presets.toml`) can be turned
//! into LV2 presets unchanged, and the same [`controls_from`] conversions serve.

use ca72::modulation::PITCH_WHEEL_SEMITONES;
use ca72::tuning::{FREQ_CENTRE, Range};
use ca72::voice::{feedback_law, ContourKnobs, OscPanel, Panel, Priority, Quality, Waveform};
use ca72_engine::character::Placement;
use ca72_engine::engine::{Controls, POLY_VOICES};

/// DOUBLE on at DETUNE's 0, as the engine is told it (`ca72-plugin`'s `params.rs`): a share of
/// DETUNE's travel so small the pair play as one pitch.
const DOUBLED_AT_NONE: f64 = 1e-6;

/// A control port's parameter, in the table's order. The order is the discriminant order: the
/// table is indexed by it (`Control as usize`), checked in the tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    Bypass,
    Tune,
    Glide,
    ModMix,
    OscMod,
    Osc3Control,
    Osc1Range,
    Osc1Waveform,
    Osc2Range,
    Osc2Frequency,
    Osc2Waveform,
    Osc3Range,
    Osc3Frequency,
    Osc3Waveform,
    Osc1On,
    Osc1Volume,
    ExtOn,
    ExtVolume,
    Osc2On,
    Osc2Volume,
    NoiseOn,
    NoiseVolume,
    NoiseType,
    Osc3On,
    Osc3Volume,
    FilterMode,
    FilterMod,
    KeyboardControl1,
    KeyboardControl2,
    Cutoff,
    Emphasis,
    ContourAmount,
    FilterAttack,
    FilterDecay,
    FilterSustain,
    LoudnessAttack,
    LoudnessDecay,
    LoudnessSustain,
    Volume,
    MainOutput,
    A440,
    GlideOn,
    DecayOn,
    PitchWheel,
    ModWheel,
    MidiBendRange,
    Poly,
    Voices,
    Entropy,
    Spread,
    Feedback,
    Lock,
    Placement,
    Unison,
    Double,
    Drive,
    Level,
    AutoGain,
    Inner,
    Doubled,
    Quality,
    NotePriority,
    Retrigger,
}

/// How a port's value is shown and rounded: a plain float, a whole number, a two-position
/// switch, or a choice among named positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Float,
    Integer,
    Toggle,
    Choice,
}

/// One control port.
#[derive(Debug, Clone, Copy)]
pub struct PortDef {
    pub key: Control,
    pub symbol: &'static str,
    pub name: &'static str,
    pub group: &'static str,
    pub kind: Kind,
    pub min: f32,
    pub max: f32,
    pub default: f32,
    pub unit: &'static str,
    /// A choice's or a switch's positions, in value order.
    pub points: &'static [(&'static str, f32)],
}

const OFF_ON: &[(&str, f32)] = &[("Off", 0.0), ("On", 1.0)];
const FOOTAGE: &[(&str, f32)] = &[
    ("LO", 0.0),
    ("32'", 1.0),
    ("16'", 2.0),
    ("8'", 3.0),
    ("4'", 4.0),
    ("2'", 5.0),
];
const WAVE: &[(&str, f32)] = &[
    ("Triangle", 0.0),
    ("Shark Tooth", 1.0),
    ("Sawtooth", 2.0),
    ("Square", 3.0),
    ("Wide Rectangle", 4.0),
    ("Narrow Rectangle", 5.0),
];
const WAVE3: &[(&str, f32)] = &[
    ("Triangle", 0.0),
    ("Reverse Sawtooth", 1.0),
    ("Sawtooth", 2.0),
    ("Square", 3.0),
    ("Wide Rectangle", 4.0),
    ("Narrow Rectangle", 5.0),
];
const NOISE: &[(&str, f32)] = &[("White", 0.0), ("Pink", 1.0)];
const FILTER_MODE: &[(&str, f32)] = &[("LO", 0.0), ("HI", 1.0)];
const SCATTER: &[(&str, f32)] = &[("EVEN", 0.0), ("CENTER", 1.0)];
const QUALITY: &[(&str, f32)] = &[("LO", 0.0), ("HI", 1.0), ("ULTRA", 2.0)];
const PRIORITY: &[(&str, f32)] = &[("LOW", 0.0), ("LAST", 1.0), ("HIGH", 2.0)];

/// The control ports, in port order after the audio and MIDI ports (`lib.rs`'s indices). The
/// symbols and plain-value ranges are the nih-plug plug-in's.
pub const PORTS: [PortDef; CONTROL_COUNT] = [
    PortDef { key: Control::Bypass, symbol: "bypass", name: "Bypass", group: "Power", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 0.0, unit: "", points: OFF_ON },
    PortDef { key: Control::Tune, symbol: "tune", name: "Tune", group: "Controllers", kind: Kind::Float, min: -2.5, max: 2.5, default: 0.0, unit: "", points: &[] },
    PortDef { key: Control::Glide, symbol: "glide", name: "Glide", group: "Controllers", kind: Kind::Float, min: 0.0, max: 10.0, default: 0.0, unit: "", points: &[] },
    PortDef { key: Control::ModMix, symbol: "mod_mix", name: "Modulation Mix", group: "Controllers", kind: Kind::Float, min: 0.0, max: 10.0, default: 0.0, unit: "", points: &[] },
    PortDef { key: Control::OscMod, symbol: "osc_mod", name: "Oscillator Modulation", group: "Controllers", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 0.0, unit: "", points: OFF_ON },
    PortDef { key: Control::Osc3Control, symbol: "osc3_control", name: "Osc 3 Control", group: "Controllers", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 1.0, unit: "", points: OFF_ON },
    PortDef { key: Control::Osc1Range, symbol: "osc1_range", name: "Osc 1 Range", group: "Oscillator Bank", kind: Kind::Choice, min: 0.0, max: 5.0, default: 3.0, unit: "", points: FOOTAGE },
    PortDef { key: Control::Osc1Waveform, symbol: "osc1_waveform", name: "Osc 1 Waveform", group: "Oscillator Bank", kind: Kind::Choice, min: 0.0, max: 5.0, default: 2.0, unit: "", points: WAVE },
    PortDef { key: Control::Osc2Range, symbol: "osc2_range", name: "Osc 2 Range", group: "Oscillator Bank", kind: Kind::Choice, min: 0.0, max: 5.0, default: 3.0, unit: "", points: FOOTAGE },
    PortDef { key: Control::Osc2Frequency, symbol: "osc2_frequency", name: "Osc 2 Frequency", group: "Oscillator Bank", kind: Kind::Float, min: -7.5, max: 7.5, default: 0.0, unit: "", points: &[] },
    PortDef { key: Control::Osc2Waveform, symbol: "osc2_waveform", name: "Osc 2 Waveform", group: "Oscillator Bank", kind: Kind::Choice, min: 0.0, max: 5.0, default: 2.0, unit: "", points: WAVE },
    PortDef { key: Control::Osc3Range, symbol: "osc3_range", name: "Osc 3 Range", group: "Oscillator Bank", kind: Kind::Choice, min: 0.0, max: 5.0, default: 3.0, unit: "", points: FOOTAGE },
    PortDef { key: Control::Osc3Frequency, symbol: "osc3_frequency", name: "Osc 3 Frequency", group: "Oscillator Bank", kind: Kind::Float, min: -7.5, max: 7.5, default: 0.0, unit: "", points: &[] },
    PortDef { key: Control::Osc3Waveform, symbol: "osc3_waveform", name: "Osc 3 Waveform", group: "Oscillator Bank", kind: Kind::Choice, min: 0.0, max: 5.0, default: 2.0, unit: "", points: WAVE3 },
    PortDef { key: Control::Osc1On, symbol: "osc1_on", name: "Osc 1 On", group: "Mixer", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 1.0, unit: "", points: OFF_ON },
    PortDef { key: Control::Osc1Volume, symbol: "osc1_volume", name: "Osc 1 Volume", group: "Mixer", kind: Kind::Float, min: 0.0, max: 10.0, default: 8.0, unit: "", points: &[] },
    PortDef { key: Control::ExtOn, symbol: "ext_on", name: "External Input On", group: "Mixer", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 0.0, unit: "", points: OFF_ON },
    PortDef { key: Control::ExtVolume, symbol: "ext_volume", name: "External Input Volume", group: "Mixer", kind: Kind::Float, min: 0.0, max: 10.0, default: 5.0, unit: "", points: &[] },
    PortDef { key: Control::Osc2On, symbol: "osc2_on", name: "Osc 2 On", group: "Mixer", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 0.0, unit: "", points: OFF_ON },
    PortDef { key: Control::Osc2Volume, symbol: "osc2_volume", name: "Osc 2 Volume", group: "Mixer", kind: Kind::Float, min: 0.0, max: 10.0, default: 8.0, unit: "", points: &[] },
    PortDef { key: Control::NoiseOn, symbol: "noise_on", name: "Noise On", group: "Mixer", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 0.0, unit: "", points: OFF_ON },
    PortDef { key: Control::NoiseVolume, symbol: "noise_volume", name: "Noise Volume", group: "Mixer", kind: Kind::Float, min: 0.0, max: 10.0, default: 8.0, unit: "", points: &[] },
    PortDef { key: Control::NoiseType, symbol: "noise_type", name: "Noise Color", group: "Mixer", kind: Kind::Choice, min: 0.0, max: 1.0, default: 0.0, unit: "", points: NOISE },
    PortDef { key: Control::Osc3On, symbol: "osc3_on", name: "Osc 3 On", group: "Mixer", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 0.0, unit: "", points: OFF_ON },
    PortDef { key: Control::Osc3Volume, symbol: "osc3_volume", name: "Osc 3 Volume", group: "Mixer", kind: Kind::Float, min: 0.0, max: 10.0, default: 8.0, unit: "", points: &[] },
    PortDef { key: Control::FilterMode, symbol: "filter_mode", name: "Filter Mode", group: "Modifiers", kind: Kind::Choice, min: 0.0, max: 1.0, default: 0.0, unit: "", points: FILTER_MODE },
    PortDef { key: Control::FilterMod, symbol: "filter_mod", name: "Filter Modulation", group: "Modifiers", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 0.0, unit: "", points: OFF_ON },
    PortDef { key: Control::KeyboardControl1, symbol: "keyboard_control_1", name: "Keyboard Control 1", group: "Modifiers", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 1.0, unit: "", points: OFF_ON },
    PortDef { key: Control::KeyboardControl2, symbol: "keyboard_control_2", name: "Keyboard Control 2", group: "Modifiers", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 0.0, unit: "", points: OFF_ON },
    PortDef { key: Control::Cutoff, symbol: "cutoff", name: "Cutoff Frequency", group: "Modifiers", kind: Kind::Float, min: -5.0, max: 5.0, default: 0.0, unit: "", points: &[] },
    PortDef { key: Control::Emphasis, symbol: "emphasis", name: "Emphasis", group: "Modifiers", kind: Kind::Float, min: 0.0, max: 10.0, default: 0.0, unit: "", points: &[] },
    PortDef { key: Control::ContourAmount, symbol: "contour_amount", name: "Amount of Contour", group: "Modifiers", kind: Kind::Float, min: 0.0, max: 10.0, default: 5.0, unit: "", points: &[] },
    PortDef { key: Control::FilterAttack, symbol: "filter_attack", name: "Filter Attack Time", group: "Modifiers", kind: Kind::Float, min: 0.0, max: 10.0, default: 0.0, unit: "", points: &[] },
    PortDef { key: Control::FilterDecay, symbol: "filter_decay", name: "Filter Decay Time", group: "Modifiers", kind: Kind::Float, min: 0.0, max: 10.0, default: 4.0, unit: "", points: &[] },
    PortDef { key: Control::FilterSustain, symbol: "filter_sustain", name: "Filter Sustain Level", group: "Modifiers", kind: Kind::Float, min: 0.0, max: 10.0, default: 5.0, unit: "", points: &[] },
    PortDef { key: Control::LoudnessAttack, symbol: "loudness_attack", name: "Loudness Attack Time", group: "Modifiers", kind: Kind::Float, min: 0.0, max: 10.0, default: 0.0, unit: "", points: &[] },
    PortDef { key: Control::LoudnessDecay, symbol: "loudness_decay", name: "Loudness Decay Time", group: "Modifiers", kind: Kind::Float, min: 0.0, max: 10.0, default: 4.0, unit: "", points: &[] },
    PortDef { key: Control::LoudnessSustain, symbol: "loudness_sustain", name: "Loudness Sustain Level", group: "Modifiers", kind: Kind::Float, min: 0.0, max: 10.0, default: 8.0, unit: "", points: &[] },
    PortDef { key: Control::Volume, symbol: "volume", name: "Volume", group: "Output", kind: Kind::Float, min: 0.0, max: 10.0, default: 10.0, unit: "", points: &[] },
    PortDef { key: Control::MainOutput, symbol: "main_output", name: "Main Output", group: "Output", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 1.0, unit: "", points: OFF_ON },
    PortDef { key: Control::A440, symbol: "a440", name: "A-440", group: "Output", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 0.0, unit: "", points: OFF_ON },
    PortDef { key: Control::GlideOn, symbol: "glide_on", name: "Glide Switch", group: "Left Hand Controller", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 0.0, unit: "", points: OFF_ON },
    PortDef { key: Control::DecayOn, symbol: "decay_on", name: "Decay Switch", group: "Left Hand Controller", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 1.0, unit: "", points: OFF_ON },
    PortDef { key: Control::PitchWheel, symbol: "pitch_wheel", name: "Pitch Wheel", group: "Left Hand Controller", kind: Kind::Float, min: -1.0, max: 1.0, default: 0.0, unit: "", points: &[] },
    PortDef { key: Control::ModWheel, symbol: "mod_wheel", name: "Modulation Wheel", group: "Left Hand Controller", kind: Kind::Float, min: 0.0, max: 1.0, default: 0.0, unit: "", points: &[] },
    PortDef { key: Control::MidiBendRange, symbol: "midi_bend_range", name: "MIDI Bend Range", group: "MIDI", kind: Kind::Float, min: 0.0, max: PITCH_WHEEL_SEMITONES as f32, default: 2.0, unit: " semitones", points: &[] },
    PortDef { key: Control::Poly, symbol: "poly", name: "Poly", group: "General", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 0.0, unit: "", points: OFF_ON },
    PortDef { key: Control::Voices, symbol: "voices", name: "Voices", group: "General", kind: Kind::Integer, min: POLY_VOICES.0 as f32, max: POLY_VOICES.2 as f32, default: POLY_VOICES.1 as f32, unit: "", points: &[] },
    PortDef { key: Control::Entropy, symbol: "entropy", name: "Entropy", group: "General", kind: Kind::Float, min: 0.0, max: 100.0, default: 0.0, unit: " %", points: &[] },
    PortDef { key: Control::Spread, symbol: "spread", name: "Width", group: "General", kind: Kind::Float, min: 0.0, max: 100.0, default: 0.0, unit: " %", points: &[] },
    PortDef { key: Control::Feedback, symbol: "feedback", name: "Feedback", group: "General", kind: Kind::Float, min: 0.0, max: 10.0, default: 0.0, unit: "", points: &[] },
    PortDef { key: Control::Lock, symbol: "lock", name: "Lock (oscillators identical)", group: "General", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 0.0, unit: "", points: OFF_ON },
    PortDef { key: Control::Placement, symbol: "placement", name: "Scatter Placement", group: "General", kind: Kind::Choice, min: 0.0, max: 1.0, default: 0.0, unit: "", points: SCATTER },
    PortDef { key: Control::Unison, symbol: "unison", name: "Unison", group: "General", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 0.0, unit: "", points: OFF_ON },
    PortDef { key: Control::Double, symbol: "double", name: "Double Detune", group: "General", kind: Kind::Float, min: 0.0, max: 100.0, default: 0.0, unit: "", points: &[] },
    PortDef { key: Control::Drive, symbol: "drive", name: "Drive", group: "General", kind: Kind::Float, min: 0.0, max: 24.0, default: 0.0, unit: " dB", points: &[] },
    PortDef { key: Control::Level, symbol: "level", name: "Level", group: "General", kind: Kind::Float, min: -30.0, max: 12.0, default: 0.0, unit: " dB", points: &[] },
    PortDef { key: Control::AutoGain, symbol: "auto_gain", name: "Auto Gain", group: "General", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 1.0, unit: "", points: OFF_ON },
    PortDef { key: Control::Inner, symbol: "inner", name: "Inner", group: "General", kind: Kind::Float, min: 0.0, max: 100.0, default: 0.0, unit: " %", points: &[] },
    PortDef { key: Control::Doubled, symbol: "doubled", name: "Double", group: "General", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 0.0, unit: "", points: OFF_ON },
    PortDef { key: Control::Quality, symbol: "quality", name: "Quality", group: "General", kind: Kind::Choice, min: 0.0, max: 2.0, default: 0.0, unit: "", points: QUALITY },
    PortDef { key: Control::NotePriority, symbol: "note_priority", name: "Note Priority", group: "General", kind: Kind::Choice, min: 0.0, max: 2.0, default: 0.0, unit: "", points: PRIORITY },
    PortDef { key: Control::Retrigger, symbol: "retrigger", name: "Retrigger", group: "General", kind: Kind::Toggle, min: 0.0, max: 1.0, default: 0.0, unit: "", points: OFF_ON },
];

/// How many control ports there are.
pub const CONTROL_COUNT: usize = 63;

/// The engine's controls from the ports' values, each indexed by [`Control`] (the table's
/// order). The conversions are `ca72-plugin`'s `controls_with`, without MIDI Learn (the host
/// does that): a knob's dial units in, the voice's 0..1 out ([`Controls`]).
pub fn controls_from(v: &[f32]) -> Controls {
    let g = |k: Control| f64::from(v[k as usize]);
    let ten = |k: Control| g(k) / 10.0;
    let freq = |k: Control| (g(k) + 7.5) / 15.0;
    let on = |k: Control| v[k as usize] >= 0.5;
    let osc = [
        OscPanel {
            range: range_at(v[Control::Osc1Range as usize]),
            waveform: wave_at(v[Control::Osc1Waveform as usize]),
            on: on(Control::Osc1On),
            volume: ten(Control::Osc1Volume),
            freq: FREQ_CENTRE,
        },
        OscPanel {
            range: range_at(v[Control::Osc2Range as usize]),
            waveform: wave_at(v[Control::Osc2Waveform as usize]),
            on: on(Control::Osc2On),
            volume: ten(Control::Osc2Volume),
            freq: freq(Control::Osc2Frequency),
        },
        OscPanel {
            range: range_at(v[Control::Osc3Range as usize]),
            waveform: wave3_at(v[Control::Osc3Waveform as usize]),
            on: on(Control::Osc3On),
            volume: ten(Control::Osc3Volume),
            freq: freq(Control::Osc3Frequency),
        },
    ];
    let panel = Panel {
        osc,
        osc3_control: on(Control::Osc3Control),
        cutoff: (g(Control::Cutoff) + 5.0) / 10.0,
        emphasis: ten(Control::Emphasis),
        contour_amount: ten(Control::ContourAmount),
        keyboard_control_1: on(Control::KeyboardControl1),
        keyboard_control_2: on(Control::KeyboardControl2),
        filter_contour: ContourKnobs {
            attack: ten(Control::FilterAttack),
            decay: ten(Control::FilterDecay),
            sustain: ten(Control::FilterSustain),
        },
        loudness_contour: ContourKnobs {
            attack: ten(Control::LoudnessAttack),
            decay: ten(Control::LoudnessDecay),
            sustain: ten(Control::LoudnessSustain),
        },
        glide: ten(Control::Glide),
        glide_on: on(Control::GlideOn),
        decay: on(Control::DecayOn),
        noise_on: on(Control::NoiseOn),
        noise_volume: ten(Control::NoiseVolume),
        noise_pink: v[Control::NoiseType as usize].round() as i32 == 1,
        mod_mix: ten(Control::ModMix),
        osc_mod: on(Control::OscMod),
        filter_mod: on(Control::FilterMod),
        filter_hi: v[Control::FilterMode as usize].round() as i32 == 1,
        pitch_wheel: g(Control::PitchWheel),
        mod_wheel: g(Control::ModWheel),
        ext_volume: ten(Control::ExtVolume),
        ext_on: on(Control::ExtOn),
        a440: on(Control::A440),
        tune: g(Control::Tune) / 2.5,
        quality: Quality::default(),
        priority: priority_at(v[Control::NotePriority as usize]),
        retrigger: on(Control::Retrigger),
    };
    let quality = v[Control::Quality as usize].round() as i32;
    Controls {
        panel,
        volume: ten(Control::Volume),
        main_output: on(Control::MainOutput),
        bend_range: g(Control::MidiBendRange),
        power: !on(Control::Bypass),
        poly: on(Control::Poly),
        voices: usize::try_from(v[Control::Voices as usize].round() as i32)
            .unwrap_or(POLY_VOICES.1),
        entropy: g(Control::Entropy) / 100.0,
        spread: g(Control::Spread) / 100.0,
        inner: g(Control::Inner) / 100.0,
        placement: placement_at(v[Control::Placement as usize]),
        unison: on(Control::Unison),
        double: if on(Control::Doubled) {
            (g(Control::Double) / 100.0).max(DOUBLED_AT_NONE)
        } else {
            0.0
        },
        drive: g(Control::Drive),
        level: g(Control::Level),
        auto_gain: on(Control::AutoGain),
        feedback: feedback_law(g(Control::Feedback) / 10.0),
        lock: on(Control::Lock),
        potato: quality == 0,
        ultra: quality == 2,
    }
}

/// RANGE's choice at a port value.
fn range_at(x: f32) -> Range {
    match x.round() as i32 {
        0 => Range::Lo,
        1 => Range::R32,
        2 => Range::R16,
        3 => Range::R8,
        4 => Range::R4,
        _ => Range::R2,
    }
}

/// WAVEFORM's choice on oscillators 1 and 2.
fn wave_at(x: f32) -> Waveform {
    match x.round() as i32 {
        0 => Waveform::Triangle,
        1 => Waveform::SharkTooth,
        2 => Waveform::Sawtooth,
        3 => Waveform::Square,
        4 => Waveform::WideRectangle,
        _ => Waveform::NarrowRectangle,
    }
}

/// WAVEFORM's choice on oscillator 3: the second position the reverse sawtooth.
fn wave3_at(x: f32) -> Waveform {
    match x.round() as i32 {
        0 => Waveform::Triangle,
        1 => Waveform::ReverseSawtooth,
        2 => Waveform::Sawtooth,
        3 => Waveform::Square,
        4 => Waveform::WideRectangle,
        _ => Waveform::NarrowRectangle,
    }
}

/// NOTE PRIORITY's choice.
fn priority_at(x: f32) -> Priority {
    match x.round() as i32 {
        0 => Priority::Low,
        1 => Priority::Last,
        _ => Priority::High,
    }
}

/// SCATTER's placement.
fn placement_at(x: f32) -> Placement {
    match x.round() as i32 {
        1 => Placement::Centre,
        _ => Placement::Even,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table's order is the enum's: values are indexed by `Control as usize`.
    #[test]
    fn the_table_is_in_control_order() {
        for (i, p) in PORTS.iter().enumerate() {
            assert_eq!(p.key as usize, i, "port {i} ({}) out of order", p.symbol);
        }
        assert_eq!(PORTS.len(), CONTROL_COUNT);
    }

    /// No two ports share a symbol (the TTL's port identifiers must be unique).
    #[test]
    fn symbols_are_unique() {
        for (i, a) in PORTS.iter().enumerate() {
            for b in &PORTS[i + 1..] {
                assert_ne!(a.symbol, b.symbol);
            }
        }
    }

    /// The defaults are within their ranges and on a choice's or a switch's position.
    #[test]
    fn defaults_are_in_range() {
        for p in &PORTS {
            assert!(
                p.default >= p.min && p.default <= p.max,
                "{} default {} outside {}..{}",
                p.symbol,
                p.default,
                p.min,
                p.max
            );
            if !p.points.is_empty() {
                assert!(
                    p.points.iter().any(|(_, v)| *v == p.default),
                    "{} default {} is no position",
                    p.symbol,
                    p.default
                );
            }
        }
    }

    /// The defaults through the table build the same controls as the parameter defaults would.
    #[test]
    fn the_defaults_build_a_playing_instrument() {
        let v: Vec<f32> = PORTS.iter().map(|p| p.default).collect();
        let c = controls_from(&v);
        assert!(c.power && c.main_output);
        assert!(c.panel.osc[0].on && !c.panel.osc[1].on && !c.panel.osc[2].on);
        assert_eq!(c.panel.osc[0].range, Range::R8);
        assert_eq!(c.panel.osc[0].waveform, Waveform::Sawtooth);
        assert_eq!(c.voices, POLY_VOICES.1);
        assert!(c.potato && !c.ultra);
    }
}
