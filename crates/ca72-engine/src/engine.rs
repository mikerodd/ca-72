//! The plug-in's audio engine, apart from any plug-in format: the circuit model's voice in its
//! real-time (Potato) quality, played by MIDI notes, pitch bend and the modulation wheel, with
//! the side chain at EXTERNAL INPUT. Once the voices are built nothing here allocates
//! (`tests/realtime.rs`).
//!
//! POLY off, the one instrument takes every key through its keyboard circuit (NOTE PRIORITY's
//! choice of the keys held, LOW by default, the circuit's own; single triggering), as before. UNISON (outranking POLY) plays VOICES whole
//! instruments together, each taking every key through its own keyboard circuit as the one
//! does, each its own parts under ENTROPY and its own place under SPREAD, turned down by the
//! square root of VOICES (decisions.md R45, the CA-74's R25). POLY on, one voice per note, up to VOICES (2 to
//! 10) of the ten built: each the whole instrument with its own keyboard circuit, which its
//! note reaches as one key (`PolyKey`; decisions.md, "POLY"). A POLY note takes the voice
//! that last played its key if that voice is free or letting go, else the free voice whose
//! last note began longest ago, else the one let go longest ago, else the one held longest
//! (the CA-74's R27; decisions.md R45), so a chord struck again stays where it was and a
//! melody goes round the voices. ENTROPY makes each voice its own parts, SPREAD puts the voices
//! across the stereo field where the placement has them (`character.rs`); the one instrument
//! (without DOUBLE) at the next of VOICES's places at each attack, a note played legato staying
//! where its phrase began (decisions.md R62); both 0 by default, the circuit as drawn.

use ca72::light::Light;
use ca72::modulation::PITCH_WHEEL_SEMITONES;
use ca72::resample::{Decimator, Interpolator};
use ca72::voice::{Jacks, Panel, Quality, RETRIGGER_GAP, Voice, audio_taper};

use crate::character::{Character, Placement};
use crate::drive::{AVERAGE, Calibration, Curve, HEARD, HELD, KEYS, KWeighted, STEPS};
use crate::pool::{Ask, Crew, MAX, Pool, Shared};
use plugin_kit_stereo::place::{
    ALONE, DOUBLE_TRIM, glide_share, pair_gains, unison_trim, voice_gains,
};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How far ENTROPY goes at 100 %, in the character's figures (`character.rs`, the DAW's):
/// twice them, as in the DAW's instrument (the owner, 2026-10-02, found them too subtle at
/// once).
pub const ENTROPY_DEPTH: f64 = 2.0;

/// The oscillators' own mismatch and drift with LOCK off, in the character's figures
/// (decisions.md R9; the DAW's): a tuning spread of 3 cents and 2.25 of drift, so oscillators
/// at one pitch never lock in phase (locked, three in unison sounded 3.3 dB louder, not
/// better). ENTROPY adds to it.
pub const ENTROPY_FLOOR: f64 = 0.75;

/// How much of the ideal gain of locking (`10 log10(coherent / incoherent)`) shows at the
/// output, the mixer's overdrive and the filter taking the rest (fitted in the DAW).
pub const LOCK_SHARE: f64 = 0.68;

/// The output's gain with LOCK on: down by the panel's switched-on oscillators that share a
/// pitch (RANGE, FREQUENCY and, for oscillator 3, OSC. 3 CONTROL alike) and their volumes, so
/// locked oscillators sound as loud as drifting ones. With no two at one pitch, 1.
pub fn lock_trim(p: &Panel) -> f64 {
    let keys = |n: usize| (p.osc[n].range, p.osc[n].freq, n < 2 || p.osc3_control);
    let on = |n: usize| p.osc[n].on && p.osc[n].volume > 0.0;
    let incoherent: f64 = (0..3)
        .filter(|&n| on(n))
        .map(|n| p.osc[n].volume.powi(2))
        .sum();
    let mut coherent = 0.0;
    for n in (0..3).filter(|&n| on(n)) {
        if (0..n).any(|m| on(m) && keys(m) == keys(n)) {
            continue;
        }
        let a: f64 = (0..3)
            .filter(|&m| on(m) && keys(m) == keys(n))
            .map(|m| p.osc[m].volume)
            .sum();
        coherent += a * a;
    }
    if incoherent > 0.0 && coherent > incoherent {
        ca72::ulp::pow(coherent / incoherent, -0.5 * LOCK_SHARE)
    } else {
        1.0
    }
}

/// The voices POLY may play: the fewest, by default, the most (all of them built).
pub const POLY_VOICES: (usize, usize, usize) = (2, 4, 10);

/// The voices whose levels the editor is told: as many as POLY can have.
pub const LEVELS: usize = POLY_VOICES.2;

/// DRIVE's top, dB (decisions.md R45, the CA-74's R29: past it the filter is saturated
/// and the note stops being one).
pub const DRIVE_TOP: f64 = 24.0;

/// LEVEL's range, dB (the CA-74's).
pub const LEVEL_RANGE: (f64, f64) = (-30.0, 12.0);

/// ENTROPY's tolerances on the panel: EMPHASIS, the filter contour's ATTACK and DECAY, the
/// loudness contour's, GLIDE.
const ENTROPY_KNOBS: usize = 6;

/// What the parameters set: the panel (its wheels where the parameters leave them), MAIN
/// OUTPUT's VOLUME (0..1 of its travel) and switch, how far a MIDI keyboard's full pitch
/// bend moves the PITCH wheel, in semitones, POWER (off: bypassed, silent), and POLY,
/// VOICES, ENTROPY, SPREAD and INNER (0..1), where SPREAD places the voices, UNISON and DOUBLE.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Controls {
    pub panel: Panel,
    pub volume: f64,
    pub main_output: bool,
    pub bend_range: f64,
    pub power: bool,
    pub poly: bool,
    pub voices: usize,
    pub entropy: f64,
    pub spread: f64,
    /// INNER: the inner edge of each side's band, a share of SPREAD's way out (decisions.md
    /// R49).
    pub inner: f64,
    /// SCATTER's placement: where SPREAD puts POLY's voices (decisions.md R45), EVEN or
    /// CENTER as the plug-in offers them (`params::Scatter`; R49).
    pub placement: Placement,
    /// UNISON: VOICES instruments on every key together (decisions.md R45).
    pub unison: bool,
    /// DOUBLE (0..1): each note played by two voices, detuned from each other by up to
    /// [`DOUBLE_CENTS`], SPREAD putting them to either side of the centre (decisions.md
    /// R45); 0, one voice a note.
    pub double: f64,
    /// DRIVE (0..[`DRIVE_TOP`] dB): the mixer's signal into the filter raised by so much, its
    /// input pair driven past where the panel's mixer can take it; 0, the circuit
    /// (decisions.md R45).
    pub drive: f64,
    /// LEVEL (dB, [`LEVEL_RANGE`]): the output's gain after MAIN OUTPUT's, the plug-in's own;
    /// 0, none (decisions.md R45).
    pub level: f64,
    /// AUTO GAIN: the output brought back down by as much as DRIVE made the sound louder, as
    /// measured for it (`drive.rs`; decisions.md R45).
    pub auto_gain: bool,
    /// FEEDBACK (0..1): the voices' output patched back into their EXTERNAL INPUT, a sample
    /// late, by this much of its level (the phones' VOLUME knob; decisions.md R8).
    pub feedback: f64,
    /// LOCK: the oscillators identical, the circuit as drawn, its level matched
    /// ([`lock_trim`]); off, each oscillator keeps its own floor of mismatch
    /// ([`ENTROPY_FLOOR`]; decisions.md R9).
    pub lock: bool,
    /// QUALITY at LO, Potato mode: the voices played by the light model ([`Light`]), not the
    /// circuit's, for computers the circuit is too heavy for (decisions.md R47).
    pub potato: bool,
    /// QUALITY at ULTRA: the circuit's model with no compromises (`Quality::NoCompromises`),
    /// not at its real-time setting (decisions.md R48). Switched while it plays.
    pub ultra: bool,
}

impl Default for Controls {
    fn default() -> Self {
        Controls {
            panel: Panel::default(),
            volume: 1.0,
            main_output: true,
            bend_range: 2.0,
            power: true,
            poly: false,
            voices: POLY_VOICES.1,
            entropy: 0.0,
            spread: 0.0,
            inner: 0.0,
            placement: Placement::Even,
            unison: false,
            double: 0.0,
            drive: 0.0,
            level: 0.0,
            auto_gain: true,
            feedback: 0.0,
            lock: false,
            potato: false,
            ultra: false,
        }
    }
}

/// MIDI input, as the engine takes it (any channel).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event {
    Note {
        key: u8,
        on: bool,
    },
    /// The pitch bend, -1..1, 0 at rest.
    PitchBend(f32),
    /// The modulation wheel (control change 1), 0..1.
    Modulation(f32),
    /// Reset All Controllers (control change 121): the bend and modulation wheel at rest.
    ResetControllers,
    /// All Notes Off (control change 123): every key released (the contours then run out as
    /// the DECAY switch has them).
    AllNotesOff,
    /// All Sound Off (control change 120): every key released and the sound gone at once, the
    /// output faded out over [`HUSH_FADE`] and every voice put to rest (decisions.md R18).
    AllSoundOff,
}

/// The control change of a keyboard's modulation wheel.
pub const MODULATION_WHEEL: u8 = 1;
pub const ALL_SOUND_OFF: u8 = 120;
pub const RESET_CONTROLLERS: u8 = 121;
pub const ALL_NOTES_OFF: u8 = 123;

/// How fast the reported OVERLOAD lamp falls from a peak, s: the lamp's own thermal lag is
/// not modelled, and a display polling now and then should still see a flash.
const LAMP_FALL: f32 = 0.05;

/// The lowest rate the voice runs at, Hz: below it, at 2, 4 or 8 times the host's. The top
/// key's fundamental on 2' (4.2 kHz) stays under its Nyquist frequency, and the model's
/// factory calibration fails at much lower rates (it gives NaN at 1.2 kHz).
pub const MIN_VOICE_RATE: f64 = 8_000.0;

/// How long POWER takes to fade the output out or in, s (switched at once, it clicks).
const POWER_FADE: f64 = 0.01;

/// How long All Sound Off takes to fade the output out before the voices are put to rest, s
/// (at once, it clicks; decisions.md R18). It fades back in as POWER does.
pub const HUSH_FADE: f64 = 0.005;

/// The longest All Sound Off keeps the output silent waiting for clean voices to put in the
/// sounding ones' places ([`Spares`]), s; they are normally ready long before.
const HUSH_WAIT: f64 = 0.1;

/// A POLY voice falls silent, and is free for another note, once its key is up and its
/// output has stayed under this for a whole block.
const SILENT: f64 = 1e-6;

/// A twin let go (DOUBLE turned off) falls silent once its output has stayed under [`SILENT`]
/// for this many samples running, counted across runs (the CA-74's).
const QUIET: usize = 256;

/// Without POLY or UNISON the one instrument sleeps (is not played, the output silent) once no
/// key is down, its output has stayed under [`SILENT`] and its slow state (`slow_state`: the
/// keyboard's hold capacitor and the contours, its twin's too with DOUBLE) has moved less than
/// [`SETTLED`] over this, s; the next note then starts where it would have (decisions.md
/// R68, the development line's R30).
const SLEEP_WINDOW: f64 = 0.05;

/// How little the slow state may move over [`SLEEP_WINDOW`] for the instrument to sleep, V: a
/// contour falling with a time constant of 10 s is then within 0.2 mV of where it settles.
const SETTLED: f64 = 1e-6;

/// Whether the one instrument sleeps, and how long it has been silent with its slow state
/// still (`quiet`, of `window` samples at the host's rate) since `anchor`; the output's peak
/// over the block so far.
#[derive(Debug, Clone, Copy, Default)]
struct Rest {
    asleep: bool,
    quiet: usize,
    window: usize,
    anchor: [f64; 6],
    peak: f64,
    /// Sleep not allowed (measurements that play on as before: [`Engine::never_sleep`]).
    never: bool,
}

impl Rest {
    /// The output's samples, as they are made.
    fn heard(&mut self, l: &[f32], r: &[f32]) {
        for y in l.iter().chain(r) {
            self.peak = self.peak.max(f64::from(y.abs()));
        }
    }

    /// After a block of `n` samples: the instrument's slow state then (None while a key is
    /// down or its voice is out of reach), the block's peak reckoned.
    fn after(&mut self, state: Option<[f64; 6]>, n: usize) {
        let silent = self.peak < SILENT;
        self.peak = 0.0;
        match state {
            Some(st) if silent && !self.never && !self.asleep => {
                if self.quiet == 0 {
                    self.anchor = st;
                }
                self.quiet += n;
                if self.quiet >= self.window {
                    self.asleep = st
                        .iter()
                        .zip(&self.anchor)
                        .all(|(a, b)| (a - b).abs() < SETTLED);
                    self.quiet = 0;
                }
            }
            _ => self.quiet = 0,
        }
    }

    fn wake(&mut self) {
        (self.asleep, self.quiet, self.peak) = (false, 0, 0.0);
    }
}

/// The one instrument's slow state (its voice's, and its twin's while DOUBLE plays it), the
/// model QUALITY plays.
fn slow_state(p: &Playing) -> Option<[f64; 6]> {
    let v = if p.potato {
        p.light.slow_state()?
    } else {
        p.voice.slow_state()?
    };
    let t = if p.double || p.twin.live {
        if p.potato {
            p.twin.light.slow_state()?
        } else {
            p.twin.voice.slow_state()?
        }
    } else {
        [0.0; 3]
    };
    Some([v[0], v[1], v[2], t[0], t[1], t[2]])
}

/// The time constant, seconds, with which a voice's place follows SPREAD, INNER, the
/// placement, VOICES and DOUBLE: plugin-kit's (its K6; decisions.md R45, the CA-74's R27
/// and R28).
pub use plugin_kit_stereo::place::GLIDE;

/// The time constant, seconds, with which the one instrument's place follows when it scatters
/// (POLY, UNISON and DOUBLE off: decisions.md R62): a fifth of the time its contours
/// take to start after its trigger contact closes (the light model's `TRIGGER_ON`, fitted to
/// the circuit's), so that at an attack it is within 1 % of its new place before its contours
/// start (with [`GLIDE`] it would still be 46 % of the way short).
pub const TURN_GLIDE: f64 = ca72::light::laws::TRIGGER_ON / 5.0;

/// The detune between DOUBLE's two voices of a note at its full amount, cents: the CA-72's own
/// (the owner, 2026-10-10: "let's make our maximum detune 100"; the kit's, the CA-74's, is 20).
pub const DOUBLE_CENTS: f64 = 100.0;

/// The voices at a multiple of the host's rate: the side chain interpolated up, the left and
/// right outputs decimated down.
#[derive(Debug)]
struct Oversampled {
    factor: usize,
    up: Interpolator,
    down: [Decimator; 2],
}

/// One POLY voice's key (decisions.md, "POLY"): its keyboard circuit gets its note as a key
/// down while the voice's gate is high. A new note on the voice while its gate was high
/// (the gate dropped for a sample between them) lifts the key and presses the new one once
/// the trigger contact has been open [`RETRIGGER_GAP`]: the circuit re-arms its trigger only
/// then, so the contours retrigger; the new note's pitch and attack come that much later.
#[derive(Debug, Clone, Copy)]
struct PolyKey {
    down: Option<i32>,
    /// Samples the trigger contact has been open, and how many it must be.
    open: u32,
    gap: u32,
}

impl PolyKey {
    fn new(rate: f64) -> PolyKey {
        let gap = (RETRIGGER_GAP * rate).ceil() as u32;
        PolyKey {
            down: None,
            open: gap,
            gap,
        }
    }

    /// The voice's gate and key for the next sample, played on its keys by `note` (the
    /// circuit's voice's, or the light one's).
    fn step(&mut self, gate: bool, key: i32, mut note: impl FnMut(i32, bool)) {
        if gate {
            match self.down {
                None if self.open >= self.gap => {
                    note(key, true);
                    self.down = Some(key);
                }
                None => {}
                Some(old) if old != key => {
                    note(key, true);
                    note(old, false);
                    self.down = Some(key);
                }
                Some(_) => {}
            }
        } else if let Some(old) = self.down.take() {
            note(old, false);
            self.open = 0;
        }
        if self.down.is_none() {
            self.open = self.open.saturating_add(1);
        }
    }

    /// Its key lifted at once (POLY switched, the voices rebuilt), on the keys `note` plays.
    fn lift(&mut self, mut note: impl FnMut(i32, bool)) {
        if let Some(old) = self.down.take() {
            note(old, false);
        }
    }

    /// As on a voice at rest: no key down, the trigger contact open long enough.
    fn rest(&mut self) {
        self.down = None;
        self.open = self.gap;
    }
}

/// A POLY voice's place: its note, gate and age, whether it still sounds, kept by the
/// engine (events change them without touching the voice); and the voice as it plays,
/// shared with the workers ([`crate::pool`]).
#[derive(Debug)]
struct Slot {
    play: Shared,
    note: u8,
    gate: bool,
    /// The gate the voice saw at its last sample, and a one-sample drop pending (a new note
    /// on a gate that was high).
    shown: bool,
    drop: bool,
    age: u64,
    /// Whether it is played (from its note until it falls silent), and its output's peak
    /// over the block so far.
    active: bool,
    peak: f64,
    /// What waits for the voice while a worker late from an earlier run holds it: its key
    /// lifted, and the one instrument's keys pressed or released (POLY off), in order.
    lift: bool,
    notes: [(i32, bool); 32],
    pending: usize,
    /// To be put to rest once All Sound Off has faded the output out.
    rest: bool,
}

impl Slot {
    /// The run's gate, note and drop to its voice. A drop it was given for a run it did not
    /// play (a worker late, the run closed before it began) stays with it for its next.
    fn hand(&self, p: &mut Playing) {
        (p.gate, p.note) = (self.gate, self.note);
        p.drop |= self.drop;
    }

    /// After a run its voice played by the deadline (`ok`) or not: the drop is the voice's
    /// now; the gate it shows is the run's, or, not known to have played it, either that or the
    /// one before, so that a new note on it retriggers whichever it saw (decisions.md R18).
    fn ran(&mut self, ok: bool) {
        self.drop = false;
        self.shown = if ok {
            self.gate
        } else {
            self.shown || self.gate
        };
    }
}

/// DOUBLE's second voice of a note (decisions.md R45, the CA-74's R28): a voice with
/// parts of its own (its character, noise and oscillators' start, a place's past the voices'),
/// its own key, its gains at the last run's end (None before its first run, or while it does
/// not play), and whether it still sounds: with DOUBLE turned off it plays its tail out, its
/// key lifted, until silent for [`QUIET`] samples (how many so far).
#[derive(Debug, Clone)]
struct Twin {
    voice: Voice,
    light: Light,
    character: Character,
    key: PolyKey,
    pan: Option<(f64, f64)>,
    live: bool,
    quiet: usize,
}

impl Twin {
    /// After a run of `n` samples whose loudest was `peak`: live while DOUBLE is on; let go,
    /// until it has been silent long enough; its gains where the run left them.
    fn ran(&mut self, n: usize, doubled: bool, peak: f64, pan: Option<(f64, f64)>) {
        if doubled {
            (self.live, self.quiet) = (true, 0);
        } else if self.live {
            self.quiet = if peak < SILENT { self.quiet + n } else { 0 };
            if self.quiet >= QUIET {
                (self.live, self.quiet) = (false, 0);
            }
        }
        self.pan = pan;
    }
}

/// Voice `k`'s twin for DOUBLE, of an instance seeded `seed`, at `rate` Hz: made as the voice
/// of place `k` past the voices' is. (Boxed, as a spare twin is: a voice is some 34 KB, and a
/// debug build copies what is made about the stack; ten spares' twins unboxed overflowed a
/// test thread's 2 MB. A voice's place stays the size it was.)
fn made_twin(rate: f64, seed: u64, k: usize) -> Box<Twin> {
    let j = k + POLY_VOICES.2;
    Box::new(Twin {
        voice: made_voice(rate, seed, j),
        light: made_light(rate, seed, j),
        character: Character::new(voice_seed(seed, j), j, 3, ENTROPY_KNOBS),
        key: PolyKey::new(rate),
        pan: None,
        live: false,
        quiet: 0,
    })
}

/// A voice as it plays: the voice, its character and its key, and a run's gate, note and
/// drop in, its samples out; with DOUBLE, its twin too.
#[derive(Debug, Clone)]
pub struct Playing {
    voice: Voice,
    /// The light model's voice, and whether it is the one played (QUALITY at LO).
    light: Light,
    potato: bool,
    character: Character,
    key: PolyKey,
    gate: bool,
    note: u8,
    drop: bool,
    /// The last run's samples (left, right) and its output's peak.
    out: [[f64; CHUNK]; 2],
    peak: f64,
    /// Its gains, left and right, at the last run's end, gliding to its place (None before
    /// its first run: there from the start).
    pan: Option<(f64, f64)>,
    /// The engine's panel's generation it plays with ([`Engine::apply`]).
    panel_gen: u64,
    /// Its output went non-finite: silent until the engine puts a clean voice in its place
    /// ([`Engine::restore`]; decisions.md R18).
    broken: bool,
    /// DOUBLE's second voice of the note, and whether DOUBLE is on as the engine's panel
    /// last had it (the twin takes the keys pressed only then).
    twin: Box<Twin>,
    double: bool,
    /// The one instrument's place among VOICES's when it scatters (decisions.md R62):
    /// the place its last attack took (None before its first, or since the pattern started
    /// again: the first place); and whether its trigger contact was closed at its last sample.
    turn: Option<usize>,
    closed: bool,
}

/// Where a run puts a voice and its twin: their gains (left, right), DOUBLE's half detune,
/// cents, whether DOUBLE is on and whether the twin plays.
struct Places {
    voice: (f64, f64),
    twin: (f64, f64),
    half: f64,
    doubled: bool,
    twin_on: bool,
}

impl Playing {
    /// A run of `n` samples with `ext` at EXTERNAL INPUT: its key moved on a sample at a time
    /// (the drop at the first), its character's offsets and its place; with DOUBLE its twin
    /// too; the samples in `out`, the peak in `peak`. A sample that is not finite marks the
    /// voice broken, and it is silent from there.
    pub fn play(&mut self, n: usize, ext: &[f64; CHUNK], rate: f64, m: &Mix) {
        self.play_from(&ext[..n], rate, m, glide_share(rate));
    }

    /// As [`Playing::play`], for as many samples as `ext` has, its place gliding by `share`
    /// a sample ([`glide_share`] at `rate`).
    fn play_from(&mut self, ext: &[f64], rate: f64, m: &Mix, share: f64) {
        if self.potato {
            self.play_light_from(ext, rate, m, share);
            return;
        }
        let mut p = self.places(m);
        self.voice.set_drive(m.drive);
        self.twin.voice.set_drive(m.drive);
        let (mut gl, mut gr) = self.pan.unwrap_or(p.voice);
        let (mut hl, mut hr) = self.twin.pan.unwrap_or(p.twin);
        let (mut peak, mut twin_peak) = (0.0f64, 0.0f64);
        let note = i32::from(self.note);
        for (k, &x) in ext.iter().enumerate() {
            let gate = if self.drop {
                self.drop = false;
                false
            } else {
                self.gate
            };
            gl += (p.voice.0 - gl) * share;
            gr += (p.voice.1 - gr) * share;
            if p.twin_on {
                hl += (p.twin.0 - hl) * share;
                hr += (p.twin.1 - hr) * share;
            }
            if self.broken {
                (self.out[0][k], self.out[1][k]) = (0.0, 0.0);
                continue;
            }
            let voice = &mut self.voice;
            self.key.step(gate, note, |k, on| voice.note(k, on));
            let mut j = jacks(&mut self.character, x, rate, m);
            if p.doubled {
                j.detune = j.detune.map(|d| d - p.half);
            }
            let mut y = self.voice.tick_jacks(&j);
            if !y.is_finite() {
                self.broken = true;
                y = 0.0;
            }
            self.turned(self.voice.trigger_closed(), m, &mut p);
            let (mut left, mut right, mut most) = (y * gl, y * gr, y.abs());
            if p.twin_on {
                let t = &mut self.twin;
                // (A twin let go plays its tail out, its key lifted.)
                let voice = &mut t.voice;
                t.key
                    .step(gate && p.doubled, note, |k, on| voice.note(k, on));
                let mut j = jacks(&mut t.character, x, rate, m);
                if p.doubled {
                    j.detune = j.detune.map(|d| d + p.half);
                }
                let mut z = t.voice.tick_jacks(&j);
                if !z.is_finite() {
                    self.broken = true;
                    z = 0.0;
                }
                left += z * hl;
                right += z * hr;
                most = most.max(z.abs());
                twin_peak = twin_peak.max(z.abs());
            }
            peak = peak.max(most);
            self.out[0][k] = left;
            self.out[1][k] = right;
        }
        self.peak = peak;
        self.pan = Some((gl, gr));
        self.twin.ran(
            ext.len(),
            p.doubled,
            twin_peak,
            p.twin_on.then_some((hl, hr)),
        );
    }

    /// [`Playing::play_from`] at QUALITY's LO: the light voices played in the circuit's
    /// voices' place, with their keys, characters, places and DOUBLE's twin as theirs (the
    /// character's offsets read each sample, as the circuit's voice reads them, so that a
    /// voice's character is the same whichever plays it).
    fn play_light_from(&mut self, ext: &[f64], rate: f64, m: &Mix, share: f64) {
        let mut p = self.places(m);
        self.light.set_drive(m.drive);
        self.twin.light.set_drive(m.drive);
        let (mut gl, mut gr) = self.pan.unwrap_or(p.voice);
        let (mut hl, mut hr) = self.twin.pan.unwrap_or(p.twin);
        let (mut peak, mut twin_peak) = (0.0f64, 0.0f64);
        let note = i32::from(self.note);
        for (k, &x) in ext.iter().enumerate() {
            let gate = if self.drop {
                self.drop = false;
                false
            } else {
                self.gate
            };
            gl += (p.voice.0 - gl) * share;
            gr += (p.voice.1 - gr) * share;
            if p.twin_on {
                hl += (p.twin.0 - hl) * share;
                hr += (p.twin.1 - hr) * share;
            }
            if self.broken {
                (self.out[0][k], self.out[1][k]) = (0.0, 0.0);
                continue;
            }
            let light = &mut self.light;
            self.key.step(gate, note, |k, on| light.note(k, on));
            let j = jacks(&mut self.character, x, rate, m);
            let detune = if p.doubled {
                j.detune.map(|d| d - p.half)
            } else {
                j.detune
            };
            self.light.set_offsets(detune, j.cutoff);
            let mut y = self.light.tick(x);
            if !y.is_finite() {
                self.broken = true;
                y = 0.0;
            }
            self.turned(self.light.trigger_closed(), m, &mut p);
            let (mut left, mut right, mut most) = (y * gl, y * gr, y.abs());
            if p.twin_on {
                let t = &mut *self.twin;
                let light = &mut t.light;
                t.key
                    .step(gate && p.doubled, note, |k, on| light.note(k, on));
                let j = jacks(&mut t.character, x, rate, m);
                let detune = if p.doubled {
                    j.detune.map(|d| d + p.half)
                } else {
                    j.detune
                };
                t.light.set_offsets(detune, j.cutoff);
                let mut z = t.light.tick(x);
                if !z.is_finite() {
                    self.broken = true;
                    z = 0.0;
                }
                left += z * hl;
                right += z * hr;
                most = most.max(z.abs());
                twin_peak = twin_peak.max(z.abs());
            }
            peak = peak.max(most);
            self.out[0][k] = left;
            self.out[1][k] = right;
        }
        self.peak = peak;
        self.pan = Some((gl, gr));
        self.twin.ran(
            ext.len(),
            p.doubled,
            twin_peak,
            p.twin_on.then_some((hl, hr)),
        );
    }

    /// Where a run with `m` puts the voice and its twin: with DOUBLE the pair about the
    /// centre, the voice half the detune flat to the right and its twin half sharp as far to
    /// the left, SPREAD's way out as far as the placement puts the voice (the CA-74's R28 and
    /// R41); else the voice at its own place, or the one instrument scattering at its turn's
    /// (a twin let go plays its tail where it was).
    fn places(&self, m: &Mix) -> Places {
        let doubled = m.double > 0.0;
        let twin_on = doubled || self.twin.live;
        // (One voice alone, no placement among others: its pair all of SPREAD's way out; a twin
        // let go as the one instrument starts to scatter, there still.)
        let out = match m.placement {
            _ if !twin_on => 0.0,
            Some(p) if !m.scatter => self.character.pair(p, m.voices),
            _ => ALONE,
        };
        let wide = |(l, r): (f32, f32)| (f64::from(l), f64::from(r));
        let [voice, twin] = pair_gains(m.spread, m.inner, out);
        Places {
            voice: if doubled {
                wide(voice)
            } else if m.scatter {
                self.scattered(m)
            } else {
                let placement = m.placement.unwrap_or_default();
                wide(self.character.gains(m.spread, m.inner, placement, m.voices))
            },
            twin: if twin_on { wide(twin) } else { (0.0, 0.0) },
            half: 0.5 * m.double,
            doubled,
            twin_on,
        }
    }

    /// The one instrument's gains at its turn's place among VOICES's, scattering (decisions.md
    /// R62): before its first attack, the first place.
    fn scattered(&self, m: &Mix) -> (f64, f64) {
        let placement = m.placement.unwrap_or_default();
        let at = self.turn.unwrap_or(0);
        let (l, r) = voice_gains(m.spread, m.inner, placement, at, m.voices);
        (f64::from(l), f64::from(r))
    }

    /// Its trigger contact `closed` at the sample just played: closing (an attack's start) with
    /// the one instrument scattering, its place the next of VOICES's, from the next sample
    /// (decisions.md R62). A note played legato, its contact staying closed, stays.
    fn turned(&mut self, closed: bool, m: &Mix, p: &mut Places) {
        if m.scatter && closed && !self.closed {
            self.turn = Some(self.turn.map_or(0, |t| (t + 1) % m.voices.max(1)));
            p.voice = self.scattered(m);
        }
        self.closed = closed;
    }

    /// A key pressed or released on the voice's keyboard (the one instrument's, or UNISON's),
    /// and on its twin's: pressed there only with DOUBLE on, so that a twin let go plays its
    /// tail out.
    fn note(&mut self, key: i32, on: bool) {
        let twin = self.double || !on;
        if self.potato {
            self.light.note(key, on);
            if twin {
                self.twin.light.note(key, on);
            }
        } else {
            self.voice.note(key, on);
            if twin {
                self.twin.voice.note(key, on);
            }
        }
    }

    /// Its key and its twin's lifted at once, on the voices played.
    fn lift(&mut self) {
        let t = &mut *self.twin;
        if self.potato {
            let (light, twin) = (&mut self.light, &mut t.light);
            self.key.lift(|k, on| light.note(k, on));
            t.key.lift(|k, on| twin.note(k, on));
        } else {
            let (voice, twin) = (&mut self.voice, &mut t.voice);
            self.key.lift(|k, on| voice.note(k, on));
            t.key.lift(|k, on| twin.note(k, on));
        }
    }
}

/// A clean voice for each voice's place, made off the audio thread, to take a voice's place
/// on it without allocating (decisions.md R18): a voice whose output went non-finite, and
/// every sounding voice at All Sound Off. The voice taken out is dropped off the audio thread
/// too, when the spares are next mended ([`Spares::mend`]). The audio thread only tries their
/// locks.
#[derive(Debug, Default)]
pub struct Spares {
    places: [Mutex<Spare>; POLY_VOICES.2],
}

/// A place's spare voice and twin (DOUBLE's): clean, or the ones they replaced. (The twin
/// boxed, as [`made_twin`]'s: the places are made together.)
#[derive(Debug, Default)]
struct Spare {
    voice: Option<Voice>,
    twin: Option<Box<Voice>>,
    clean: bool,
}

impl Spares {
    /// Each place whose voice was taken given a clean one again, at its rate, and the one
    /// taken out dropped. Not on the audio thread: the plug-in has its helper thread do it when
    /// the engine asks ([`Engine::take_mending`]; decisions.md R23).
    pub fn mend(&self) {
        self.mend_until(|| false);
    }

    /// As [`Spares::mend`], stopping before the next voice once `stop` (the plug-in dropped:
    /// decisions.md R23), so that waiting for it takes at most one voice's making.
    pub fn mend_until(&self, stop: impl Fn() -> bool) {
        for place in &self.places {
            if stop() {
                return;
            }
            let used = place
                .lock()
                .ok()
                .filter(|s| !s.clean)
                .and_then(|s| s.voice.as_ref().map(Voice::rate));
            if let Some(rate) = used {
                fill(place, rate, |s| {
                    !s.clean && s.voice.as_ref().map(Voice::rate) == Some(rate)
                });
            }
        }
    }

    /// Every place given a clean voice at `rate` Hz (the voices' own) unless it has one.
    fn make(&self, rate: f64) {
        for place in &self.places {
            fill(place, rate, |s| {
                !(s.clean && s.voice.as_ref().map(Voice::rate) == Some(rate))
            });
        }
    }
}

/// A clean voice and twin at `rate` Hz put in `place` while it is `wanted` there: made before
/// the lock is taken, and the ones they replace dropped after.
fn fill(place: &Mutex<Spare>, rate: f64, wanted: impl Fn(&Spare) -> bool) {
    if !place.lock().is_ok_and(|s| wanted(&s)) {
        return;
    }
    let (mut voice, mut twin) = (
        Some(Voice::prototype(rate)),
        Some(Box::new(Voice::prototype(rate))),
    );
    if let Ok(mut s) = place.lock()
        && wanted(&s)
    {
        std::mem::swap(&mut s.voice, &mut voice);
        std::mem::swap(&mut s.twin, &mut twin);
        s.clean = true;
    }
    drop((voice, twin));
}

/// The voices, the MIDI input's state and the controls they play with.
#[derive(Debug)]
pub struct Engine {
    slots: Vec<Slot>,
    /// Each voice's output's peak over the last block (the editor's drops and lamps follow
    /// them; each slot's peak is kept anyway, for knowing when its voice falls silent).
    levels: [f32; LEVELS],
    /// The one instrument asleep, or how near it is (decisions.md R68).
    rest: Rest,
    over: Option<Oversampled>,
    rate: f64,
    seed: u64,
    controls: Controls,
    gain: f64,
    /// Keys held (MIDI numbers), played on by voices built for a new rate (POLY off).
    held: [bool; 128],
    /// POLY's notes taken so far (each voice's age).
    counter: u64,
    bend: f32,
    modulation: f32,
    lamp: f32,
    /// A voice's place's glide a sample at the voices' rate ([`glide_share`]), and the one
    /// instrument's while it scatters ([`TURN_GLIDE`]).
    glide: f64,
    turn_glide: f64,
    /// The place the one instrument's voice took at its last attack, scattering (the editor
    /// shows it), and whether its pattern starts again from the first place at its next run
    /// (every key let go by the host: decisions.md R62).
    turn: usize,
    restart: bool,
    /// AUTO GAIN (decisions.md R45): the sound's curve (`drive.rs`), the correction the
    /// controls ask for from it (1: none), the correction gliding to it a sample at a time,
    /// and the glide's share a sample at the host's rate. Only it glides: the output's other
    /// gains step as they did.
    curve: Curve<STEPS>,
    auto: f64,
    auto_now: f64,
    auto_share: f64,
    /// Where AUTO GAIN's measurements are asked for and kept, told the voices' rate.
    calibration: Option<Calibration>,
    /// The output's gain under POWER (0 off, 1 on), and its step a sample while it fades.
    fade: f64,
    fade_step: f64,
    /// All Sound Off under way (the output fading out, or out and the voices being put to
    /// rest), its fade's step a sample and the samples it has waited, faded out.
    hush: bool,
    hush_step: f64,
    hush_waited: usize,
    /// The clean voices put in voices' places, and whether some are to be made again.
    spares: Arc<Spares>,
    mend: bool,
    /// The panel's generation (each [`Engine::apply`] a new one), and the voices'
    /// preamplifier inside FEEDBACK's loop.
    panel_gen: u64,
    in_loop: ca72::preamp::InLoop,
    /// The workers sharing POLY's voices (none: every voice on the caller's thread), how many
    /// were asked for and the block period they were made audio threads for; and the
    /// present block's deadline for them.
    pool: Option<Pool>,
    workers: (usize, Option<Duration>),
    deadline: Option<Instant>,
    /// Where POLY's workers are started and stopped off the audio thread, the plan they are
    /// started for there (0: none), this thread's ask for them and whether the helper
    /// thread is to serve it (decisions.md R21, R23).
    crew: Arc<Crew>,
    plan: u64,
    ask: Ask,
    serve: bool,
    /// How often the workers were started (for the tests).
    #[cfg(test)]
    pool_starts: u64,
}

impl Default for Engine {
    fn default() -> Self {
        Engine::new()
    }
}

/// How a voice's place reaches its voice: `f` on it at once, unless a worker late from an
/// earlier run holds it.
fn with_voice<T>(s: &Slot, f: impl FnOnce(&mut Playing) -> T) -> Option<T> {
    s.play.try_lock().ok().map(|mut p| f(&mut p))
}

impl Engine {
    /// An engine without voices (silent until [`Engine::prepare`]).
    pub fn new() -> Engine {
        let controls = Controls::default();
        Engine {
            slots: Vec::new(),
            levels: [0.0; LEVELS],
            rest: Rest::default(),
            over: None,
            rate: 0.0,
            seed: 0,
            gain: output_gain(controls.volume, controls.main_output),
            controls,
            held: [false; 128],
            counter: 0,
            bend: 0.0,
            modulation: 0.0,
            lamp: 0.0,
            glide: 1.0,
            turn_glide: 1.0,
            turn: 0,
            restart: false,
            curve: AVERAGE,
            auto: 1.0,
            auto_now: 1.0,
            auto_share: 1.0,
            calibration: None,
            fade: 1.0,
            fade_step: 1.0,
            hush: false,
            hush_step: 1.0,
            hush_waited: 0,
            spares: Arc::new(Spares::default()),
            mend: false,
            panel_gen: 0,
            in_loop: ca72::preamp::InLoop::default(),
            pool: None,
            workers: (0, None),
            deadline: None,
            crew: Arc::new(Crew::new()),
            plan: 0,
            ask: Ask::Not,
            serve: false,
            #[cfg(test)]
            pool_starts: 0,
        }
    }

    /// The voices for `rate` Hz, their noise seeded from `seed` (the first voice's is `seed`
    /// itself, as with one voice), playing the keys held, and their spares ([`Spares`]). Not
    /// on the audio thread: voices at a rate not yet built take up to a second (the
    /// oscillators' factory tuning, the tables); later ones are copies. At the rate they were
    /// built for, the voices are kept, their noise and (for a new seed) their characters
    /// drawn from `seed` again, as a session's saved seed loaded into a running instance has
    /// them (decisions.md R18). New voices stop the workers, here ([`Engine::start_workers`]
    /// plans them again).
    pub fn prepare(&mut self, rate: f64, seed: u64) {
        self.rest.wake();
        self.rest.window = (SLEEP_WINDOW * rate).ceil() as usize;
        let reseeded = seed != self.seed;
        self.seed = seed;
        if rate == self.rate && !self.slots.is_empty() {
            let voice_rate = self.voice_rate();
            for (k, s) in self.slots.iter_mut().enumerate() {
                if let Ok(mut p) = s.play.lock() {
                    if reseeded {
                        // (Made again, as an instance opened with the seed makes them: its
                        // voices' oscillators start where it says.)
                        p.voice = made_voice(voice_rate, seed, k);
                        p.light = made_light(voice_rate, seed, k);
                        p.character = Character::new(voice_seed(seed, k), k, 3, ENTROPY_KNOBS);
                        p.twin = made_twin(voice_rate, seed, k);
                        p.panel_gen = u64::MAX;
                    } else {
                        p.voice.set_seed(voice_seed(seed, k));
                        p.light.set_seed(voice_seed(seed, k));
                    }
                }
            }
            if reseeded {
                self.apply();
                self.press_held();
            }
            self.auto_now = self.auto;
            self.spares.make(voice_rate);
            return;
        }
        self.stop_workers();
        self.rate = rate;
        self.fade_step = (1.0 / (POWER_FADE * rate)).min(1.0);
        self.hush_step = (1.0 / (HUSH_FADE * rate)).min(1.0);
        self.hush = false;
        let factor = [1, 2, 4, 8]
            .into_iter()
            .find(|f| rate * *f as f64 >= MIN_VOICE_RATE)
            .unwrap_or(8);
        self.over = (factor > 1).then(|| Oversampled {
            factor,
            up: Interpolator::new(factor),
            down: [Decimator::new(factor), Decimator::new(factor)],
        });
        let voice_rate = rate * factor as f64;
        self.glide = glide_share(voice_rate);
        self.turn_glide = 1.0 - (-1.0 / (TURN_GLIDE * voice_rate)).exp();
        (self.auto_now, self.auto_share) = (self.auto, glide_share(rate));
        if let Some(c) = &self.calibration {
            c.set_rate(voice_rate);
        }
        self.slots = (0..POLY_VOICES.2)
            .map(|k| Slot {
                play: Arc::new(Mutex::new(Playing {
                    voice: made_voice(voice_rate, seed, k),
                    light: made_light(voice_rate, seed, k),
                    potato: false,
                    character: Character::new(voice_seed(seed, k), k, 3, ENTROPY_KNOBS),
                    key: PolyKey::new(voice_rate),
                    gate: false,
                    note: 0,
                    drop: false,
                    out: [[0.0; CHUNK]; 2],
                    peak: 0.0,
                    pan: None,
                    panel_gen: u64::MAX,
                    broken: false,
                    twin: made_twin(voice_rate, seed, k),
                    double: false,
                    turn: None,
                    closed: false,
                })),
                note: 0,
                gate: false,
                shown: false,
                drop: false,
                age: 0,
                active: false,
                peak: 0.0,
                lift: false,
                notes: [(0, false); 32],
                pending: 0,
                rest: false,
            })
            .collect();
        // (Each voice's lock taken once here, off the audio thread: on macOS the standard
        // library may make a mutex's lock on its first use; history.md, "The worker locks".
        // The spares' are taken as they are made.)
        for s in &self.slots {
            drop(s.play.lock());
        }
        self.spares.make(voice_rate);
        self.apply();
        self.press_held();
    }

    /// The keys held pressed again on the instruments' voices just made (POLY's voices take
    /// the next keys).
    fn press_held(&self) {
        for k in 0..self.instruments() {
            if let Ok(mut p) = self.slots[k].play.lock() {
                for (key, _) in self.held.iter().enumerate().filter(|(_, h)| **h) {
                    p.note(key as i32, true);
                }
            }
        }
    }

    /// `workers` threads of the plug-in's own to share POLY's voices with the caller's
    /// thread, made audio threads for blocks of `period` (None: left ordinary, for
    /// measurements that run as fast as they go; decisions.md R11); 0, none. Held only while
    /// POLY is on (R21): started here if it is, else when it is switched on, off the audio
    /// thread ([`Crew::serve`]); let go when it is switched off. Not on the audio thread.
    /// Workers already running for these voices, as many and for the same period, are kept
    /// while POLY stays on (a host's state loaded into a running instance prepares it again:
    /// decisions.md R18). Returns how many run (none with POLY off, or if the system would not
    /// start them).
    pub fn start_workers(&mut self, workers: usize, period: Option<Duration>) -> usize {
        if self.pool.is_some() && self.workers == (workers, period) && self.many() {
            return self.workers();
        }
        self.stop_workers();
        self.workers = (workers, period);
        if workers == 0 || self.slots.is_empty() {
            return 0;
        }
        let voices: Vec<Shared> = self.slots.iter().map(|s| s.play.clone()).collect();
        self.plan = self.crew.plan(&voices, workers, period);
        if self.many() {
            self.pool = self.crew.start(self.plan);
            self.ask = Ask::Answered;
            #[cfg(test)]
            {
                self.pool_starts += 1;
            }
        }
        self.workers()
    }

    /// POLY's workers stopped, here, their share of the process's budget given back, and none
    /// started again until [`Engine::start_workers`] (a plug-in deactivated: decisions.md
    /// R21). Not on the audio thread.
    pub fn stop_workers(&mut self) {
        self.pool = None;
        self.crew.clear();
        (self.plan, self.ask) = (0, Ask::Not);
    }

    /// At a block's start, on the audio thread: POLY's workers as POLY asks (decisions.md
    /// R21), without waiting. POLY switched on, a pool asked of the helper thread
    /// ([`Engine::take_serving`]) and taken once it is ready, every voice played here until
    /// then; switched off, the pool let go, to be stopped there.
    fn follow_poly(&mut self) {
        let on = self.many();
        if self
            .crew
            .follow(on, self.plan, &mut self.pool, &mut self.ask)
        {
            self.serve = true;
        }
    }

    /// Where POLY's workers are started and stopped off the audio thread ([`Crew::serve`]).
    pub fn crew(&self) -> Arc<Crew> {
        self.crew.clone()
    }

    /// Whether POLY's workers are to be started or stopped ([`Crew::serve`], off the audio
    /// thread) since this was last asked.
    pub fn take_serving(&mut self) -> bool {
        std::mem::take(&mut self.serve)
    }

    /// The clean voices put in voices' places ([`Spares`]): for mending off the audio thread.
    pub fn spares(&self) -> Arc<Spares> {
        self.spares.clone()
    }

    /// Whether the spares are to be mended ([`Spares::mend`], off the audio thread) since this
    /// was last asked.
    pub fn take_mending(&mut self) -> bool {
        std::mem::take(&mut self.mend)
    }

    /// The workers sharing POLY's voices.
    pub fn workers(&self) -> usize {
        self.pool.as_ref().map_or(0, Pool::workers)
    }

    /// When the present block's voices must be done by: a voice a worker has not finished
    /// then is silent for the run, not waited for (None: waited for). The plug-in sets it at
    /// each block's start, when POLY's workers also follow POLY ([`Engine::follow_poly`]).
    pub fn set_deadline(&mut self, deadline: Option<Instant>) {
        self.deadline = deadline;
        self.follow_poly();
        if let Some(p) = &self.pool {
            p.block();
        }
    }

    /// The controls for the samples that follow. POLY, UNISON or DOUBLE switched lifts every
    /// key first (the notes then sounding stop; the next notes play in the new way).
    pub fn set(&mut self, c: &Controls) {
        if *c != self.controls {
            self.rest.wake();
            if c.poly != self.controls.poly
                || c.potato != self.controls.potato
                || c.unison != self.controls.unison
                || (c.unison && c.voices != self.controls.voices)
                || (c.double > 0.0) != (self.controls.double > 0.0)
            {
                self.lift_all();
            }
            if c.voices < self.controls.voices {
                // The voices past the new count let go (they run on until silent).
                for s in self.slots.iter_mut().skip(c.voices.max(1)) {
                    s.gate = false;
                }
            }
            self.controls = *c;
            self.gain = output_gain(c.volume, c.main_output)
                * if c.lock { lock_trim(&c.panel) } else { 1.0 }
                * unison_trim_of(c)
                * double_trim(c)
                * level_gain(c.level);
            self.auto = auto_trim(c, &self.curve);
            self.apply();
        }
    }

    /// The one instrument's key pressed or released (POLY off), on the first voice (once it
    /// is free, if a worker late from an earlier run holds it); with UNISON, on each of its
    /// instruments.
    fn one_note(&mut self, key: i32, on: bool) {
        for k in 0..self.instruments().max(1) {
            let s = &mut self.slots[k];
            if with_voice(s, |p| p.note(key, on)).is_none() && s.pending < s.notes.len() {
                s.notes[s.pending] = (key, on);
                s.pending += 1;
            }
            if self.controls.unison && on {
                s.active = true;
            }
        }
    }

    /// Whether more than the one instrument plays: POLY's voices or UNISON's.
    fn many(&self) -> bool {
        self.controls.poly || self.controls.unison
    }

    /// The voices that take every key through their keyboard circuits, as the one instrument
    /// does: UNISON's VOICES, POLY off's one, POLY's none.
    fn instruments(&self) -> usize {
        if self.controls.unison {
            self.voices()
        } else if self.controls.poly {
            0
        } else {
            1.min(self.slots.len())
        }
    }

    pub fn event(&mut self, e: Event) {
        self.rest.wake();
        match e {
            Event::Note { key, on } => {
                let Some(h) = self.held.get_mut(usize::from(key)) else {
                    return;
                };
                *h = on;
                if self.slots.is_empty() {
                    return;
                }
                if self.controls.unison || !self.controls.poly {
                    self.one_note(i32::from(key), on);
                } else if on {
                    self.poly_on(key);
                } else {
                    for s in &mut self.slots {
                        if s.gate && s.note == key {
                            s.gate = false;
                        }
                    }
                }
            }
            Event::PitchBend(b) => {
                self.bend = b.clamp(-1.0, 1.0);
                self.apply();
            }
            Event::Modulation(m) => {
                self.modulation = m.clamp(0.0, 1.0);
                self.apply();
            }
            Event::ResetControllers => {
                (self.bend, self.modulation) = (0.0, 0.0);
                self.apply();
            }
            Event::AllNotesOff => self.release_all(),
            Event::AllSoundOff => self.all_sound_off(),
        }
    }

    /// All Sound Off (decisions.md R18): every key released and the output faded out over
    /// [`HUSH_FADE`]; then each voice that sounded is put to rest ([`Engine::settle`]) and the
    /// output fades back in. A voice taken for a note meanwhile plays on.
    fn all_sound_off(&mut self) {
        self.release_all();
        let (many, instruments) = (self.many(), self.instruments());
        for (k, s) in self.slots.iter_mut().enumerate() {
            s.rest = if many {
                s.active || k < instruments
            } else {
                k == 0
            };
        }
        self.hush = !self.slots.is_empty();
        self.hush_waited = 0;
    }

    /// Once All Sound Off has faded the output out, the voices it marked put to rest, a clean
    /// voice in each one's place ([`Engine::restore`]), and the output let back in; a voice
    /// whose clean voice is not yet made is waited for, silent, up to [`HUSH_WAIT`]. At the
    /// start of each run (one sample, or the samples between events).
    fn settle(&mut self) {
        if !self.hush || self.fade > 0.0 {
            return;
        }
        let poly = self.many();
        let mut waiting = false;
        for k in 0..self.slots.len() {
            if !self.slots[k].rest {
                continue;
            }
            if poly && self.slots[k].gate {
                self.slots[k].rest = false;
                continue;
            }
            let play = self.slots[k].play.clone();
            let rested = play.try_lock().is_ok_and(|mut p| self.restore(k, &mut p));
            let s = &mut self.slots[k];
            if rested {
                s.rest = false;
                if poly {
                    (s.active, s.peak) = (false, 0.0);
                }
            } else {
                waiting = true;
            }
        }
        if !waiting || self.hush_waited as f64 >= HUSH_WAIT * self.rate {
            self.hush = false;
            for s in &mut self.slots {
                s.rest = false;
            }
        }
    }

    /// Voice `k` put back to rest without allocating (decisions.md R18): a clean voice in its
    /// place from its spare ([`Spares`]), its noise seeded, the engine's panel, its key up
    /// (pressed again if its gate is high; POLY off, the keys held pressed again). False if
    /// the spare has not been made again since it was last used. Either way the spares are
    /// then to be mended ([`Engine::take_mending`]).
    fn restore(&mut self, k: usize, p: &mut Playing) -> bool {
        if p.potato {
            // (The light voices are put to rest in place: nothing to make.)
            p.light.rest();
            p.twin.light.rest();
            light_seeded(&mut p.light, self.seed, k);
            light_seeded(&mut p.twin.light, self.seed, k + POLY_VOICES.2);
        } else if !self.restore_voices(k, p) {
            return false;
        }
        p.key.rest();
        p.twin.key.rest();
        (p.twin.live, p.twin.quiet, p.twin.pan) = (false, 0, None);
        (p.broken, p.drop) = (false, false);
        let s = &mut self.slots[k];
        (s.lift, s.pending, s.shown) = (false, 0, false);
        // (The panel before the keys: DOUBLE's, which the twin takes keys by.)
        self.voice_panel(p, &self.panel());
        if k < self.instruments() {
            for (key, _) in self.held.iter().enumerate().filter(|(_, h)| **h) {
                p.note(key as i32, true);
            }
        }
        true
    }

    /// Place `k`'s circuit voice and twin swapped for clean spares, seeded as made: whether
    /// they were (not if the spares are not clean or not at the voices' rate).
    fn restore_voices(&mut self, k: usize, p: &mut Playing) -> bool {
        self.mend = true;
        let rate = p.voice.rate();
        let Ok(mut spare) = self.spares.places[k].try_lock() else {
            return false;
        };
        if !spare.clean {
            return false;
        }
        let Some(clean) = spare.voice.as_mut().filter(|v| v.rate() == rate) else {
            return false;
        };
        std::mem::swap(&mut p.voice, clean);
        if let Some(twin) = spare.twin.as_mut().filter(|v| v.rate() == rate) {
            std::mem::swap(&mut p.twin.voice, &mut **twin);
        }
        spare.clean = false;
        drop(spare);
        seeded(&mut p.voice, self.seed, k);
        seeded(&mut p.twin.voice, self.seed, k + POLY_VOICES.2);
        true
    }

    /// A POLY note: the voice that last played the key if it is free or letting go (a chord
    /// struck again stays where it was in the field), else the free voice whose last note began
    /// longest ago (the notes go round the voices, to either side in turn), else the one let
    /// go longest ago, else the one held longest (its key lifted and pressed again: `PolyKey`)
    /// (the CA-74's R27).
    fn poly_on(&mut self, key: u8) {
        let n = self.voices();
        let pick = (0..n)
            .filter(|&k| {
                let s = &self.slots[k];
                s.note == key && s.age > 0 && !s.gate
            })
            .max_by_key(|&k| self.slots[k].age)
            .or_else(|| {
                (0..n)
                    .filter(|&k| !self.slots[k].active)
                    .min_by_key(|&k| self.slots[k].age)
            })
            .or_else(|| {
                (0..n)
                    .filter(|&k| !self.slots[k].gate)
                    .min_by_key(|&k| self.slots[k].age)
            })
            .or_else(|| (0..n).min_by_key(|&k| self.slots[k].age))
            .unwrap_or(0);
        self.counter += 1;
        let s = &mut self.slots[pick];
        s.drop = s.active && s.shown;
        (s.note, s.gate, s.age, s.active, s.peak) = (key, true, self.counter, true, 0.0);
    }

    /// The voices POLY plays.
    fn voices(&self) -> usize {
        self.controls
            .voices
            .clamp(POLY_VOICES.0, POLY_VOICES.2)
            .min(self.slots.len())
    }

    /// Every held key released (the contours then run out as the DECAY switch has them), and
    /// the one instrument's pattern started again from the first place (decisions.md
    /// R62): the host resets the plug-in, and sends All Notes Off as its transport
    /// stops, so that a part plays and renders its places alike each time.
    pub fn release_all(&mut self) {
        self.rest.wake();
        for key in 0..self.held.len() {
            if std::mem::take(&mut self.held[key]) && self.instruments() > 0 {
                self.one_note(key as i32, false);
            }
        }
        for s in &mut self.slots {
            s.gate = false;
        }
        self.restart = true;
    }

    /// Every voice's key lifted at once and the held keys forgotten (POLY switched).
    fn lift_all(&mut self) {
        for k in 0..self.slots.len() {
            let s = &mut self.slots[k];
            if with_voice(s, Playing::lift).is_none() {
                s.lift = true;
            }
            s.gate = false;
        }
        if self.instruments() > 0 {
            for key in 0..self.held.len() {
                if self.held[key] {
                    self.one_note(key as i32, false);
                }
            }
        }
        self.held = [false; 128];
    }

    /// One output sample (the left and right channels together, as on a mono output) with
    /// `ext` at EXTERNAL INPUT (1.0 the same volts as the output's 1.0).
    pub fn tick(&mut self, ext: f32) -> f32 {
        let (l, r) = self.tick_stereo(ext);
        0.5 * (l + r)
    }

    /// One output sample, left and right, with `ext` at EXTERNAL INPUT (a sample that is not
    /// finite taken as 0: decisions.md R18); silence before [`Engine::prepare`]. With POWER
    /// off, once faded out, the voices rest (the MIDI input still reaches them).
    pub fn tick_stereo(&mut self, ext: f32) -> (f32, f32) {
        if self.rest.asleep {
            if self.dormant(&[ext]) {
                return (0.0, 0.0);
            }
            self.rest.wake();
        }
        self.settle();
        let on = self.controls.power;
        if (!on && self.fade == 0.0) || self.slots.is_empty() {
            return (0.0, 0.0);
        }
        let ext = if ext.is_finite() { ext } else { 0.0 };
        let rate = self.voice_rate();
        let (l, r) = match self.over.take() {
            None => self.sample(f64::from(ext), rate),
            Some(mut o) => {
                let mut ins = [0.0; 8];
                o.up.push(f64::from(ext), &mut ins[..o.factor]);
                let (mut l, mut r) = (0.0, 0.0);
                for x in &ins[..o.factor] {
                    let (a, b) = self.sample(*x, rate);
                    if let Some(d) = o.down[0].push(a) {
                        l = d;
                    }
                    if let Some(d) = o.down[1].push(b) {
                        r = d;
                    }
                }
                self.over = Some(o);
                (l, r)
            }
        };
        let (l, r) = self.faded(l, r);
        if !self.many() {
            self.rest.heard(&[l], &[r]);
        }
        (l, r)
    }

    /// `left.len()` output samples, left and right, with `ext[i]` at EXTERNAL INPUT (an empty
    /// `ext`: nothing there; a sample that is not finite, 0) and no event among them: the same
    /// samples, to the bit, as as many [`Engine::tick_stereo`]. With POLY the voices play
    /// [`CHUNK`] samples at a time, one after another (a voice's state, some 30 KB, then stays
    /// in the processor's nearest cache), or shared with the workers, and are summed in their
    /// order (decisions.md R11). The one instrument asleep, and nothing at EXTERNAL INPUT:
    /// silence, as as many samples would give, without them (decisions.md R68).
    pub fn render(&mut self, ext: &[f32], left: &mut [f32], right: &mut [f32]) {
        let len = left.len().min(right.len());
        if self.rest.asleep {
            if self.dormant(&ext[..ext.len().min(len)]) {
                left[..len].fill(0.0);
                right[..len].fill(0.0);
                return;
            }
            self.rest.wake();
        }
        self.render_voices(ext, &mut left[..len], &mut right[..len]);
        if !self.many() {
            self.rest.heard(&left[..len], &right[..len]);
        }
    }

    /// Whether the one instrument, asleep, stays so for samples with `ext` at EXTERNAL INPUT:
    /// POWER on and faded in, no All Sound Off fading, and nothing there.
    fn dormant(&self, ext: &[f32]) -> bool {
        !self.many()
            && self.controls.power
            && !self.hush
            && self.fade == 1.0
            && ext
                .iter()
                .all(|x| !x.is_finite() || f64::from(x.abs()) < SILENT)
    }

    /// [`Engine::render`]'s samples, played.
    fn render_voices(&mut self, ext: &[f32], left: &mut [f32], right: &mut [f32]) {
        let len = left.len().min(right.len());
        let at = |i: usize| ext.get(i).copied().filter(|x| x.is_finite()).unwrap_or(0.0);
        // (At QUALITY's LO the one instrument too plays a run at a time: its light voice costs
        // too little for a lock and a place worked out each sample to be nothing beside it.)
        let runs = self.many() || self.controls.potato;
        if !runs || self.over.is_some() || self.slots.is_empty() {
            for i in 0..len {
                (left[i], right[i]) = self.tick_stereo(at(i));
            }
            return;
        }
        let rate = self.voice_rate();
        let mix = self.mix();
        let mut start = 0;
        while start < len {
            self.settle();
            let mut n = (len - start).min(CHUNK);
            // The samples the voices play: all of them, unless POWER is off, when they rest
            // from the sample the fade reaches 0. All Sound Off's run ends at that sample, its
            // voices put to rest from the next, as a sample at a time has them.
            let mut run = n;
            if !self.controls.power || self.hush {
                let (mut fade, mut out) = (self.fade, 0);
                while out < n && fade != 0.0 {
                    fade = self.fade_on(fade);
                    out += 1;
                }
                if self.hush && self.fade > 0.0 {
                    n = out;
                }
                run = if self.controls.power { n } else { out };
            }
            let mut ins = [0.0f64; CHUNK];
            for (k, x) in ins.iter_mut().enumerate().take(run) {
                *x = f64::from(at(start + k));
            }
            let (mut l, mut r) = ([0.0f64; CHUNK], [0.0f64; CHUNK]);
            if run > 0 {
                if self.many() {
                    self.run(run, &ins, rate, &mix, true, &mut l, &mut r);
                } else {
                    self.run_one(run, &ins, rate, &mix, &mut l, &mut r);
                }
            }
            for k in 0..n {
                (left[start + k], right[start + k]) = if k < run {
                    self.faded(l[k], r[k])
                } else {
                    (0.0, 0.0)
                };
            }
            start += n;
        }
    }

    /// The one instrument's voice (POLY and UNISON off) over `n` samples, as as many samples
    /// of [`Engine::tick_stereo`] play it, its lock taken once: silent for the run if a worker
    /// late from an earlier run still holds it.
    fn run_one(
        &mut self,
        n: usize,
        ext: &[f64; CHUNK],
        rate: f64,
        mix: &Mix,
        l: &mut [f64; CHUNK],
        r: &mut [f64; CHUNK],
    ) {
        let play = self.slots[0].play.clone();
        let Ok(mut p) = play.try_lock() else {
            return;
        };
        self.bring_up(0, &mut p);
        p.play_from(&ext[..n], rate, mix, self.one_glide(mix));
        self.turn = p.turn.unwrap_or(0);
        l[..n].copy_from_slice(&p.out[0][..n]);
        r[..n].copy_from_slice(&p.out[1][..n]);
    }

    /// The one instrument's place's glide a sample: [`TURN_GLIDE`]'s while it scatters (its
    /// knobs' moves too), else [`GLIDE`]'s.
    fn one_glide(&self, mix: &Mix) -> f64 {
        if mix.scatter {
            self.turn_glide
        } else {
            self.glide
        }
    }

    /// POLY's sounding voices over `n` samples, summed into `l` and `r` in their order: each
    /// given its gate, note and drop and brought up to the engine's panel, then played here,
    /// or shared with the workers (with `share`, and a run long enough to be worth it). A
    /// voice a worker late from an earlier run still holds sits the run out.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        n: usize,
        ext: &[f64; CHUNK],
        rate: f64,
        mix: &Mix,
        share: bool,
        l: &mut [f64; CHUNK],
        r: &mut [f64; CHUNK],
    ) {
        let mut list = [0usize; MAX];
        let mut count = 0;
        for k in 0..self.slots.len() {
            if !self.slots[k].active || count == MAX {
                continue;
            }
            let play = self.slots[k].play.clone();
            let Ok(mut p) = play.try_lock() else {
                self.slots[k].peak = self.slots[k].peak.max(SILENT);
                continue;
            };
            self.bring_up(k, &mut p);
            self.slots[k].hand(&mut p);
            drop(p);
            list[count] = k;
            count += 1;
        }
        let deadline = self.deadline;
        // (At QUALITY's LO a voice costs less than waking a worker for it: every voice here.)
        let share = share && !self.controls.potato;
        let played = match &mut self.pool {
            Some(pool) if share && count > 1 && n >= SHARED_RUN => {
                pool.play(&list[..count], n, ext, rate, mix, deadline)
            }
            _ => {
                let mut played = [false; MAX];
                for (ok, &k) in played.iter_mut().zip(&list[..count]) {
                    if let Ok(mut p) = self.slots[k].play.try_lock() {
                        p.play(n, ext, rate, mix);
                        *ok = true;
                    }
                }
                played
            }
        };
        for (&ok, &k) in played.iter().zip(&list[..count]) {
            let s = &mut self.slots[k];
            s.ran(ok);
            // (A voice not heard this run is not known to have fallen silent: it is not freed
            // for it at the block's end. decisions.md R18.)
            let p = if ok { s.play.try_lock().ok() } else { None };
            let Some(p) = p else {
                s.peak = s.peak.max(SILENT);
                continue;
            };
            for i in 0..n {
                l[i] += p.out[0][i];
                r[i] += p.out[1][i];
            }
            s.peak = s.peak.max(p.peak);
        }
    }

    /// A voice brought up to what the engine asked of it meanwhile: a clean voice in its place
    /// if it broke ([`Engine::restore`]), the panel, its key lifted and the one instrument's
    /// keys.
    fn bring_up(&mut self, k: usize, p: &mut Playing) {
        if p.broken {
            self.restore(k, p);
        }
        // (The panel before the keys: DOUBLE's, which the twin takes keys by.)
        if p.panel_gen != self.panel_gen {
            self.voice_panel(p, &self.panel());
        }
        if k == 0 && std::mem::take(&mut self.restart) {
            p.turn = None;
        }
        let s = &mut self.slots[k];
        if std::mem::take(&mut s.lift) {
            p.lift();
        }
        for &(key, on) in &s.notes[..s.pending] {
            p.note(key, on);
        }
        s.pending = 0;
    }

    /// A sample of the voices' sum, left and right, through MAIN OUTPUT's gain and POWER's
    /// or All Sound Off's fade (moved a sample on). Never a sample that is not finite
    /// (decisions.md R18): 0 instead, and the resamplers, which may hold one, start again.
    fn faded(&mut self, l: f64, r: f64) -> (f32, f32) {
        self.fade = self.fade_on(self.fade);
        if self.hush && self.fade == 0.0 {
            self.hush_waited += 1;
        }
        // (As the one voice was: the gain, then the fade; AUTO GAIN's correction, gliding,
        // between them, 1 without DRIVE.)
        self.auto_now += (self.auto - self.auto_now) * self.auto_share;
        let (l, r) = (
            (l * self.gain * self.auto_now * self.fade) as f32,
            (r * self.gain * self.auto_now * self.fade) as f32,
        );
        if l.is_finite() && r.is_finite() {
            return (l, r);
        }
        if let Some(o) = &mut self.over {
            o.up.reset();
            o.down.iter_mut().for_each(Decimator::reset);
        }
        (0.0, 0.0)
    }

    /// The output's gain under POWER and All Sound Off a sample after `fade`: rising while
    /// POWER is on, else falling (All Sound Off's faster).
    fn fade_on(&self, fade: f64) -> f64 {
        if self.controls.power && !self.hush {
            (fade + self.fade_step).min(1.0)
        } else if self.hush {
            (fade - self.hush_step).max(0.0)
        } else {
            (fade - self.fade_step).max(0.0)
        }
    }

    /// How ENTROPY, SPREAD, DOUBLE and DRIVE stand for the voices' samples.
    fn mix(&self) -> Mix {
        mix_of(&self.controls, self.voices())
    }

    /// AUTO GAIN's curve for the sound (`drive.rs`), the correction following it.
    pub fn set_curve(&mut self, curve: Curve<STEPS>) {
        self.curve = curve;
        self.auto = auto_trim(&self.controls, &curve);
    }

    /// AUTO GAIN's curve the engine plays with.
    pub fn curve(&self) -> Curve<STEPS> {
        self.curve
    }

    /// Where AUTO GAIN's measurements are asked for and kept (`drive.rs`), told the voices'
    /// rate now and whenever the engine is prepared. Not on the audio thread.
    pub fn calibrate_with(&mut self, c: Calibration) {
        if self.rate > 0.0 {
            c.set_rate(self.voice_rate());
        }
        self.calibration = Some(c);
    }

    /// One sample of the voices at their rate (`rate` Hz): left and right.
    fn sample(&mut self, ext: f64, rate: f64) -> (f64, f64) {
        let mix = self.mix();
        if !self.many() {
            let play = self.slots[0].play.clone();
            let Ok(mut p) = play.try_lock() else {
                return (0.0, 0.0);
            };
            self.bring_up(0, &mut p);
            // (The one voice at its turn's place, or DOUBLE's pair; at SPREAD 0 in the centre,
            // both sides whole. The voice's key is the keyboard's.)
            p.play_from(&[ext], rate, &mix, self.one_glide(&mix));
            self.turn = p.turn.unwrap_or(0);
            return (p.out[0][0], p.out[1][0]);
        }
        let mut ins = [0.0f64; CHUNK];
        ins[0] = ext;
        let (mut l, mut r) = ([0.0f64; CHUNK], [0.0f64; CHUNK]);
        self.run(1, &ins, rate, &mix, false, &mut l, &mut r);
        (l[0], r[0])
    }

    fn voice_rate(&self) -> f64 {
        self.rate * self.over.as_ref().map_or(1, |o| o.factor) as f64
    }

    /// After a block of `len` samples: the OVERLOAD lamp (0 off, 1 fully lit), its peak held
    /// and falling with a time constant of [`LAMP_FALL`]; POLY's voices let go and fallen
    /// silent over the block freed.
    pub fn end_block(&mut self, len: usize) -> f32 {
        if !self.many() && !self.rest.asleep {
            let state = self
                .slots
                .first()
                .and_then(|s| with_voice(s, |p| slow_state(p)));
            self.rest.after(state.flatten(), len);
        }
        let mut peak = 0.0f64;
        // (UNISON's instruments, which take their keys straight, are held while a key is.)
        let held = if self.controls.unison && self.held.iter().any(|h| *h) {
            self.instruments()
        } else {
            0
        };
        for (k, s) in self.slots.iter_mut().enumerate() {
            let overload = |p: &mut Playing| {
                if p.potato {
                    let twin = p.twin.light.take_overload_peak();
                    return p.light.take_overload_peak().max(twin);
                }
                let twin = p.twin.voice.take_overload_peak();
                p.voice.take_overload_peak().max(twin)
            };
            if let Some(p) = with_voice(s, overload) {
                peak = peak.max(p);
            }
            if s.active && !s.gate && s.peak < SILENT && k >= held {
                s.active = false;
            }
            if let Some(l) = self.levels.get_mut(k) {
                *l = if s.peak.is_finite() {
                    s.peak as f32
                } else {
                    0.0
                };
            }
            s.peak = 0.0;
        }
        let fall = (-(len as f32) / (self.rate as f32 * LAMP_FALL)).exp();
        let fall = if fall.is_finite() { fall } else { 0.0 };
        self.lamp = (self.lamp * fall).max(peak.clamp(0.0, 1.0) as f32);
        self.lamp
    }

    pub fn lamp(&self) -> f32 {
        self.lamp
    }

    /// Each voice's output's peak over the last block, by its slot (POLY's and UNISON's voices;
    /// the one instrument's is the first).
    pub fn levels(&self) -> &[f32; LEVELS] {
        &self.levels
    }

    /// Which voices sound now, a bit a voice (the editor's display of where they sound, A6):
    /// POLY's and UNISON's sounding, the one instrument while a key is held (scattering, the
    /// bit of the place its last attack took).
    pub fn sounding_mask(&self) -> u32 {
        if self.many() {
            self.slots
                .iter()
                .enumerate()
                .filter(|(_, s)| s.active)
                .fold(0, |m, (k, _)| m | 1 << k)
        } else {
            let at = if self.mix().scatter { self.turn } else { 0 };
            u32::from(self.held.iter().any(|h| *h)) << at.min(LEVELS - 1)
        }
    }

    /// Whether the one instrument sleeps: silent and settled, not played until an event, a
    /// control changed or a sound at EXTERNAL INPUT wakes it (decisions.md R68).
    pub fn asleep(&self) -> bool {
        self.rest.asleep
    }

    /// The one instrument never to sleep (or to again): for measuring what it costs waiting.
    pub fn never_sleep(&mut self, never: bool) {
        self.rest.never = never;
        self.rest.wake();
    }

    /// The voices sounding now (POLY or UNISON on), or 1 (off).
    pub fn sounding(&self) -> usize {
        if self.many() {
            self.slots.iter().filter(|s| s.active).count()
        } else {
            1
        }
    }

    /// The wheels where the parameters and the MIDI input leave them: PITCH -1..1 and
    /// MODULATION 0..1.
    pub fn wheels(&self) -> (f64, f64) {
        let p = self.panel();
        (p.pitch_wheel, p.mod_wheel)
    }

    /// Where a MIDI keyboard's wheels move the panel's, as the panel draws them: its pitch
    /// bend's share of the PITCH wheel's travel (0..1, 0.5 at rest) and its modulation wheel
    /// (0..1).
    pub fn midi_wheels(&self) -> (f32, f32) {
        let share = (self.controls.bend_range / PITCH_WHEEL_SEMITONES).clamp(0.0, 1.0);
        let share = if share.is_finite() { share as f32 } else { 0.0 };
        (
            0.5 + self.bend * share / 2.0,
            ca72::modulation::midi_wheel(f64::from(self.modulation)) as f32,
        )
    }

    /// The panel the voices play: the controls', in Potato (at ULTRA, No Compromises), the
    /// MIDI input's bend moving
    /// the PITCH wheel by its share of the wheel's travel and its modulation wheel holding
    /// MODULATION up to where the hardware reference's MIDI puts it
    /// (`ca72::modulation::midi_wheel`).
    fn panel(&self) -> Panel {
        let c = &self.controls;
        let share = (c.bend_range / PITCH_WHEEL_SEMITONES).clamp(0.0, 1.0);
        let share = if share.is_finite() { share } else { 0.0 };
        let mut p = c.panel;
        p.quality = if c.ultra {
            Quality::NoCompromises
        } else {
            Quality::Potato
        };
        p.pitch_wheel = (p.pitch_wheel + f64::from(self.bend) * share).clamp(-1.0, 1.0);
        p.mod_wheel = p
            .mod_wheel
            .max(ca72::modulation::midi_wheel(f64::from(self.modulation)));
        p
    }

    /// How every voice's preamplifier is solved inside FEEDBACK's loop (for comparing the
    /// models: decisions.md R11).
    pub fn set_in_loop(&mut self, m: ca72::preamp::InLoop) {
        self.in_loop = m;
        self.apply();
    }

    /// The panel to every voice (each with its tolerances under ENTROPY): a new generation,
    /// which a voice a worker still holds takes before it next plays.
    fn apply(&mut self) {
        self.panel_gen = self.panel_gen.wrapping_add(1);
        let p = self.panel();
        for k in 0..self.slots.len() {
            let play = self.slots[k].play.clone();
            if let Ok(mut v) = play.try_lock() {
                self.voice_panel(&mut v, &p);
            }
        }
    }

    /// A voice's panel `p` with its tolerances, FEEDBACK and its loop's preamplifier; its
    /// twin's with the twin's tolerances; and whether DOUBLE is on.
    fn voice_panel(&self, v: &mut Playing, p: &Panel) {
        let entropy = self.controls.entropy * ENTROPY_DEPTH;
        let feedback = self.controls.feedback.clamp(0.0, 1.0);
        for (voice, character) in [
            (&mut v.voice, &v.character),
            (&mut v.twin.voice, &v.twin.character),
        ] {
            voice.feedback = feedback;
            voice.in_loop = self.in_loop;
            let pv = if entropy > 0.0 {
                entropy_panel(character, *p, entropy)
            } else {
                *p
            };
            if voice.panel != pv {
                voice.panel = pv;
            }
        }
        // (The light voices only at LO: their laws are worked out again as their panel
        // changes, which the circuit's voices have no need of.)
        v.potato = self.controls.potato;
        if v.potato {
            for (light, voice) in [(&mut v.light, &v.voice), (&mut v.twin.light, &v.twin.voice)] {
                light.feedback = feedback;
                light.set_panel(&voice.panel);
            }
        }
        v.double = self.controls.double > 0.0;
        v.panel_gen = self.panel_gen;
    }
}

/// How many workers share POLY's voices on this machine (decisions.md R11, R39):
/// [`workers_for`] its processors.
pub fn default_workers() -> usize {
    workers_for(std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get))
}

/// The workers for a machine of `n` processors (decisions.md R11, R39): a third of them less
/// two, at most four (four on the Mac's 14, two on 8), and at least one from 3 up, so that on a
/// machine of 3 or 4 (an older quad-core) POLY's voices are not all on the host's thread while
/// its other processors idle; none on 1 or 2, where the host's thread and a worker would be all
/// the machine has.
pub fn workers_for(n: usize) -> usize {
    if n < 3 {
        0
    } else {
        (n.saturating_sub(2) / 3).clamp(1, 4)
    }
}

/// When a block's voices must be done by, as a share of its period from its start: a voice a
/// worker has not finished then is silent for its run (decisions.md R11). Normally they are
/// done long before.
pub const DEADLINE: f64 = 0.75;

/// The shortest run worth sharing with the workers, samples (waking them takes some
/// microseconds).
const SHARED_RUN: usize = 8;

/// Samples [`Engine::render`] plays a voice for at a time.
pub const CHUNK: usize = 128;

/// ENTROPY, SPREAD, DOUBLE and DRIVE as the voices' samples take them: ENTROPY's depth, the
/// oscillators' (their floor with it, unless LOCK), the depth the character's offsets are
/// drawn at, SPREAD and INNER, SCATTER's placement (none for one voice alone with DOUBLE, its
/// pair at the edge) and the voices it places among (VOICES), DOUBLE's detune between a note's
/// two voices, cents (0: one voice a note), DRIVE's gain into the filter (1: the circuit), and
/// whether the one instrument scatters (POLY, UNISON and DOUBLE off: its voice at the next of
/// VOICES's places at each attack; decisions.md R62).
#[derive(Debug, Clone, Copy)]
pub struct Mix {
    entropy: f64,
    oscillators: f64,
    at: f64,
    spread: f64,
    inner: f64,
    placement: Option<Placement>,
    voices: usize,
    double: f64,
    drive: f64,
    scatter: bool,
}

/// [`Mix`]'s values, as the workers are handed them.
pub const MIX_LEN: usize = 10;

impl Mix {
    pub fn to_array(self) -> [f64; MIX_LEN] {
        [
            self.entropy,
            self.oscillators,
            self.at,
            self.spread,
            self.inner,
            self.placement.map_or(-1.0, |p| p.index() as f64),
            self.voices as f64,
            self.double,
            self.drive,
            f64::from(u8::from(self.scatter)),
        ]
    }

    pub fn from_array(
        [
            entropy,
            oscillators,
            at,
            spread,
            inner,
            placement,
            voices,
            double,
            drive,
            scatter,
        ]: [f64; MIX_LEN],
    ) -> Mix {
        Mix {
            entropy,
            oscillators,
            at,
            spread,
            inner,
            placement: (placement >= 0.0).then(|| Placement::from_index(placement as usize)),
            voices: voices as usize,
            double,
            drive,
            scatter: scatter > 0.0,
        }
    }
}

/// A voice's jacks for its next sample: `ext` at EXTERNAL INPUT, and its character's offsets
/// (a sample on).
fn jacks(c: &mut Character, ext: f64, rate: f64, m: &Mix) -> Jacks {
    let mut j = Jacks {
        ext,
        ..Jacks::default()
    };
    if m.at > 0.0 {
        let (cents, cutoff) = c.offsets(m.at, rate);
        j.detune = cents.map(|c| c * m.oscillators / m.at);
        j.cutoff = cutoff * m.entropy / m.at;
    }
    j
}

/// The share of a block's period the voices may take: the rest is the host's, its other
/// plug-ins' and the machine's own jitter (the DAW's figure).
pub const HEADROOM: f64 = 0.6;

/// How many POLY voices this machine plays in real time at `rate` Hz in blocks of `block`
/// frames with `workers` workers (decisions.md, "POLY in real time"; R11): the engine itself,
/// its workers made audio threads, played paced at the block's period on an audio thread of
/// its own for a moment, as a host plays it: chords of as many notes as voices tried, a new
/// one every few blocks (each taking voices still sounding, as legato), gliding, oscillators
/// 2 and 3 on, oscillator 3 on the filter through the MODULATION wheel, ENTROPY at half. The
/// count is the most voices whose blocks where the keys change take, at their median, no
/// more than [`HEADROOM`] of the period. With `feedback`, FEEDBACK on (at half, into the
/// external input at half VOLUME). About a fifth of a second a count tried, ten first, then
/// the estimate from it downward; once a process for each rate, block, FEEDBACK and number
/// of workers. Not on the audio thread.
pub fn realtime_voices(rate: f64, block: usize, feedback: bool, workers: usize) -> u32 {
    /// A measurement: the rate's bits, the block, FEEDBACK, the workers, the voices.
    type Measured = (u64, usize, bool, usize, u32);
    static MEASURED: std::sync::OnceLock<Mutex<Vec<Measured>>> = std::sync::OnceLock::new();
    let cache = MEASURED.get_or_init(|| Mutex::new(Vec::new()));
    let key = (rate.to_bits(), block, feedback, workers);
    if let Some(n) = cache
        .lock()
        .ok()
        .and_then(|c| c.iter().find(|e| (e.0, e.1, e.2, e.3) == key).map(|e| e.4))
    {
        return n;
    }
    let n = std::thread::scope(|s| {
        std::thread::Builder::new()
            .name("ca72-measure".to_owned())
            .spawn_scoped(s, || measure_voices(rate, block, feedback, workers))
            .ok()
            .and_then(|h| h.join().ok())
            .unwrap_or(0)
    });
    if let Ok(mut c) = cache.lock() {
        c.push((key.0, key.1, key.2, key.3, n));
    }
    n
}

/// [`realtime_voices`]' measurement, on a thread of its own that it makes an audio thread.
fn measure_voices(rate: f64, block: usize, feedback: bool, workers: usize) -> u32 {
    let block = block.max(1);
    let period = Duration::from_secs_f64(block as f64 / rate.max(1.0));
    let mut c = Controls {
        poly: true,
        voices: POLY_VOICES.2,
        entropy: 0.5,
        ..Controls::default()
    };
    c.panel.osc[1].on = true;
    c.panel.osc[2].on = true;
    c.panel.filter_mod = true;
    c.panel.mod_wheel = 0.4;
    c.panel.glide = 0.15;
    c.panel.glide_on = true;
    c.panel.emphasis = 0.5;
    if feedback {
        c.panel.ext_on = true;
        c.panel.ext_volume = 0.5;
        c.feedback = 0.5;
    }
    // (Paced and bounded: a fifth of a second a count.)
    let _ = ca72_rt::promote(period, period / 2);
    let (mut l, mut r) = (vec![0.0f32; block], vec![0.0f32; block]);
    let mut held: Vec<u8> = Vec::with_capacity(POLY_VOICES.2);
    let mut chord = 0usize;
    // The median time of a count's blocks where the keys change, on an engine of its own
    // (no voices of the last count still sounding).
    let mut trial = |voices: usize| -> Duration {
        c.voices = voices;
        let mut e = Engine::new();
        e.set(&c);
        e.prepare(rate, 1);
        e.start_workers(workers, Some(period));
        held.clear();
        let blocks = ((0.2 / period.as_secs_f64()) as usize).clamp(12, 64);
        let every = (blocks / 6).max(2);
        let mut changes = Vec::with_capacity(8);
        let start = Instant::now();
        for i in 0..blocks + every {
            if let Some(wait) = (start + period * i as u32).checked_duration_since(Instant::now()) {
                std::thread::sleep(wait);
            }
            let t = Instant::now();
            e.set_deadline(Some(t + period.mul_f64(DEADLINE)));
            let change = i % every == 0;
            if change {
                for key in held.drain(..) {
                    e.event(Event::Note { key, on: false });
                }
                const ROOTS: [u8; 6] = [45, 41, 43, 48, 40, 46];
                const SHAPE: [u8; 10] = [0, 7, 12, 16, 19, 24, 28, 31, 34, 36];
                let root = ROOTS[chord % ROOTS.len()];
                chord += 1;
                for d in SHAPE.iter().take(voices) {
                    e.event(Event::Note {
                        key: root + d,
                        on: true,
                    });
                    held.push(root + d);
                }
            }
            e.render(&[], &mut l, &mut r);
            e.end_block(block);
            // (The first chord's blocks warm the voices and the caches up.)
            if change && i >= every {
                changes.push(t.elapsed());
            }
        }
        std::hint::black_box((&l, &r));
        changes.sort();
        changes
            .get(changes.len() / 2)
            .copied()
            .unwrap_or(Duration::MAX)
    };
    let room = period.mul_f64(HEADROOM);
    let most = POLY_VOICES.2;
    let t = trial(most);
    if t <= room {
        return most as u32;
    }
    let mut voices =
        ((most as f64 * room.as_secs_f64() / t.as_secs_f64()).floor() as usize).min(most - 1);
    while voices > 0 {
        if trial(voices) <= room {
            break;
        }
        voices -= 1;
    }
    voices as u32
}

/// UNISON's trim on the output: its VOICES voices, not in phase, add in power, so it is turned
/// down by the square root of VOICES's setting and sounds about as loud as one voice (the
/// CA-74's R25).
fn unison_trim_of(c: &Controls) -> f64 {
    if c.unison {
        unison_trim(c.voices.clamp(POLY_VOICES.0, POLY_VOICES.2))
    } else {
        1.0
    }
}

/// Where voice `k` of seed `s` starts its oscillators on their ramps (decisions.md R45,
/// the CA-74's R25): the first voice at the circuit's initial condition, as the reference
/// starts (so one voice, and every preset with POLY and UNISON off, is the model's to the bit);
/// every other voice where its seed says, as a real instrument's free-running oscillators are
/// wherever they are when a note comes. A voice's three start together, as the reference's do,
/// only the voices apart (the CA-74 starts each oscillator apart: here LOCK's trim, R9, is for
/// three started together, and each voice's would otherwise have its own level at LOCK). Only a
/// voice made or put back to rest starts so: one that has played stops wherever its oscillators
/// were, and the next note finds them there.
fn out_of_step(k: usize, s: u64) -> Option<f64> {
    if k == 0 {
        return None;
    }
    let x = voice_seed(s, k)
        .wrapping_add(3)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    Some(((x ^ (x >> 29)) >> 11) as f64 / (1u64 << 53) as f64)
}

/// Voice `k` of an instance seeded `seed`, at `rate` Hz: its own copy from the prototype (its
/// key lists' room reserved: a key press must not allocate on the audio thread), [`seeded`].
fn made_voice(rate: f64, seed: u64, k: usize) -> Voice {
    let mut voice = Voice::prototype(rate);
    seeded(&mut voice, seed, k);
    voice
}

/// Voice `k`'s light model (QUALITY's LO) of an instance seeded `seed`, at `rate` Hz,
/// [`light_seeded`]. (Made with the voice, off the audio thread; it allocates nothing after.)
fn made_light(rate: f64, seed: u64, k: usize) -> Light {
    let mut light = Light::new(rate);
    light_seeded(&mut light, seed, k);
    light
}

/// A light voice at rest made voice `k` of an instance seeded `seed`, as [`seeded`] makes the
/// circuit's: its noise seeded, its oscillators started where [`out_of_step`] says.
fn light_seeded(light: &mut Light, seed: u64, k: usize) {
    light.set_seed(voice_seed(seed, k));
    if let Some(at) = out_of_step(k, seed) {
        light.start_oscillators_at(at);
    }
}

/// A clean voice made voice `k` of an instance seeded `seed`: its noise seeded, its
/// oscillators started where [`out_of_step`] says.
fn seeded(voice: &mut Voice, seed: u64, k: usize) {
    voice.set_seed(voice_seed(seed, k));
    if let Some(at) = out_of_step(k, seed) {
        voice.start_oscillators_at(at);
    }
}

/// LEVEL's gain at `db` (within [`LEVEL_RANGE`]; not finite, 0 dB). At 0 dB, 1 exactly.
pub fn level_gain(db: f64) -> f64 {
    let db = if db.is_finite() {
        db.clamp(LEVEL_RANGE.0, LEVEL_RANGE.1)
    } else {
        0.0
    };
    10f64.powf(db / 20.0)
}

/// How ENTROPY, SPREAD, DOUBLE and DRIVE stand for the voices' samples under controls `c`,
/// `voices` (VOICES) of them placed.
fn mix_of(c: &Controls, voices: usize) -> Mix {
    let many = c.poly || c.unison;
    // (The one instrument without DOUBLE scatters: its places are VOICES's, one an attack.)
    let scatter = !many && c.double <= 0.0;
    let entropy = c.entropy * ENTROPY_DEPTH;
    // The oscillators': their own floor and ENTROPY's, unless LOCK; the cutoff's, ENTROPY's.
    let oscillators = if c.lock { 0.0 } else { ENTROPY_FLOOR + entropy };
    Mix {
        entropy,
        oscillators,
        at: oscillators.max(entropy),
        spread: c.spread,
        inner: c.inner.clamp(0.0, 1.0),
        // (The one instrument with DOUBLE has no place among others: none, and its pair as far
        // out as SPREAD, the kit's `ALONE`.)
        placement: (many || scatter).then_some(c.placement),
        voices,
        double: if c.double > 0.0 {
            c.double.min(1.0) * DOUBLE_CENTS
        } else {
            0.0
        },
        drive: if c.drive > 0.0 {
            10f64.powf(c.drive.min(DRIVE_TOP) / 20.0)
        } else {
            1.0
        },
        scatter,
    }
}

/// AUTO GAIN's correction under controls `c` from the sound's `curve` (decisions.md
/// R45): 1 without DRIVE or with AUTO GAIN off.
fn auto_trim(c: &Controls, curve: &Curve<STEPS>) -> f64 {
    if c.auto_gain && c.drive > 0.0 {
        10f64.powf(curve.at(c.drive, DRIVE_TOP) / 20.0)
    } else {
        1.0
    }
}

/// The seed of AUTO GAIN's measuring voices: their own, not the instance's, so that a sound
/// measures the same in any instance (the CA-74's R29).
const PROBE_SEED: u64 = 0x0CA7_2D21_7E00_0001;

/// The voices a sound's panel plays (ENTROPY's tolerances on each, FEEDBACK), one for each of
/// the notes AUTO GAIN's measurement plays ([`KEYS`]), for measuring DRIVE's loudness off the
/// audio thread (decisions.md R45; `drive.rs`): made as an engine's first voices are, each
/// its own (its parts, its oscillators' start, its noise), POLY's, one a note, in the centre,
/// without the output's gain (MAIN OUTPUT's, LEVEL's, the trims', AUTO GAIN's).
#[derive(Debug)]
pub struct Probe {
    voices: Vec<Playing>,
    controls: Controls,
    rate: f64,
}

impl Probe {
    /// For the sound of controls `c` with the voices at `rate` Hz.
    pub fn new(c: &Controls, rate: f64) -> Probe {
        let mut controls = *c;
        (
            controls.poly,
            controls.unison,
            controls.double,
            controls.spread,
        ) = (true, false, 0.0, 0.0);
        let entropy = controls.entropy * ENTROPY_DEPTH;
        let mut panel = c.panel;
        panel.quality = Quality::Potato;
        let voices = (0..KEYS.len())
            .map(|k| {
                let character = Character::new(voice_seed(PROBE_SEED, k), k, 3, ENTROPY_KNOBS);
                let mut voice = made_voice(rate, PROBE_SEED, k);
                voice.feedback = controls.feedback.clamp(0.0, 1.0);
                voice.panel = if entropy > 0.0 {
                    entropy_panel(&character, panel, entropy)
                } else {
                    panel
                };
                // (At QUALITY's LO the light voices measured, as they play.)
                let mut light = made_light(rate, PROBE_SEED, k);
                if controls.potato {
                    light.feedback = voice.feedback;
                    light.set_panel(&voice.panel);
                }
                Playing {
                    voice,
                    light,
                    potato: controls.potato,
                    character,
                    key: PolyKey::new(rate),
                    gate: false,
                    note: 0,
                    drop: false,
                    out: [[0.0; CHUNK]; 2],
                    peak: 0.0,
                    pan: None,
                    panel_gen: 0,
                    broken: false,
                    twin: made_twin(rate, PROBE_SEED, k),
                    double: false,
                    turn: None,
                    closed: false,
                }
            })
            .collect();
        Probe {
            voices,
            controls,
            rate,
        }
    }

    /// The K-weighted energy of [`KEYS`] played together, held [`HELD`] s and heard for
    /// [`HEARD`] s, at DRIVE `db` (dB): each time from the voices as they were made.
    pub fn energy(&self, db: f64) -> f64 {
        let mut c = self.controls;
        c.drive = db;
        let m = mix_of(&c, KEYS.len());
        let mut voices = self.voices.clone();
        for (v, key) in voices.iter_mut().zip(KEYS) {
            v.note = key;
        }
        let mut k = KWeighted::new(self.rate);
        let held = (HELD * self.rate) as usize;
        let heard = (HEARD * self.rate) as usize;
        let ext = [0.0; CHUNK];
        let share = glide_share(self.rate);
        let mut at = 0;
        while at < heard {
            let until = if at < held { held } else { heard };
            let n = CHUNK.min(until - at);
            for v in &mut voices {
                v.gate = at < held;
                v.play_from(&ext[..n], self.rate, &m, share);
            }
            for i in 0..n {
                k.push(voices.iter().map(|v| v.out[0][i]).sum());
            }
            at += n;
        }
        k.energy
    }
}

/// DOUBLE's trim on the output: a note's two voices, each its own, add in power, so it is
/// turned down by two's square root and sounds about as loud as one voice a note (the CA-74's
/// R28).
fn double_trim(c: &Controls) -> f64 {
    if c.double > 0.0 { DOUBLE_TRIM } else { 1.0 }
}

/// Voice `k`'s noise seed from the instance's: the first voice's the instance's own.
fn voice_seed(seed: u64, k: usize) -> u64 {
    if k == 0 {
        seed
    } else {
        seed ^ ((k as u64 + 1) << 48)
    }
}

/// The panel with voice `c`'s tolerances, by `amount` (0..1) of ENTROPY.
fn entropy_panel(c: &Character, mut p: Panel, amount: f64) -> Panel {
    let k = |i: usize, x: f64| c.knob(i, x, amount);
    p.emphasis = k(0, p.emphasis);
    p.filter_contour.attack = k(1, p.filter_contour.attack);
    p.filter_contour.decay = k(2, p.filter_contour.decay);
    p.loudness_contour.attack = k(3, p.loudness_contour.attack);
    p.loudness_contour.decay = k(4, p.loudness_contour.decay);
    p.glide = k(5, p.glide);
    p
}

/// MAIN OUTPUT VOLUME: R20 (5K, audio taper) from the output into the jack's 10K load, and
/// its switch; `volume` 0..1 of the knob's travel.
pub fn output_gain(volume: f64, on: bool) -> f64 {
    if !on {
        return 0.0;
    }
    let t = audio_taper(volume);
    let (top, bottom) = (5e3 * (1.0 - t), 5e3 * t);
    let load = bottom * 10e3 / (bottom + 10e3);
    if load <= 0.0 {
        0.0
    } else {
        load / (top + load)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::Budget;
    use ca72::voice::Priority;

    /// A voice's output's peak as the engine reports it, playing the panel as it opens and
    /// louder: printed, for the editor's level scale (`-- --ignored --nocapture`).
    #[test]
    #[ignore = "prints levels"]
    fn voice_levels() {
        for (name, c) in [
            ("as it opens", Controls::default()),
            (
                "POLY, four voices",
                Controls {
                    poly: true,
                    voices: 4,
                    ..Controls::default()
                },
            ),
        ] {
            let mut e = Engine::new();
            e.set(&c);
            e.prepare(48_000.0, 1);
            for key in [45, 52, 57, 64] {
                e.event(Event::Note { key, on: true });
            }
            let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
            let mut most = [0.0f32; LEVELS];
            let mut out = 0.0f32;
            for _ in 0..200 {
                e.render(&[], &mut l, &mut r);
                e.end_block(256);
                for (m, v) in most.iter_mut().zip(e.levels()) {
                    *m = m.max(*v);
                }
                out = l.iter().chain(r.iter()).fold(out, |m, v| m.max(v.abs()));
            }
            println!("{name}: voices' peaks {most:?}, the output's {out}");
        }
    }

    fn four_voices(workers: usize) -> Engine {
        let mut e = Engine::new();
        e.set(&Controls {
            poly: true,
            voices: 4,
            ..Controls::default()
        });
        e.prepare(48_000.0, 1);
        assert_eq!(e.start_workers(workers, None), workers);
        for key in [45, 52, 57, 64] {
            e.event(Event::Note { key, on: true });
        }
        let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
        e.render(&[], &mut l, &mut r);
        e.end_block(256);
        e
    }

    /// The workers by the machine's processors (decisions.md R11, R39): none on 1 or 2, one
    /// from 3 (a quad-core's 4 among them: before R39, none below 5), then a third of them
    /// less two, at most four.
    #[test]
    fn a_machine_of_three_processors_or_more_has_a_worker() {
        let by: Vec<usize> = (1..=20).map(workers_for).collect();
        assert_eq!(
            by,
            [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]
        );
        assert_eq!(workers_for(0), 0);
        assert_eq!(workers_for(256), 4);
    }

    /// A voice that a worker late from an earlier run still holds sits the next runs out:
    /// the block is not held up for it (it is still held when the block is done), the other
    /// voices play, and it plays again once let go. (Bounded by what happens, not by time: on
    /// a loaded machine any thread of the test may be held up for milliseconds.)
    #[test]
    fn a_voice_still_held_sits_out_and_nothing_waits_for_it() {
        let mut e = four_voices(2);
        let held = e.slots[1].play.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let (done, until) = std::sync::mpsc::channel::<()>();
        let late = std::thread::spawn(move || {
            let g = held.lock();
            tx.send(()).unwrap();
            let _ = until.recv_timeout(Duration::from_secs(30));
            drop(g);
        });
        rx.recv().unwrap();
        let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
        // (No deadline: the voices the workers take are waited for, however late they are.)
        e.set_deadline(None);
        e.render(&[], &mut l, &mut r);
        assert!(!late.is_finished(), "the block waited for the held voice");
        assert!(l.iter().any(|x| x.abs() > 0.0), "the other voices silent");
        done.send(()).unwrap();
        late.join().unwrap();
        e.render(&[], &mut l, &mut r);
        assert!(e.slots[1].play.try_lock().is_ok());
    }

    /// A worker that has taken a voice and does not finish it by the deadline: the audio
    /// thread waits no longer than the deadline (it returns while the voice is still held, far
    /// past it), and that voice is silent for the run. The deadline is generous so that a
    /// loaded machine holding a worker up does not silence the others too.
    #[test]
    fn a_worker_late_past_the_deadline_is_not_waited_for() {
        let e = four_voices(1);
        let voices: Vec<Shared> = e.slots.iter().map(|s| s.play.clone()).collect();
        let mut pool = Pool::start(1, &voices, None).unwrap();
        let held = voices[0].clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let (done, until) = std::sync::mpsc::channel::<()>();
        let stall = std::thread::spawn(move || {
            let g = held.lock();
            tx.send(()).unwrap();
            let _ = until.recv_timeout(Duration::from_secs(30));
            drop(g);
        });
        rx.recv().unwrap();
        let mix = e.mix();
        let played = pool.play(
            &[0, 1, 2, 3],
            64,
            &[0.0; CHUNK],
            48_000.0,
            &mix,
            Some(Instant::now() + Duration::from_secs(1)),
        );
        assert!(!stall.is_finished(), "waited for the held voice");
        assert!(!played[0], "the held voice played");
        assert!(played[1..4].iter().all(|&p| p), "{played:?}");
        done.send(()).unwrap();
        stall.join().unwrap();
    }

    /// A voice let go that sits a run out (a worker late from an earlier run holds it) is not
    /// freed for it at the block's end: its tail is not known to have ended (R18).
    #[test]
    fn a_voice_that_misses_a_run_is_not_freed_for_it() {
        let mut e = four_voices(0);
        let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
        for _ in 0..40 {
            e.render(&[], &mut l, &mut r);
            e.end_block(256);
        }
        for key in [45, 52, 57, 64] {
            e.event(Event::Note { key, on: false });
        }
        let held = e.slots[1].play.clone();
        let g = held.lock().unwrap();
        e.render(&[], &mut l, &mut r);
        e.end_block(256);
        drop(g);
        assert!(e.slots.iter().take(4).all(|s| s.active && !s.gate));
    }

    /// A retrigger's drop handed to a voice for a run it did not play is kept for its next run,
    /// not overwritten there, and played once; the gate it shows, not known to have played the
    /// run, keeps a new note on it retriggering (R18).
    #[test]
    fn a_drop_for_a_run_not_played_is_kept_for_the_next() {
        let mut e = four_voices(0);
        let play = e.slots[0].play.clone();
        let mut p = play.lock().unwrap();
        let s = &mut e.slots[0];
        (s.drop, s.shown) = (true, true);
        s.hand(&mut p);
        s.ran(false);
        assert!(!s.drop && p.drop);
        s.hand(&mut p);
        assert!(p.drop, "the drop lost");
        p.play(1, &[0.0; CHUNK], 48_000.0, &Mix::from_array([0.0; MIX_LEN]));
        assert!(!p.drop);
        s.gate = false;
        s.ran(false);
        assert!(s.shown, "a gate the voice may not have seen");
        s.ran(true);
        assert!(!s.shown);
    }

    /// A voice whose output goes non-finite (its preamplifier's states poisoned, as one such
    /// sample at EXTERNAL INPUT left them) is put back to rest: the output stays finite and
    /// the held key sounds again; and again once the spares are mended (R18). POLY off and on.
    #[test]
    fn a_voice_gone_non_finite_is_put_back_to_rest() {
        for poly in [false, true] {
            let mut c = Controls {
                poly,
                ..Controls::default()
            };
            c.panel.ext_on = true;
            c.panel.ext_volume = 1.0;
            let mut e = Engine::new();
            e.set(&c);
            e.prepare(48_000.0, 1);
            e.event(Event::Note { key: 57, on: true });
            let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
            for round in 0..3 {
                if let Ok(mut p) = e.slots[0].play.lock() {
                    p.voice.tick_jacks(&ca72::voice::Jacks {
                        ext: f64::NAN,
                        ..ca72::voice::Jacks::default()
                    });
                }
                let mut last = 0.0f32;
                for _ in 0..40 {
                    e.render(&[], &mut l, &mut r);
                    e.end_block(256);
                    assert!(l.iter().chain(&r).all(|x| x.is_finite()), "poly {poly}");
                    last = l.iter().fold(0.0, |m, x| m.max(x.abs()));
                }
                assert!(e.take_mending(), "poly {poly}, round {round}: not restored");
                assert!(last > 1e-2, "poly {poly}, round {round}: silent ({last})");
                e.spares().mend();
            }
        }
    }

    /// Workers running for the voices, as many and for the same period, are kept when the
    /// engine is prepared again at the same rate (a host's state loaded into a running
    /// instance), and started once for voices built at a new rate (R18); a state with POLY off
    /// stops them (R21).
    #[test]
    fn workers_running_for_the_voices_are_kept() {
        let mut e = Engine::new();
        let mut c = Controls {
            poly: true,
            ..Controls::default()
        };
        e.set(&c);
        e.prepare(48_000.0, 1);
        assert_eq!(e.start_workers(2, None), 2);
        e.prepare(48_000.0, 2);
        assert_eq!(e.start_workers(2, None), 2);
        assert_eq!(e.pool_starts, 1, "started again at the same rate");
        e.prepare(44_100.0, 2);
        assert_eq!(e.workers(), 0, "kept with the old voices");
        assert_eq!(e.start_workers(2, None), 2);
        assert_eq!(e.start_workers(1, None), 1);
        assert_eq!(e.pool_starts, 3);
        c.poly = false;
        e.set(&c);
        e.prepare(44_100.0, 2);
        assert_eq!(e.start_workers(1, None), 0, "kept with POLY off");
    }

    /// A mend stops before its next voice once told to (the plug-in dropped: decisions.md
    /// R23): told after one, it makes one clean voice; the next mend makes the rest.
    #[test]
    fn a_mend_stops_before_its_next_voice() {
        let spares = Spares::default();
        spares.make(48_000.0);
        for place in &spares.places[..3] {
            place.lock().unwrap().clean = false;
        }
        let asked = std::cell::Cell::new(0);
        spares.mend_until(|| {
            asked.set(asked.get() + 1);
            asked.get() > 1
        });
        let clean = || {
            spares
                .places
                .iter()
                .filter(|p| p.lock().unwrap().clean)
                .count()
        };
        assert_eq!(clean(), POLY_VOICES.2 - 2);
        spares.mend();
        assert_eq!(clean(), POLY_VOICES.2);
    }

    /// An engine whose workers come out of `budget`, its voices built, POLY as `poly`, and
    /// workers asked of it as the plug-in asks them (for 5 ms blocks).
    fn planned(budget: &'static Budget, poly: bool, workers: usize) -> Engine {
        let mut e = Engine::new();
        e.crew = Arc::new(Crew::with_budget(budget));
        e.set(&Controls {
            poly,
            voices: 4,
            ..Controls::default()
        });
        e.prepare(48_000.0, 1);
        e.start_workers(workers, Some(Duration::from_millis(5)));
        e
    }

    /// POLY switched on or off, and the next block's start; returns whether serving is due
    /// (which the plug-in asks of its helper: decisions.md R23).
    fn switch(e: &mut Engine, poly: bool) -> bool {
        let c = Controls { poly, ..e.controls };
        e.set(&c);
        e.set_deadline(None);
        e.take_serving()
    }

    /// The helper thread's serving, run here (decisions.md R21, R23).
    fn serve(e: &Engine) {
        e.crew().serve();
    }

    /// An instance holds workers only while POLY is on (R21): activated with POLY off, none;
    /// switched on, none at once (its voices play on the caller's thread) and its workers once
    /// the helper thread has started them; switched off, let go at the next block and
    /// back in the budget once the helper thread has stopped them.
    #[test]
    fn an_instance_holds_workers_only_while_poly_is_on() {
        static THREE: Budget = Budget::new(|| 3);
        let mut e = planned(&THREE, false, 2);
        assert_eq!((e.workers(), THREE.held()), (0, 0));
        assert!(!switch(&mut e, false));
        assert!(switch(&mut e, true), "no workers asked for");
        assert_eq!(e.workers(), 0);
        for key in [45, 52, 57, 64] {
            e.event(Event::Note { key, on: true });
        }
        let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
        e.render(&[], &mut l, &mut r);
        assert!(l.iter().any(|x| x.abs() > 0.0), "silent while waiting");
        serve(&e);
        assert_eq!(THREE.held(), 2);
        e.set_deadline(None);
        assert_eq!(e.workers(), 2);
        e.render(&[], &mut l, &mut r);
        e.end_block(256);
        assert!(switch(&mut e, false), "not let go");
        assert_eq!(e.workers(), 0);
        assert_eq!(THREE.held(), 2, "stopped on the audio thread");
        serve(&e);
        assert_eq!(THREE.held(), 0);
    }

    /// The bug R21 fixes: an instance with POLY off, activated first, held its share of the
    /// budget, and one with POLY on, activated later, got none. Now the second takes them all;
    /// the first, switched on, finds none left and plays its voices itself, and asks again
    /// when POLY is next switched on, by which time the second has given them back.
    #[test]
    fn an_instance_with_poly_off_leaves_its_workers_to_one_with_it_on() {
        static FOUR: Budget = Budget::new(|| 4);
        let mut off = planned(&FOUR, false, 4);
        let mut on = planned(&FOUR, true, 4);
        assert_eq!((off.workers(), on.workers()), (0, 4));
        assert!(switch(&mut off, true));
        serve(&off);
        off.set_deadline(None);
        assert_eq!(off.workers(), 0);
        assert!(!switch(&mut off, true), "asked again while POLY stays on");
        assert!(switch(&mut on, false));
        serve(&on);
        assert_eq!(FOUR.held(), 0);
        switch(&mut off, false);
        assert!(switch(&mut off, true));
        serve(&off);
        off.set_deadline(None);
        assert_eq!(off.workers(), 4);
        drop((off, on));
        assert_eq!(FOUR.held(), 0);
    }

    /// Voices built at a new rate while a pool started for the old ones waits to be taken, or
    /// the plug-in deactivated then: that pool is stopped, and the new voices' own pool gets
    /// every worker; nothing is kept from the budget.
    #[test]
    fn a_pool_for_voices_gone_is_stopped() {
        static TWO: Budget = Budget::new(|| 2);
        let mut e = planned(&TWO, false, 2);
        assert!(switch(&mut e, true));
        serve(&e);
        assert_eq!(TWO.held(), 2);
        e.prepare(44_100.0, 1);
        assert_eq!(TWO.held(), 0);
        assert_eq!(e.start_workers(2, Some(Duration::from_millis(5))), 2);
        e.set_deadline(None);
        assert_eq!(e.workers(), 2);
        assert!(switch(&mut e, false));
        assert!(switch(&mut e, true));
        serve(&e);
        e.stop_workers();
        assert_eq!(TWO.held(), 0);
        e.set_deadline(None);
        assert_eq!(e.workers(), 0);
        assert!(!e.take_serving());
    }

    /// A new seed at the rate the voices were built for draws their characters again, as
    /// voices built with it have them (R18).
    #[test]
    fn a_new_seed_draws_the_characters_again() {
        let mut a = Engine::new();
        a.prepare(48_000.0, 1);
        a.prepare(48_000.0, 2);
        let mut b = Engine::new();
        b.prepare(48_000.0, 2);
        for (k, (x, y)) in a.slots.iter().zip(&b.slots).enumerate() {
            let (x, y) = (x.play.lock().unwrap(), y.play.lock().unwrap());
            assert_eq!(
                format!("{:?}", x.character),
                format!("{:?}", y.character),
                "voice {k}"
            );
        }
    }

    /// Left and right power of `keys` held for a second with `c`, and the largest difference
    /// of a sample's two sides.
    fn sides(c: &Controls, keys: &[u8]) -> (f64, f64, f64) {
        let mut e = Engine::new();
        e.set(c);
        e.prepare(48_000.0, 9);
        for &key in keys {
            e.event(Event::Note { key, on: true });
        }
        let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
        let (mut mid, mut side, mut diff) = (0.0f64, 0.0f64, 0.0f64);
        for _ in 0..(48_000 / 256) {
            e.render(&[], &mut l, &mut r);
            e.end_block(256);
            for (&a, &b) in l.iter().zip(&r) {
                let (a, b) = (f64::from(a), f64::from(b));
                mid += (0.5 * (a + b)).powi(2);
                side += (0.5 * (a - b)).powi(2);
                diff = diff.max((a - b).abs());
            }
        }
        (mid, side, diff)
    }

    /// At full SPREAD a three-note chord fills the field (side within 3 dB of mid; the old
    /// places and pan law, a voice in the centre weighed double, gave less), and one voice
    /// alone without SPREAD stays in the centre, left and right the same (decisions.md R45).
    #[test]
    fn a_chord_fills_the_field_and_one_voice_without_spread_stays_in_the_centre() {
        let c = Controls {
            poly: true,
            voices: 8,
            spread: 1.0,
            ..Controls::default()
        };
        let (mid, side, _) = sides(&c, &[57, 61, 64]);
        let db = 10.0 * (side / mid).log10();
        assert!(db > -3.0, "side {db:.1} dB against mid");
        let mono = Controls {
            poly: false,
            spread: 0.0,
            ..c
        };
        let (_, _, diff) = sides(&mono, &[57]);
        assert_eq!(diff, 0.0, "one voice off the centre");
    }

    /// INNER at 100 % clears the centre: a chord on EVEN's three voices, one of them in the
    /// centre at INNER 0, is all at the edges, its side nearer its mid; one voice alone without
    /// SPREAD stays in the centre, whatever INNER (decisions.md R49).
    #[test]
    fn inner_clears_the_centre_and_one_voice_stays_there() {
        let c = Controls {
            poly: true,
            voices: 3,
            spread: 1.0,
            ..Controls::default()
        };
        let share = |c: &Controls| {
            let (mid, side, _) = sides(c, &[57, 61, 64]);
            side / mid
        };
        let (open, cleared) = (share(&c), share(&Controls { inner: 1.0, ..c }));
        assert!(
            cleared > 1.3 * open,
            "side against mid: {open:.3} then {cleared:.3}"
        );
        let mono = Controls {
            poly: false,
            spread: 0.0,
            inner: 1.0,
            ..c
        };
        let (_, _, diff) = sides(&mono, &[57]);
        assert_eq!(diff, 0.0, "one voice off the centre");
    }

    /// Where the one instrument sounds for each of `keys` played in turn with `c`, each held
    /// 0.15 s and let go for 0.25 s: the right side's share of its power while held, from 20 ms
    /// in (its place reached); and the editor's bits for it then.
    fn scattered(c: &Controls, keys: &[u8]) -> Vec<(f64, u32)> {
        let mut e = Engine::new();
        e.set(c);
        e.prepare(48_000.0, 9);
        let mut out = Vec::new();
        for &key in keys {
            e.event(Event::Note { key, on: true });
            let (l, r) = run_for(&mut e, 0.15);
            let from = (0.02 * 48_000.0) as usize;
            let power = |x: &[f64]| x[from..].iter().map(|v| v * v).sum::<f64>();
            let (pl, pr) = (power(&l), power(&r));
            assert!(pl + pr > 1e-3, "key {key} silent");
            out.push((pr / (pl + pr), e.sounding_mask()));
            e.event(Event::Note { key, on: false });
            run_for(&mut e, 0.25);
        }
        out
    }

    /// `e` rendered for `seconds`, left and right.
    fn run_for(e: &mut Engine, seconds: f64) -> (Vec<f64>, Vec<f64>) {
        let (mut l, mut r) = ([0.0f32; 240], [0.0f32; 240]);
        let (mut ls, mut rs) = (Vec::new(), Vec::new());
        for _ in 0..(seconds * 48_000.0 / 240.0).round() as usize {
            e.render(&[], &mut l, &mut r);
            e.end_block(240);
            ls.extend(l.iter().map(|&x| f64::from(x)));
            rs.extend(r.iter().map(|&x| f64::from(x)));
        }
        (ls, rs)
    }

    /// The right side's share of a voice's power at place `k` of `n` (the kit's gains).
    fn share_at(c: &Controls, k: usize, n: usize) -> f64 {
        let (l, r) = voice_gains(c.spread, c.inner, c.placement, k, n);
        let (l, r) = (f64::from(l), f64::from(r));
        r * r / (l * l + r * r)
    }

    /// The one instrument (POLY, UNISON and DOUBLE off) scatters: each attack at the next of
    /// VOICES's places, round and round, where the placement puts POLY's voices (EVEN; CENTER
    /// with INNER), at QUALITY's LO the same; the editor told the place (decisions.md
    /// R62).
    #[test]
    fn the_one_instrument_takes_the_next_place_at_each_attack() {
        let keys = [57, 60, 62, 64, 65, 67, 69];
        let even = Controls {
            voices: 4,
            spread: 1.0,
            ..Controls::default()
        };
        let centre = Controls {
            voices: 5,
            spread: 0.8,
            inner: 0.3,
            placement: Placement::Centre,
            ..Controls::default()
        };
        let light = Controls {
            potato: true,
            ..even
        };
        for c in [even, centre, light] {
            let got = scattered(&c, &keys);
            for (i, &(share, mask)) in got.iter().enumerate() {
                let k = i % c.voices;
                let want = share_at(&c, k, c.voices);
                assert!(
                    (share - want).abs() < 1e-6,
                    "{:?} (LO {}) note {i}: {share:.6} against place {k}'s {want:.6}",
                    c.placement,
                    c.potato
                );
                assert_eq!(mask, 1 << k, "note {i}: the editor's bit");
            }
            // (Not all in one place.)
            assert!(got.iter().any(|g| (g.0 - got[0].0).abs() > 0.1));
        }
    }

    /// A note played legato (its trigger contact staying closed, single triggering) stays at
    /// the place its phrase began; with RETRIGGER, a key pressed among others held starts the
    /// contours again, and moves to the next place (decisions.md R62).
    #[test]
    fn a_legato_note_stays_and_retrigger_moves_it() {
        for retrigger in [false, true] {
            let mut c = Controls {
                voices: 4,
                spread: 1.0,
                ..Controls::default()
            };
            c.panel.retrigger = retrigger;
            c.panel.priority = Priority::Last;
            let mut e = Engine::new();
            e.set(&c);
            e.prepare(48_000.0, 9);
            e.event(Event::Note { key: 57, on: true });
            run_for(&mut e, 0.15);
            e.event(Event::Note { key: 64, on: true });
            let (l, r) = run_for(&mut e, 0.15);
            let from = (0.04 * 48_000.0) as usize;
            let power = |x: &[f64]| x[from..].iter().map(|v| v * v).sum::<f64>();
            let share = power(&r) / (power(&l) + power(&r));
            let k = usize::from(retrigger);
            assert!(
                (share - share_at(&c, k, 4)).abs() < 1e-6,
                "retrigger {retrigger}: {share:.6} against place {k}'s"
            );
        }
    }

    /// All Notes Off (and the host's reset, which lets every key go so) starts the one
    /// instrument's pattern again from the first place, so that a part plays and renders its
    /// places alike each time (decisions.md R62).
    #[test]
    fn all_notes_off_starts_the_pattern_again() {
        let c = Controls {
            voices: 4,
            spread: 1.0,
            ..Controls::default()
        };
        let mut e = Engine::new();
        e.set(&c);
        e.prepare(48_000.0, 9);
        for key in [57, 60] {
            e.event(Event::Note { key, on: true });
            run_for(&mut e, 0.1);
            e.event(Event::Note { key, on: false });
            run_for(&mut e, 0.1);
        }
        assert_eq!(e.turn, 1);
        e.event(Event::AllNotesOff);
        e.event(Event::Note { key: 62, on: true });
        let (l, r) = run_for(&mut e, 0.1);
        let from = (0.02 * 48_000.0) as usize;
        let power = |x: &[f64]| x[from..].iter().map(|v| v * v).sum::<f64>();
        let share = power(&r) / (power(&l) + power(&r));
        assert!((share - share_at(&c, 0, 4)).abs() < 1e-6, "{share:.6}");
        assert_eq!(e.turn, 0);
    }

    /// At an attack the one instrument reaches its new place before its contours start
    /// ([`TURN_GLIDE`], `TRIGGER_ON` after its trigger contact closes): within 1 % of the way
    /// there, from the edge to the other (with [`GLIDE`], 46 % short), so that the note is
    /// heard where it sounds, not sliding there (decisions.md R62).
    #[test]
    fn the_one_instrument_is_at_its_new_place_before_its_contours_start() {
        let c = Controls {
            voices: 2,
            spread: 1.0,
            ..Controls::default()
        };
        let mut e = Engine::new();
        e.set(&c);
        e.prepare(48_000.0, 9);
        e.event(Event::Note { key: 57, on: true });
        run_for(&mut e, 0.1);
        e.event(Event::Note { key: 57, on: false });
        run_for(&mut e, 0.2);
        e.event(Event::Note { key: 60, on: true });
        for _ in 0..(ca72::light::laws::TRIGGER_ON * 48_000.0).round() as usize {
            e.tick_stereo(0.0);
        }
        let (gl, _) = e.slots[0].play.lock().unwrap().pan.unwrap();
        let ((from, _), (to, _)) = (
            voice_gains(1.0, 0.0, Placement::Even, 0, 2),
            voice_gains(1.0, 0.0, Placement::Even, 1, 2),
        );
        let short = (gl - f64::from(to)) / f64::from(from - to);
        assert!(
            short.abs() < 0.01,
            "{:.1} % of the way short",
            100.0 * short
        );
    }

    /// POLY's notes go round the voices: a note after one let go and fallen silent takes the
    /// next voice, not the first again.
    #[test]
    fn poly_notes_go_round_the_voices() {
        let c = Controls {
            poly: true,
            voices: 4,
            ..Controls::default()
        };
        let mut e = Engine::new();
        e.set(&c);
        e.prepare(48_000.0, 9);
        let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
        for (i, key) in [48u8, 50, 52, 53, 55].into_iter().enumerate() {
            e.event(Event::Note { key, on: true });
            let k = e
                .slots
                .iter()
                .position(|s| s.gate)
                .expect("a voice for the note");
            assert_eq!(k, i % 4, "note {i}");
            e.event(Event::Note { key, on: false });
            for _ in 0..(3 * 48_000 / 256) {
                e.render(&[], &mut l, &mut r);
                e.end_block(256);
            }
            assert!(e.slots.iter().all(|s| !s.active), "note {i} not silent");
        }
    }

    /// A key played again takes the voice it last had, letting go or fallen silent, so a chord
    /// struck again stays where it was in the field; a new key still takes the next voice.
    #[test]
    fn a_key_played_again_keeps_its_voice() {
        let c = Controls {
            poly: true,
            voices: 8,
            ..Controls::default()
        };
        let mut e = Engine::new();
        e.set(&c);
        e.prepare(48_000.0, 9);
        let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
        let chord = [60u8, 64, 67];
        let voices = |e: &Engine| -> Vec<usize> {
            chord
                .iter()
                .map(|&key| {
                    e.slots
                        .iter()
                        .position(|s| s.gate && s.note == key)
                        .expect("a voice for the key")
                })
                .collect()
        };
        let strike = |e: &mut Engine, on: bool| {
            for key in chord {
                e.event(Event::Note { key, on });
            }
        };
        strike(&mut e, true);
        let first = voices(&e);
        strike(&mut e, false);
        // Struck again while letting go, then again after falling silent.
        for wait in [2usize, 3 * 48_000 / 256] {
            for _ in 0..wait {
                e.render(&[], &mut l, &mut r);
                e.end_block(256);
            }
            strike(&mut e, true);
            assert_eq!(voices(&e), first, "after {wait} blocks");
            strike(&mut e, false);
        }
        e.event(Event::Note { key: 69, on: true });
        let new = e.slots.iter().position(|s| s.gate).unwrap();
        assert!(
            !first.contains(&new),
            "a new key on voice {new}, one of the chord's"
        );
    }

    /// A held note's power over `seconds` (after `skip` of them) with `c`, on key `key`.
    fn held_power(c: &Controls, key: u8, skip: f64, seconds: f64) -> f64 {
        let mut e = Engine::new();
        e.set(c);
        e.prepare(48_000.0, 9);
        e.event(Event::Note { key, on: true });
        let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
        let mut sum = 0.0f64;
        for k in 0..((skip + seconds) * 48_000.0 / 256.0) as usize {
            e.render(&[], &mut l, &mut r);
            e.end_block(256);
            if k as f64 >= skip * 48_000.0 / 256.0 {
                sum += l
                    .iter()
                    .chain(&r)
                    .map(|&x| f64::from(x).powi(2))
                    .sum::<f64>();
            }
        }
        sum
    }

    /// UNISON's voices come in anywhere in their cycles, not in step: with LOCK (every voice
    /// the circuit as drawn, nothing to pull them apart) a held note is still about as loud
    /// with eight of them as with one (within 2 dB), where eight in step would sum 9 dB louder
    /// (decisions.md R45, the CA-74's R25).
    #[test]
    fn unison_voices_come_in_out_of_step() {
        let c = Controls {
            voices: 8,
            lock: true,
            ..Controls::default()
        };
        let one = held_power(&c, 40, 0.0, 1.0);
        let all = held_power(&Controls { unison: true, ..c }, 40, 0.0, 1.0);
        let db = 10.0 * (all / one).log10();
        assert!(db.abs() < 2.0, "UNISON {db:+.1} dB against one voice");
    }

    /// UNISON's voices, each its own, are trimmed by their count's square root: a held note
    /// is about as loud with eight of them as with one (within 2 dB, over its sustain).
    #[test]
    fn unison_is_about_as_loud_as_one_voice() {
        let c = Controls {
            voices: 8,
            entropy: 0.3,
            ..Controls::default()
        };
        let one = held_power(&c, 45, 1.0, 2.0);
        let all = held_power(&Controls { unison: true, ..c }, 45, 1.0, 2.0);
        let db = 10.0 * (all / one).log10();
        assert!(db.abs() < 2.0, "UNISON {db:+.1} dB against one voice");
    }

    /// Every UNISON voice takes the keys as the one instrument's keyboard does (NOTE PRIORITY,
    /// LOW by default, and single triggering): its keyboard voltage the one instrument's after each of a
    /// run of keys pressed and let go, legato and not; and every voice sounds.
    #[test]
    fn unison_voices_take_the_keys_as_the_instrument_does() {
        let events: [(u8, bool); 8] = [
            (60, true),
            (64, true),
            (55, true),
            (55, false),
            (60, false),
            (64, false),
            (67, true),
            (62, true),
        ];
        let run = |unison: bool| -> Vec<Vec<f64>> {
            let c = Controls {
                unison,
                voices: 4,
                ..Controls::default()
            };
            let mut e = Engine::new();
            e.set(&c);
            e.prepare(48_000.0, 9);
            let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
            let mut out = Vec::new();
            for (key, on) in events {
                e.event(Event::Note { key, on });
                for _ in 0..8 {
                    e.render(&[], &mut l, &mut r);
                    e.end_block(256);
                }
                let n = if unison { 4 } else { 1 };
                out.push(
                    (0..n)
                        .map(|k| e.slots[k].play.lock().unwrap().voice.probe().0)
                        .collect(),
                );
            }
            if unison {
                assert_eq!(e.sounding(), 4, "every UNISON voice sounding");
            }
            out
        };
        let (one, all) = (run(false), run(true));
        for (i, (o, a)) in one.iter().zip(&all).enumerate() {
            for (k, v) in a.iter().enumerate() {
                assert!(
                    (v - o[0]).abs() < 1e-9,
                    "after event {i}, voice {k}'s keyboard at {v} V, the instrument's {}",
                    o[0]
                );
            }
        }
    }

    /// NOTE PRIORITY reaches the voices (decisions.md R50): with POLY off, and in each
    /// of UNISON's voices, the keyboard after 60, 67, 55 and 64 are pressed is where the key the
    /// priority chooses puts it alone (LOW the circuit's own, its lowest a little sharp with
    /// higher keys held: within 1.5 mV).
    #[test]
    fn note_priority_reaches_the_one_instrument_and_unisons_voices() {
        let keyboard = |priority: Priority, unison: bool, keys: &[u8]| -> Vec<f64> {
            let mut c = Controls {
                unison,
                voices: 4,
                ..Controls::default()
            };
            c.panel.priority = priority;
            let mut e = Engine::new();
            e.set(&c);
            e.prepare(48_000.0, 9);
            let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
            for &key in keys {
                e.event(Event::Note { key, on: true });
                for _ in 0..8 {
                    e.render(&[], &mut l, &mut r);
                    e.end_block(256);
                }
            }
            let n = if unison { 4 } else { 1 };
            (0..n)
                .map(|k| e.slots[k].play.lock().unwrap().voice.probe().0)
                .collect()
        };
        for (priority, sounding, within) in [
            (Priority::Low, 55, 1.5e-3),
            (Priority::Last, 64, 5e-5),
            (Priority::High, 67, 5e-5),
        ] {
            let alone = keyboard(Priority::Low, false, &[sounding])[0];
            for unison in [false, true] {
                for (k, v) in keyboard(priority, unison, &[60, 67, 55, 64])
                    .into_iter()
                    .enumerate()
                {
                    assert!(
                        (v - alone).abs() < within,
                        "{priority:?}, UNISON {unison}: voice {k}'s keyboard {:.3} mV from {sounding}'s",
                        (v - alone) * 1e3
                    );
                }
            }
        }
    }

    /// Switching UNISON lets every key go: no voice's keyboard holds a key after it.
    #[test]
    fn switching_unison_lets_every_key_go() {
        let c = Controls {
            unison: true,
            voices: 4,
            ..Controls::default()
        };
        let mut e = Engine::new();
        e.set(&c);
        e.prepare(48_000.0, 9);
        let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
        e.event(Event::Note { key: 60, on: true });
        e.render(&[], &mut l, &mut r);
        e.set(&Controls { unison: false, ..c });
        e.render(&[], &mut l, &mut r);
        for k in 0..4 {
            assert!(
                !e.slots[k].play.lock().unwrap().voice.envelopes().2,
                "voice {k} still holds a key"
            );
        }
    }

    /// Left and right of `c` with `keys` held for `seconds`, as `sides` plays them.
    fn played(c: &Controls, keys: &[u8], seconds: f64) -> (Vec<f64>, Vec<f64>) {
        let mut e = Engine::new();
        e.set(c);
        e.prepare(48_000.0, 9);
        for &key in keys {
            e.event(Event::Note { key, on: true });
        }
        let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
        let (mut ls, mut rs) = (Vec::new(), Vec::new());
        for _ in 0..(seconds * 48_000.0 / 256.0) as usize {
            e.render(&[], &mut l, &mut r);
            e.end_block(256);
            ls.extend(l.iter().map(|&x| f64::from(x)));
            rs.extend(r.iter().map(|&x| f64::from(x)));
        }
        (ls, rs)
    }

    /// DOUBLE: a note's twin half the detune sharp on the left, its voice half flat on the
    /// right (at full SPREAD each alone on its side), the two apart by the detune
    /// (decisions.md R45, the CA-74's R28): 7 % of 100 cents. One oscillator, LOCK.
    #[test]
    fn double_puts_the_sharp_twin_left_and_the_flat_voice_right() {
        let mut c = Controls {
            double: 0.07,
            spread: 1.0,
            lock: true,
            ..Controls::default()
        };
        c.panel.osc[1].on = false;
        c.panel.osc[2].on = false;
        let (ls, rs) = played(&c, &[57], 2.0);
        let hz = |x: &[f64]| {
            ca72_analysis::pitch::pitch(&x[24_000..], 48_000)
                .median_hz
                .expect("a pitch")
        };
        let cents = 1200.0 * (hz(&ls) / hz(&rs)).log2();
        assert!(
            (cents - 7.0).abs() < 1.5,
            "left against right {cents:.1} cents"
        );
    }

    /// DOUBLE's two voices a note, each its own, are trimmed by two's square root: a chord is
    /// about as loud with DOUBLE as without (within 2 dB); with SPREAD at 0 the pair stays in
    /// the centre, left and right the same.
    #[test]
    fn double_is_about_as_loud_and_stays_in_the_centre_without_spread() {
        let loud = |double: f64| {
            let c = Controls {
                poly: true,
                double,
                entropy: 0.3,
                spread: 1.0,
                ..Controls::default()
            };
            let (mid, side, _) = sides(&c, &[57, 61, 64]);
            mid + side
        };
        let db = 10.0 * (loud(0.35) / loud(0.0)).log10();
        assert!(
            db.abs() < 2.0,
            "DOUBLE {db:+.1} dB against one voice a note"
        );
        let centred = Controls {
            double: 0.35,
            ..Controls::default()
        };
        let (_, _, diff) = sides(&centred, &[57]);
        assert_eq!(diff, 0.0, "the pair off the centre with SPREAD at 0");
    }

    /// DOUBLE takes the placement (the CA-74's R41): with CENTER a note on the first voice
    /// has its pair in the centre, left and right the same; with EVEN out to the sides (its
    /// first voice's place is an edge). INNER at 100 % puts every pair at the edges, as EDGES
    /// did (R49): CENTER's first pair is then where EVEN's is, to the bit. One voice alone
    /// (POLY off) has its pair as far out as SPREAD whatever the placement (the kit's `ALONE`),
    /// there too.
    #[test]
    fn double_takes_the_placement_its_pairs_mirrored() {
        let doubled = |poly: bool, placement: Placement, inner: f64| Controls {
            poly,
            voices: 4,
            double: 0.35,
            spread: 1.0,
            inner,
            placement,
            ..Controls::default()
        };
        let (_, _, diff) = sides(&doubled(true, Placement::Centre, 0.0), &[57]);
        assert_eq!(diff, 0.0, "CENTER's first pair in the centre");
        let even = sides(&doubled(true, Placement::Even, 0.0), &[57]);
        assert!(even.1 > 0.1 * even.0, "EVEN: the pair out to the sides");
        let inner = sides(&doubled(true, Placement::Centre, 1.0), &[57]);
        assert_eq!(
            inner, even,
            "INNER at 100 %: CENTER's first pair at the edges"
        );
        let alone = sides(&doubled(false, Placement::Centre, 0.0), &[57]);
        assert_eq!(alone, even, "POLY off: the pair at the edges");
    }

    /// DOUBLE's pairs played by the workers are the same to the bit as on the caller's thread
    /// (a pair is one place's: whoever takes it plays both).
    #[test]
    fn doubled_voices_on_the_workers_are_the_same_to_the_bit() {
        let c = Controls {
            poly: true,
            double: 0.35,
            spread: 1.0,
            entropy: 0.3,
            ..Controls::default()
        };
        let run = |workers: usize| -> Vec<f32> {
            let mut e = Engine::new();
            e.set(&c);
            e.prepare(48_000.0, 9);
            if workers > 0 {
                e.start_workers(workers, None);
            }
            for key in [48, 55, 60, 64, 67] {
                e.event(Event::Note { key, on: true });
            }
            let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
            let mut out = Vec::new();
            for _ in 0..100 {
                e.render(&[], &mut l, &mut r);
                e.end_block(256);
                out.extend_from_slice(&l);
                out.extend_from_slice(&r);
            }
            out
        };
        let (alone, shared) = (run(0), run(2));
        assert!(alone.iter().any(|x| x.abs() > 1e-3), "silence compared");
        assert!(
            alone
                .iter()
                .zip(&shared)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    /// DOUBLE turned off lets the note go, and the twin plays its tail out on its side (INNER at
    /// 100 %: every pair at the edges, as EDGES had them), then falls silent and stops; a key
    /// pressed after reaches the voice alone (POLY off, at its first place: it scatters now).
    /// POLY off and on.
    #[test]
    fn a_twin_let_go_plays_its_tail_out_and_takes_no_new_key() {
        for poly in [false, true] {
            let c = Controls {
                poly,
                double: 0.35,
                spread: 1.0,
                inner: 1.0,
                ..Controls::default()
            };
            let mut e = Engine::new();
            e.set(&c);
            e.prepare(48_000.0, 9);
            e.event(Event::Note { key: 57, on: true });
            let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
            // (POLY off, the voice alone scatters once DOUBLE is off: its first note at the
            // first of VOICES's places. R62.)
            let (gl, gr) = voice_gains(1.0, 1.0, Placement::Even, 0, c.voices);
            let mut block = |e: &mut Engine| {
                e.render(&[], &mut l, &mut r);
                e.end_block(256);
                l.iter().zip(&r).fold((0.0, 0.0), |(d, off), (&a, &b)| {
                    (
                        f64::max(d, f64::from(a - b).abs()),
                        f64::max(off, f64::from(a * gr - b * gl).abs()),
                    )
                })
            };
            for _ in 0..40 {
                block(&mut e);
            }
            e.set(&Controls { double: 0.0, ..c });
            assert!(
                block(&mut e).0 > 1e-3,
                "poly {poly}: the tail not to either side"
            );
            for _ in 0..(10 * 48_000 / 256) {
                block(&mut e);
            }
            let live = |e: &Engine| e.slots.iter().any(|s| s.play.lock().unwrap().twin.live);
            assert!(!live(&e), "poly {poly}: a twin still live after its tail");
            e.event(Event::Note { key: 60, on: true });
            let (mut most, mut heard) = (0.0f64, 0.0f64);
            for _ in 0..40 {
                let (d, off) = block(&mut e);
                if !poly {
                    (most, heard) = (most.max(off), heard.max(d));
                }
            }
            assert!(
                most < 1e-6 * heard.max(1.0),
                "poly {poly}: a new key off its place"
            );
            assert!(poly || heard > 1e-3, "the new key silent");
            assert!(!live(&e), "poly {poly}: a twin took the new key");
            assert!(
                e.slots
                    .iter()
                    .all(|s| !s.play.lock().unwrap().twin.voice.envelopes().2),
                "poly {poly}: a twin holds a key"
            );
        }
    }

    /// A held note's energy (left and right) below and above `hz`, `skip` to `skip + 1` s in,
    /// with `c`; and the samples. A sustained sound: oscillator 1's sawtooth alone, the
    /// filter half open.
    fn bands(c: &Controls, hz: f64, skip: f64) -> (f64, f64, Vec<f64>) {
        let (l, r) = played(c, &[45], skip + 1.0);
        let from = (skip * 48_000.0) as usize;
        let y: Vec<f64> = l[from..]
            .iter()
            .zip(&r[from..])
            .map(|(a, b)| a + b)
            .collect();
        let n = y.len();
        let (mut low, mut high) = (0.0, 0.0);
        // (A plain DFT on whole-hertz bins: one second, 1 Hz apart.)
        for f in (5..4_000).step_by(5) {
            let w = 2.0 * std::f64::consts::PI * f as f64 / 48_000.0;
            let (mut re, mut im) = (0.0, 0.0);
            for (k, &v) in y.iter().enumerate() {
                re += v * (w * k as f64).cos();
                im += v * (w * k as f64).sin();
            }
            let e = (re * re + im * im) / n as f64;
            if (f as f64) < hz {
                low += e;
            } else {
                high += e;
            }
        }
        (low, high, y)
    }

    fn sustained() -> Controls {
        let mut c = Controls {
            lock: true,
            ..Controls::default()
        };
        c.panel.osc[1].on = false;
        c.panel.osc[2].on = false;
        c.panel.loudness_contour.sustain = 1.0;
        c
    }

    /// DRIVE drives the filter's input pair harder: a held note grows louder with it, and by
    /// less the higher it goes, as the pair saturates (on a sawtooth +2.7, +5.1, +8.5 and +10.7
    /// dB at 3, 6, 12 and 24 dB; decisions.md R45). (Not brighter: the ladder's stages,
    /// slew-limited by their currents, take the highs the clipping adds.)
    #[test]
    fn drive_drives_the_filter_harder() {
        let c = sustained();
        let (lo0, hi0, _) = bands(&c, 1_000.0, 0.5);
        // (AUTO GAIN off: DRIVE alone.)
        let louder = |drive: f64| {
            let c = Controls {
                drive,
                auto_gain: false,
                ..c
            };
            let (lo, hi, _) = bands(&c, 1_000.0, 0.5);
            10.0 * ((lo + hi) / (lo0 + hi0)).log10()
        };
        let (a, b, d) = (louder(6.0), louder(12.0), louder(24.0));
        assert!(a > 3.0, "DRIVE 6 dB: {a:+.1} dB");
        assert!(
            a < b && b < d,
            "DRIVE 6, 12, 24 dB: {a:+.1}, {b:+.1}, {d:+.1} dB"
        );
        assert!(d < 18.0, "DRIVE 24 dB: {d:+.1} dB, the pair not saturating");
    }

    /// LEVEL moves the output by its decibels, and at 0 dB is no gain at all.
    #[test]
    fn level_moves_the_output_by_its_decibels() {
        assert_eq!(level_gain(0.0), 1.0);
        let c = sustained();
        let (l0, _) = played(&c, &[45], 0.3);
        let (l, _) = played(&Controls { level: -6.0, ..c }, &[45], 0.3);
        let (a, b) = (
            l0.iter().map(|x| x * x).sum::<f64>(),
            l.iter().map(|x| x * x).sum::<f64>(),
        );
        let db = 10.0 * (b / a).log10();
        assert!((db + 6.0).abs() < 1e-3, "LEVEL -6 dB: {db:+.4} dB");
    }

    /// AUTO GAIN's correction glides on the output: switched off at DRIVE 12 dB, it takes the
    /// output up by the curve's correction there, a sample at a time, without a step (the
    /// ratio of the outputs every sample from 1 to the correction, by at most a hundredth a
    /// sample). Without DRIVE it asks for none: AUTO GAIN on or off, the same to the bit
    /// (decisions.md R45, the CA-74's R29).
    #[test]
    fn autos_correction_glides_and_without_drive_there_is_none() {
        let run = |c: Controls, then: Controls| -> Vec<f32> {
            let mut e = Engine::new();
            e.set(&c);
            e.prepare(48_000.0, 9);
            e.event(Event::Note { key: 45, on: true });
            let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
            let mut y = Vec::new();
            for k in 0..60 {
                if k == 40 {
                    e.set(&then);
                }
                e.render(&[], &mut l, &mut r);
                e.end_block(256);
                y.extend_from_slice(&l);
            }
            y
        };
        let c = Controls {
            drive: 12.0,
            ..sustained()
        };
        let off = Controls {
            auto_gain: false,
            ..c
        };
        let (on, switched) = (run(c, c), run(c, off));
        let at = 40 * 256;
        assert_eq!(on[..at], switched[..at]);
        let want = 10f64.powf(-AVERAGE.at(12.0, DRIVE_TOP) / 20.0);
        // (Each sample's ratio where the output is not near nought, and its index.)
        let ratios: Vec<(usize, f64)> = on[at..]
            .iter()
            .zip(&switched[at..])
            .enumerate()
            .filter(|(_, (a, _))| a.abs() > 1e-3)
            .map(|(i, (a, b))| (i, f64::from(*b) / f64::from(*a)))
            .collect();
        assert!(ratios[0].1 < 1.01, "a step: {}", ratios[0].1);
        assert!(
            ratios
                .windows(2)
                .filter(|w| w[1].0 == w[0].0 + 1)
                .all(|w| (w[1].1 - w[0].1).abs() < 0.01),
            "a step within the glide"
        );
        let last = ratios[ratios.len() - 1].1;
        assert!((last - want).abs() < 1e-3 * want, "{last} against {want}");
        let plain = sustained();
        let (a, b) = (
            run(plain, plain),
            run(
                plain,
                Controls {
                    auto_gain: false,
                    ..plain
                },
            ),
        );
        assert_eq!(a, b);
    }

    /// Blocks of `e` for `seconds`, the loudest sample's level.
    fn loudest(e: &mut Engine, seconds: f64) -> f32 {
        let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
        let mut most = 0.0f32;
        for _ in 0..(seconds * 48_000.0 / 256.0) as usize {
            e.render(&[], &mut l, &mut r);
            e.end_block(256);
            most = l.iter().chain(&r).fold(most, |a, x| a.max(x.abs()));
        }
        most
    }

    /// QUALITY at LO (decisions.md R47): the one instrument and POLY's voices play the
    /// light model, not the circuit's, and switching QUALITY lets go of the keys held, as
    /// switching POLY does.
    #[test]
    fn quality_lo_plays_the_light_voices_and_switching_it_lets_go_of_the_keys() {
        let mut c = Controls {
            potato: true,
            ..Controls::default()
        };
        let mut e = Engine::new();
        e.set(&c);
        e.prepare(48_000.0, 1);
        e.event(Event::Note { key: 57, on: true });
        assert!(loudest(&mut e, 0.5) > 0.01, "the light voice silent");
        {
            let p = e.slots[0].play.lock().unwrap();
            assert!(p.light.held(), "the key not on the light voice");
            assert!(!p.voice.envelopes().2, "the circuit's voice got the key");
        }
        c.potato = false;
        e.set(&c);
        assert!(
            !e.slots[0].play.lock().unwrap().light.held(),
            "the key still held on the light voice"
        );
        // POLY at LO: a chord's voices sound, and fall silent and free after (the DECAY
        // switch off: the contours let go at once).
        (c.potato, c.poly, c.panel.decay) = (true, true, false);
        e.set(&c);
        for key in [48, 55, 64] {
            e.event(Event::Note { key, on: true });
        }
        assert!(loudest(&mut e, 0.5) > 0.01, "POLY's light voices silent");
        assert_eq!(e.sounding(), 3);
        e.event(Event::AllNotesOff);
        // (Two seconds: the VCA's thump dies away over about one, as the circuit's does.)
        loudest(&mut e, 2.0);
        assert_eq!(e.sounding(), 0, "POLY's light voices not freed");
    }

    /// QUALITY at ULTRA (decisions.md R48): the circuit's voices play with no compromises,
    /// and switching to HI and back changes the model's setting under the key held, without
    /// letting go of it. (A short while: No Compromises is slow, more so in a debug build.)
    #[test]
    fn quality_ultra_plays_the_circuit_with_no_compromises_and_keeps_the_key() {
        let mut c = Controls {
            ultra: true,
            ..Controls::default()
        };
        let mut e = Engine::new();
        e.set(&c);
        e.prepare(48_000.0, 1);
        e.event(Event::Note { key: 57, on: true });
        assert!(loudest(&mut e, 0.05) > 0.001, "ULTRA's voice silent");
        {
            let p = e.slots[0].play.lock().unwrap();
            assert_eq!(p.voice.panel.quality, Quality::NoCompromises);
            assert!(p.voice.envelopes().2, "the key not on the circuit's voice");
        }
        c.ultra = false;
        e.set(&c);
        loudest(&mut e, 0.01);
        let p = e.slots[0].play.lock().unwrap();
        assert_eq!(p.voice.panel.quality, Quality::Potato);
        assert!(p.voice.envelopes().2, "switching to HI let go of the key");
    }

    /// At LO the one instrument plays a run at a time ([`Engine::run_one`]): the same samples,
    /// to the bit, as a sample at a time, ENTROPY's movements and the glide of its place
    /// among them.
    #[test]
    fn the_one_voice_at_lo_plays_a_run_as_a_sample_at_a_time() {
        let c = Controls {
            potato: true,
            entropy: 0.5,
            ..Controls::default()
        };
        let make = || {
            let mut e = Engine::new();
            e.set(&c);
            e.prepare(48_000.0, 3);
            e.event(Event::Note { key: 60, on: true });
            e
        };
        let (mut runs, mut samples) = (make(), make());
        let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
        let mut heard = 0.0f32;
        for block in 0..40 {
            runs.render(&[], &mut l, &mut r);
            for i in 0..256 {
                let (a, b) = samples.tick_stereo(0.0);
                assert!(
                    a.to_bits() == l[i].to_bits() && b.to_bits() == r[i].to_bits(),
                    "block {block}, sample {i}"
                );
                heard = heard.max(a.abs());
            }
            runs.end_block(256);
            samples.end_block(256);
        }
        assert!(heard > 0.01, "silent");
    }

    /// At LO DOUBLE's pair and UNISON's voices are placed and levelled as at HI: the pair to
    /// either side at full SPREAD, the unison about as loud as one voice.
    #[test]
    fn lo_places_double_and_levels_unison_as_hi() {
        let doubled = Controls {
            potato: true,
            poly: true,
            double: 1.0,
            spread: 1.0,
            ..Controls::default()
        };
        let (mid, side, _) = sides(&doubled, &[57]);
        assert!(side > 0.05 * mid && mid > 0.0, "mid {mid}, side {side}");
        let c = Controls {
            potato: true,
            voices: 8,
            entropy: 0.3,
            ..Controls::default()
        };
        let one = held_power(&c, 45, 1.0, 2.0);
        let all = held_power(&Controls { unison: true, ..c }, 45, 1.0, 2.0);
        let db = 10.0 * (all / one).log10();
        assert!(db.abs() < 2.0, "UNISON {db:+.1} dB against one voice");
    }

    /// AUTO GAIN measures a sound at LO with the light voices (`drive.rs`), its key the sound's
    /// at LO; at HI keyed as before.
    #[test]
    fn auto_gain_measures_lo_with_the_light_voices() {
        let hi = Controls::default();
        let lo = Controls { potato: true, ..hi };
        assert_ne!(crate::drive::sound(&lo), crate::drive::sound(&hi));
        let probe = Probe::new(&lo, MIN_VOICE_RATE);
        assert!(probe.voices.iter().all(|v| v.potato));
        let e = probe.energy(0.0);
        assert!(e.is_finite() && e > 0.0, "{e}");
        assert!(
            Probe::new(&hi, MIN_VOICE_RATE)
                .voices
                .iter()
                .all(|v| !v.potato)
        );
    }

    /// The one instrument at 48 kHz with `c`.
    fn one_instrument(c: &Controls) -> Engine {
        let mut e = Engine::new();
        e.set(c);
        e.prepare(48_000.0, 1);
        e
    }

    /// `seconds` of blocks of 256 samples, `ext` at EXTERNAL INPUT: the output's peak.
    fn played_for(e: &mut Engine, seconds: f64, ext: f32) -> f32 {
        let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
        let x = [ext; 256];
        let mut peak = 0.0f32;
        for _ in 0..(seconds * 48_000.0 / 256.0).ceil() as usize {
            e.render(&x, &mut l, &mut r);
            e.end_block(256);
            peak = l.iter().chain(&r).fold(peak, |m, y| m.max(y.abs()));
        }
        peak
    }

    /// Played until asleep, at most `seconds`: whether it fell asleep.
    fn sleeps_within(e: &mut Engine, seconds: f64) -> bool {
        let mut t = 0.0;
        while !e.asleep() && t < seconds {
            played_for(e, 0.1, 0.0);
            t += 0.1;
        }
        e.asleep()
    }

    /// The one instrument sleeps once silent and settled, never while a key is down, and wakes
    /// for a note, a control moved, a wheel, All Notes Off and a sound at EXTERNAL INPUT; asleep,
    /// the output is silent. So at QUALITY LO and with DOUBLE; never with POLY or UNISON, whose
    /// voices are freed instead, nor while A-440 sounds (decisions.md R68).
    #[test]
    fn a_silent_settled_instrument_sleeps_and_wakes() {
        let mut e = one_instrument(&Controls::default());
        assert!(sleeps_within(&mut e, 30.0), "awake with nothing played");
        assert_eq!(played_for(&mut e, 0.1, 0.0), 0.0);
        e.event(Event::Note { key: 60, on: true });
        assert!(!e.asleep());
        assert!(played_for(&mut e, 1.0, 0.0) > 0.01, "the note silent");
        played_for(&mut e, 4.0, 0.0);
        assert!(!e.asleep(), "asleep with a key down");
        e.event(Event::Note { key: 60, on: false });
        assert!(
            sleeps_within(&mut e, 30.0),
            "awake 30 s after the key was let go"
        );
        let mut c = Controls::default();
        c.panel.cutoff = 0.7;
        e.set(&c);
        assert!(!e.asleep(), "a control moved left it asleep");
        assert!(sleeps_within(&mut e, 30.0));
        e.set(&c);
        assert!(e.asleep(), "the same controls again woke it");
        e.event(Event::Modulation(0.5));
        assert!(!e.asleep(), "the wheel left it asleep");
        assert!(sleeps_within(&mut e, 30.0));
        e.release_all();
        assert!(!e.asleep(), "All Notes Off left it asleep");
        assert!(sleeps_within(&mut e, 30.0));
        let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
        e.render(&[0.0, 0.0, 0.25], &mut l, &mut r);
        assert!(!e.asleep(), "a sound at EXTERNAL INPUT left it asleep");
        e.never_sleep(true);
        assert!(!sleeps_within(&mut e, 5.0), "asleep when told never to");
        for (what, c) in [
            (
                "QUALITY LO",
                Controls {
                    potato: true,
                    ..Controls::default()
                },
            ),
            (
                "DOUBLE",
                Controls {
                    double: 0.3,
                    ..Controls::default()
                },
            ),
        ] {
            let mut e = one_instrument(&c);
            assert!(
                sleeps_within(&mut e, 30.0),
                "{what}: awake with nothing played"
            );
            e.event(Event::Note { key: 60, on: true });
            assert!(
                played_for(&mut e, 0.5, 0.0) > 0.01,
                "{what}: the note silent"
            );
            e.event(Event::Note { key: 60, on: false });
            assert!(sleeps_within(&mut e, 30.0), "{what}: awake after the note");
        }
        for (what, c) in [
            (
                "POLY",
                Controls {
                    poly: true,
                    ..Controls::default()
                },
            ),
            (
                "UNISON",
                Controls {
                    unison: true,
                    ..Controls::default()
                },
            ),
        ] {
            let mut e = one_instrument(&c);
            assert!(
                !sleeps_within(&mut e, 5.0),
                "{what}: the one instrument's sleep"
            );
        }
        let mut c = Controls::default();
        c.panel.a440 = true;
        let mut e = one_instrument(&c);
        assert!(!sleeps_within(&mut e, 5.0), "asleep while A-440 sounds");
    }

    /// An instrument woken from sleep plays its next note as one that never slept: at rest its
    /// contours within 20 uV of the other's, through the note its keyboard voltage the same,
    /// its contours within 0.5 mV of their 5 V (a sample or two of time: they tick at a
    /// sub-rate whose cycle stopped while it slept), its level within 0.1 dB. Otherwise only
    /// its free-running oscillators' and noise's phases differ (decisions.md R68).
    #[test]
    fn a_woken_instrument_plays_its_next_note_as_one_that_never_slept() {
        let (mut a, mut b) = (
            one_instrument(&Controls::default()),
            one_instrument(&Controls::default()),
        );
        b.never_sleep(true);
        for e in [&mut a, &mut b] {
            e.event(Event::Note { key: 48, on: true });
            played_for(e, 0.5, 0.0);
            e.event(Event::Note { key: 48, on: false });
        }
        assert!(sleeps_within(&mut a, 30.0));
        played_for(&mut b, 30.0, 0.0);
        assert!(!b.asleep());
        let probe = |e: &Engine| {
            let p = e.slots[0].play.lock().unwrap();
            let (f, l, _) = p.voice.envelopes();
            (f, l, p.voice.probe().0)
        };
        let (x, y) = (probe(&a), probe(&b));
        let rested = (x.0 - y.0).abs().max((x.1 - y.1).abs());
        assert!(rested < 2e-5, "the contours {rested:e} V apart at rest");
        for e in [&mut a, &mut b] {
            e.event(Event::Note { key: 55, on: true });
        }
        assert!(!a.asleep());
        let (mut contours, mut keyboard, mut power) = (0.0f64, 0.0f64, [0.0f64; 2]);
        let (mut l, mut r) = ([0.0f32; 256], [0.0f32; 256]);
        for _ in 0..(0.5 * 48_000.0 / 256.0) as usize {
            for (k, e) in [&mut a, &mut b].into_iter().enumerate() {
                e.render(&[], &mut l, &mut r);
                e.end_block(256);
                power[k] += l.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>();
            }
            let (x, y) = (probe(&a), probe(&b));
            contours = contours.max((x.0 - y.0).abs()).max((x.1 - y.1).abs());
            keyboard = keyboard.max((x.2 - y.2).abs());
        }
        assert!(keyboard < 1e-9, "the keyboard {keyboard:e} V apart");
        assert!(contours < 5e-4, "the contours {contours:e} V apart");
        let db = 10.0 * (power[0] / power[1]).log10();
        assert!(db.abs() < 0.1, "levels {db:.3} dB apart");
    }
}
