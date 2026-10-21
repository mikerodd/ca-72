//! The CA-72 as a VST3 and CLAP plug-in: the circuit model's voice in its real-time quality,
//! played by MIDI, its EXTERNAL INPUT the side chain; one voice, or with POLY one per note
//! (up to ten), placed across the stereo output by SPREAD.

pub mod compat;
pub mod editor;
pub mod learn;
pub mod learning;
pub mod library;
pub mod local_notes;
pub mod opening;
pub mod params;
pub mod playing;
pub mod presets;
mod resizing;
// The audio engine, apart from any plug-in format (`crates/ca72-engine`), re-exported so that
// this plug-in's own modules and tests keep reaching it as `crate::engine` and the like.
pub use ca72_engine::{character, drive, engine, helper, pool};
pub mod settings;
#[cfg(feature = "standalone-production")]
pub mod standalone_error;
#[cfg(feature = "standalone-production")]
pub mod standalone_settings;
pub mod update;

use std::sync::Arc;

use nih_plug::prelude::*;
use nih_plug::wrapper::state::ParamValue;

use crate::editor::{Ca72Editor, Meters};
use crate::engine::{
    ALL_NOTES_OFF, ALL_SOUND_OFF, Engine, Event, MODULATION_WHEEL, RESET_CONTROLLERS,
};
use crate::helper::{Helper, MEND, SERVE};
use crate::learn::{Dezip, Incoming};
use crate::params::Ca72Params;

pub struct Ca72 {
    params: Arc<Ca72Params>,
    engine: Engine,
    meters: Arc<Meters>,
    local_notes: Arc<local_notes::LocalNotes>,
    note_sources: local_notes::Sources,
    local_generation: u64,
    panic_generation: u64,
    deferred_local: Option<local_notes::LocalEvent>,
    /// The host's sample rate (for each block's deadline).
    rate: f64,
    /// The plug-in's own thread for what the audio thread asks of it (decisions.md R23),
    /// started when the plug-in is first initialised (none if the system would not start it).
    helper: Option<Helper>,
    /// The knobs learned MIDI controllers have just moved, gliding in the voices (decisions.md
    /// R34).
    dezip: Dezip,
    /// AUTO GAIN's curve's writes as the audio thread last read them (`drive.rs`).
    curve_seen: u64,
}

impl std::fmt::Debug for Ca72 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ca72")
            .field("engine", &self.engine)
            .finish()
    }
}

impl Default for Ca72 {
    fn default() -> Self {
        Ca72::with_params(Ca72Params::default())
    }
}

impl Ca72 {
    /// Its parameters, typed (the host's view is [`Plugin::params`]).
    pub fn parameters(&self) -> &Arc<Ca72Params> {
        &self.params
    }

    fn with_params(params: Ca72Params) -> Ca72 {
        Ca72 {
            params: Arc::new(params),
            engine: Engine::new(),
            meters: Arc::new(Meters::default()),
            local_notes: Arc::new(local_notes::LocalNotes::default()),
            note_sources: local_notes::Sources::default(),
            local_generation: 0,
            panic_generation: 0,
            deferred_local: None,
            rate: 48_000.0,
            helper: None,
            dezip: Dezip::default(),
            curve_seen: 0,
        }
    }
}

impl Drop for Ca72 {
    /// The helper stopped and joined first (a mend in progress stops before its next voice),
    /// then the voices' workers, as the engine drops: when the host's destroy returns, no
    /// thread of the plug-in's runs, and the host may unload it (decisions.md R23).
    fn drop(&mut self) {
        drop(self.helper.take());
    }
}

/// The side chain at sample `i`, its channels mixed to mono: a channel shorter than the block
/// (one the host did not supply may be) or a sample that is not finite counts as 0
/// (decisions.md R18).
fn mono(channels: &[&mut [f32]], i: usize) -> f32 {
    if channels.is_empty() {
        return 0.0;
    }
    channels
        .iter()
        .map(|c| c.get(i).copied().filter(|x| x.is_finite()).unwrap_or(0.0))
        .sum::<f32>()
        / channels.len() as f32
}

/// The host's MIDI as the engine takes it (any channel), if it takes it.
fn event_of(e: NoteEvent<()>) -> Option<Event> {
    Some(match e {
        NoteEvent::NoteOn { note, velocity, .. } => Event::Note {
            key: note,
            on: velocity > 0.0,
        },
        NoteEvent::NoteOff { note, .. } | NoteEvent::Choke { note, .. } => Event::Note {
            key: note,
            on: false,
        },
        NoteEvent::MidiPitchBend { value, .. } => Event::PitchBend(value * 2.0 - 1.0),
        NoteEvent::MidiCC { cc, value, .. } => match cc {
            MODULATION_WHEEL => Event::Modulation(value),
            RESET_CONTROLLERS => Event::ResetControllers,
            ALL_SOUND_OFF => Event::AllSoundOff,
            ALL_NOTES_OFF => Event::AllNotesOff,
            _ => return None,
        },
        _ => return None,
    })
}

impl Ca72 {
    /// A MIDI event: a learnable control change to MIDI Learn (decisions.md R34), anything else
    /// as before (the reserved control changes the engine follows, CC 1, 120, 121 and 123, among
    /// them). Whether it set a parameter.
    fn midi(&mut self, e: NoteEvent<()>, context: &mut impl ProcessContext<Self>) -> bool {
        if let NoteEvent::MidiCC {
            timing,
            channel,
            cc,
            value,
        } = e
        {
            if learn::reserved(cc).is_none() {
                return match self.params.midi_map.incoming(channel, cc) {
                    Incoming::Assigned(i) => self.learned(i, value, timing, context),
                    // Caught for the parameter being learned: the sound is not changed by it.
                    Incoming::Caught | Incoming::Unassigned => false,
                };
            }
            // (Said in the editor while it learns: this one is not learned.)
            self.params.midi_map.refuse(cc);
        }
        if let Some(event) = event_of(e) {
            if let Event::Note { key, on } = event {
                let channel = match e {
                    NoteEvent::NoteOn { channel, .. }
                    | NoteEvent::NoteOff { channel, .. }
                    | NoteEvent::Choke { channel, .. } => channel,
                    _ => 0,
                };
                if self.note_sources.physical(channel, key, on) {
                    self.engine.event(event);
                }
            } else {
                if matches!(event, Event::AllSoundOff | Event::AllNotesOff) {
                    self.clear_note_sources();
                }
                self.engine.event(event);
            }
        }
        false
    }

    fn clear_note_sources(&mut self) {
        self.local_notes.release();
        self.deferred_local = None;
        self.local_generation = self.local_notes.generation();
        self.note_sources.clear();
        self.local_notes.publish([0, 0]);
    }

    /// Apply one boundary reset before local input. A paired standalone request may arrive
    /// from UI just after the host captured its block: leave fresh notes queued until its
    /// hardware panic arrives, so a later reset cannot kill a note already played.
    fn local_reset(&mut self, standalone_panic: bool, token: Option<u64>) -> bool {
        let awaiting_standalone = self.local_notes.standalone_panic_pending();
        if awaiting_standalone && !standalone_panic {
            self.release_local_generation();
            return false;
        }
        let panic = self.local_notes.panic_generation();
        let ui_panic = panic != self.panic_generation;
        if standalone_panic || ui_panic {
            if standalone_panic {
                if let Some(token) = token {
                    self.local_notes.finish_standalone_panic(token);
                }
                // An external host panic invalidates pending local input. The explicit UI
                // button already advanced its epoch, so coalesce that paired request.
                if token.is_none() && !ui_panic && !awaiting_standalone {
                    self.local_notes.release();
                }
            }
            self.note_sources.clear();
            self.local_generation = self.local_notes.generation();
            self.local_notes.publish([0, 0]);
            self.engine.event(Event::AllSoundOff);
            self.engine.event(Event::ResetControllers);
            if !self.local_notes.standalone_panic_pending() {
                self.panic_generation = panic;
            }
        }
        self.release_local_generation();
        !self.local_notes.standalone_panic_pending()
    }

    fn release_local_generation(&mut self) {
        let generation = self.local_notes.generation();
        if generation != self.local_generation {
            self.local_generation = generation;
            let engine = &mut self.engine;
            self.note_sources
                .release_local(|key| engine.event(Event::Note { key, on: false }));
        }
    }

    /// A bounded batch at the next callback's boundary; host events retain their sample offsets.
    fn local_events(&mut self, standalone_panic: bool, token: Option<u64>) {
        if !self.local_reset(standalone_panic, token) {
            return;
        }
        for _ in 0..local_notes::CAPACITY {
            let Some(e) = self
                .deferred_local
                .take()
                .or_else(|| self.local_notes.pop())
            else {
                break;
            };
            // Observe a UI reset that arrived during the drain before accepting its fresh
            // epoch. If it awaits hardware panic, retain this one event for the next callback.
            if !self.local_reset(false, None) {
                self.deferred_local = Some(e);
                break;
            }
            if e.generation == self.local_generation && self.note_sources.local(e.note, e.on) {
                self.engine.event(Event::Note {
                    key: e.note,
                    on: e.on,
                });
            }
        }
    }

    /// Learnable parameter `i`'s controller at `value` (0..1, as the host gives a control change;
    /// 7 bits), `timing` samples into the block: the parameter set through the host, as the
    /// host's automation sets it (the editor and the host following, the state holding it), and
    /// a knob gliding there in the voices ([`Dezip`]). Whether it changed.
    fn learned(
        &mut self,
        i: usize,
        value: f32,
        timing: u32,
        context: &mut impl ProcessContext<Self>,
    ) -> bool {
        let seven = (value.clamp(0.0, 1.0) * 127.0).round() as u8;
        let Some(target) = learn::target(&self.params, i) else {
            return false;
        };
        let normalized = learn::normalized_for(&self.params, i, target, seven);
        if target.at(normalized) {
            return false;
        }
        let knob = learn::knob_param(&self.params, i);
        let from = knob.map(|k| self.dezip.value(k));
        context.set_parameter_normalized(target.ptr(), normalized, timing);
        if let (Some(k), Some(from)) = (knob, from) {
            self.dezip.start(i, k, from, k.value());
        }
        true
    }

    /// The engine's controls: the parameters, a knob a learned controller has just moved where
    /// its glide has it.
    fn controls(&self) -> crate::engine::Controls {
        self.params.controls_with(&|k| self.dezip.value(k))
    }

    /// What the engine asks done off the audio thread since the last block, asked of the
    /// helper (decisions.md R23): allocates nothing and does not wait.
    fn ask_helper(&mut self) {
        let mend = if self.engine.take_mending() { MEND } else { 0 };
        let serve = if self.engine.take_serving() { SERVE } else { 0 };
        if let Some(h) = &self.helper {
            h.ask(mend | serve);
        }
    }
}

/// EXTERNAL INPUT: a side chain of as many channels as the output (mixed to mono at the jack).
const SIDE_CHAIN_STEREO: &[std::num::NonZeroU32] = &[new_nonzero_u32(2)];
const SIDE_CHAIN_MONO: &[std::num::NonZeroU32] = &[new_nonzero_u32(1)];

const fn layout(channels: u32, name: &'static str) -> AudioIOLayout {
    AudioIOLayout {
        main_input_channels: None,
        main_output_channels: Some(new_nonzero_u32(channels)),
        aux_input_ports: if channels == 1 {
            SIDE_CHAIN_MONO
        } else {
            SIDE_CHAIN_STEREO
        },
        names: PortNames {
            layout: Some(name),
            main_output: Some("Output"),
            aux_inputs: &["External Input"],
            ..PortNames::const_default()
        },
        ..AudioIOLayout::const_default()
    }
}

impl Plugin for Ca72 {
    const NAME: &'static str = "CA-72";
    const VENDOR: &'static str = "Idle Foundry";
    const URL: &'static str = "https://github.com/idlefoundry/ca-72";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[layout(2, "Stereo"), layout(1, "Mono")];

    const MIDI_INPUT: MidiConfig = MidiConfig::MidiCCs;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    /// None: nih-plug's background thread, shared by every instance, holds a plug-in alive
    /// while it runs one of its tasks, past the host's destroy. The plug-in's own helper does
    /// that work instead (decisions.md R23).
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    /// A session saved before ENTROPY (decisions.md R14) holds ANALOG's `analog`, which no
    /// parameter takes: it opens with ENTROPY at its default, not at what the instance had
    /// (R18).
    ///
    /// A state without MIDI assignments (saved before MIDI Learn) opens with none, whatever the
    /// instance had; a table that is not understood is none either (decisions.md R34).
    ///
    /// A state 0.1 saved has the knobs whose laws 0.2.0's calibration measured moved to where
    /// they set what they set then (`compat`, decisions.md R-TRANSLATE).
    ///
    /// A session saved with SCATTER's EDGES, no longer offered (INNER does its work: decisions.md
    /// R49), or with a placement not understood, opens with EVEN, the default, not at what
    /// the instance had.
    fn filter_state(state: &mut PluginState) {
        learn::filter_state(&mut state.fields);
        // A state saved before DOUBLE was a switch of its own: DOUBLE where its DETUNE is
        // above 0 (decisions.md R45).
        if !state.params.contains_key("doubled") {
            let detuned =
                matches!(state.params.get("double"), Some(ParamValue::F32(d)) if *d > 0.0);
            state
                .params
                .insert("doubled".to_owned(), ParamValue::Bool(detuned));
        }
        // A state without AUTO GAIN's curve (saved before DRIVE), or with one that is not
        // understood, opens with the average's, whatever the instance had (decisions.md
        // R45, the CA-74's R29).
        let curve = state
            .fields
            .get(drive::STATE_KEY)
            .and_then(|t| serde_json::from_str::<drive::Saved>(t).ok())
            .unwrap_or_else(drive::unmeasured);
        if let Ok(t) = serde_json::to_string(&curve) {
            state.fields.insert(drive::STATE_KEY.to_owned(), t);
        }
        state.params.remove("analog");
        if !state.params.contains_key("entropy") {
            let entropy = Ca72Params::default().entropy.default_plain_value();
            state
                .params
                .insert("entropy".to_owned(), ParamValue::F32(entropy));
        }
        // A state saved before NOTE PRIORITY: the instrument's own, LOW, whatever the instance
        // had (decisions.md R50).
        if !state.params.contains_key("note_priority") {
            let low = Ca72Params::default().note_priority.default_plain_value();
            state.params.insert(
                "note_priority".to_owned(),
                ParamValue::I32(low.to_index() as i32),
            );
        }
        // And before RETRIGGER: off, single triggering, the instrument's own.
        if !state.params.contains_key("retrigger") {
            state
                .params
                .insert("retrigger".to_owned(), ParamValue::Bool(false));
        }
        let offered = |id: &str| params::Scatter::ids().is_some_and(|ids| ids.contains(&id));
        if matches!(state.params.get("placement"), Some(ParamValue::String(id)) if !offered(id)) {
            let even = Ca72Params::default().placement.default_plain_value();
            state.params.insert(
                "placement".to_owned(),
                ParamValue::I32(even.to_index() as i32),
            );
        }
        // A state 0.1 saved: each knob whose law 0.2.0's calibration changed moved to where it
        // sets what it set (decisions.md R-TRANSLATE).
        if compat::before_0_2(&state.version) {
            compat::translate(&mut state.params);
        }
    }

    fn editor(&mut self, _executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        Some(Box::new(Ca72Editor::with_notes(
            self.params.clone(),
            self.meters.clone(),
            self.local_notes.clone(),
        )))
    }

    fn initialize(
        &mut self,
        _layout: &AudioIOLayout,
        config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        // How this computer's FREQUENCY dials read: a learned controller turns the knob through
        // it, the editor open or not (decisions.md R54).
        if !cfg!(test) {
            self.params
                .set_freq_law(crate::settings::Settings::load().frequency);
        }
        self.engine.set(&self.params.controls());
        self.engine
            .prepare(f64::from(config.sample_rate), self.params.seed());
        let block = config.max_buffer_size.max(1) as usize;
        let rate = f64::from(config.sample_rate);
        self.rate = rate;
        self.dezip.prepare(rate);
        // (AUTO GAIN's measurements on the helper thread: decisions.md R45.)
        self.engine.calibrate_with(self.params.drive_curve.clone());
        if self.helper.is_none() {
            self.helper = Helper::start(
                self.engine.spares(),
                self.engine.crew(),
                self.params.drive_curve.clone(),
            )
            .ok();
        }
        // POLY's workers, audio threads for the host's largest block (decisions.md R11), held
        // only while POLY is on: started here if it is, else on the helper thread when it is
        // switched on (R21, R23). A host loading a state into the running instance comes here
        // again with the audio thread waiting: running workers are kept then (R18).
        let period = std::time::Duration::from_secs_f64(block as f64 / rate);
        self.engine
            .start_workers(crate::engine::default_workers(), Some(period));
        true
    }

    /// POLY's workers stopped and given back to the process's budget (decisions.md R21): the
    /// host initialises the plug-in again before it next plays.
    fn deactivate(&mut self) {
        self.engine.stop_workers();
    }

    fn reset(&mut self) {
        self.clear_note_sources();
        self.dezip.clear();
        self.engine.set(&self.params.controls());
        self.engine.release_all();
        self.engine.event(Event::ResetControllers);
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // (A knob whose parameter something else has set since a learned controller moved it
        // stops gliding: the host's automation, the editor and presets set the voices as ever.)
        self.dezip.follow(&self.params);
        if let Some(c) = self.params.drive_curve.read(&mut self.curve_seen) {
            self.engine.set_curve(c);
        }
        self.engine.set(&self.controls());
        let panic_token = context.standalone_panic_token();
        let standalone_panic = context.take_standalone_panic();
        self.local_events(standalone_panic, panic_token);
        let len = buffer.samples();
        // The voices' deadline: a share of the block's period from now (decisions.md R11);
        // none rendering offline, where every voice is waited for (R18).
        let period = len as f64 / self.rate;
        let deadline = (context.process_mode() != ProcessMode::Offline).then(|| {
            std::time::Instant::now()
                + std::time::Duration::from_secs_f64(period * crate::engine::DEADLINE)
        });
        self.engine.set_deadline(deadline);
        let side = aux.inputs.first().map(|b| b.as_slice_immutable());
        let out = buffer.as_slice();
        let mut next = context.next_event();
        // The samples between one event and the next, a run at a time (the engine plays
        // POLY's voices one after another over a run: decisions.md R11), the side chain mixed
        // to mono. A learned MIDI controller sets its parameter at its event, and the voices take
        // it from there; while a knob it moved glides, a run is at most `DEZIP_STEP` samples
        // (decisions.md R34).
        const RUN: usize = 128;
        let (mut ext, mut l, mut r) = ([0.0f32; RUN], [0.0f32; RUN], [0.0f32; RUN]);
        let mut i = 0;
        let mut glided = false;
        while i < len {
            let mut learned = false;
            while let Some(e) = next {
                if e.timing() as usize > i {
                    break;
                }
                learned |= self.midi(e, context);
                next = context.next_event();
            }
            if learned || glided || self.dezip.moving() {
                self.dezip.follow(&self.params);
                self.engine.set(&self.controls());
            }
            let mut end = next
                .map_or(len, |e| e.timing() as usize)
                .min(len)
                .min(i + RUN);
            if self.dezip.moving() {
                end = end.min(i + learn::DEZIP_STEP);
            }
            let n = end - i;
            let ext = match side {
                Some(s) if !s.is_empty() => {
                    for (k, x) in ext[..n].iter_mut().enumerate() {
                        *x = mono(s, i + k);
                    }
                    &ext[..n]
                }
                _ => &[][..],
            };
            // Left and right (SPREAD's voices placed); on a mono output, the two together.
            match &mut out[..] {
                [left, right, ..] => self
                    .engine
                    .render(ext, &mut left[i..end], &mut right[i..end]),
                [only] => {
                    self.engine.render(ext, &mut l[..n], &mut r[..n]);
                    for k in 0..n {
                        only[i + k] = 0.5 * (l[k] + r[k]);
                    }
                }
                [] => self.engine.render(ext, &mut l[..n], &mut r[..n]),
            }
            glided = self.dezip.moving();
            self.dezip.advance(n, &self.params);
            i = end;
        }
        // (Past the block's end: a parameter set here reaches the voices at the next block's
        // start.)
        while let Some(e) = next {
            self.midi(e, context);
            next = context.next_event();
        }
        self.local_notes.publish(self.note_sources.held());
        let lamp = self.engine.end_block(len);
        self.meters
            .publish(lamp, self.engine.midi_wheels(), self.engine.sounding_mask());
        // The voices' levels and the output's, for the editor's drops and QUALITY's lamps:
        // only while an editor is open (else a flag read a block, and nothing more).
        if self.meters.watched() {
            let peak = out.iter().fold(0.0f32, |m, ch| {
                ch.iter().take(len).fold(m, |m, v| m.max(v.abs()))
            });
            self.meters.publish_levels(self.engine.levels(), peak);
        }
        self.ask_helper();
        ProcessStatus::KeepAlive
    }
}

impl ClapPlugin for Ca72 {
    const CLAP_ID: &'static str = "com.idlefoundry.ca-72";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("A synthesizer modelled from its circuit, one voice or up to ten");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::Instrument,
        ClapFeature::Synthesizer,
        ClapFeature::Mono,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for Ca72 {
    const VST3_CLASS_ID: [u8; 16] = *b"IdleFoundryCA72!";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Instrument, Vst3SubCategory::Synth];
}

#[cfg(test)]
mod tests {
    use super::*;
    use ca72::modulation::PITCH_WHEEL_SEMITONES;
    use nih_plug::context::process::TestProcessContext;
    use std::collections::BTreeMap;

    /// A context with no host, for the plug-in's parameters (nih-plug's `TestProcessContext`).
    fn hostless(p: &Ca72) -> TestProcessContext<Ca72> {
        let params: Arc<dyn Params> = p.params.clone();
        TestProcessContext::new(params, 48_000.0, ProcessMode::Realtime)
    }

    fn note(timing: u32, channel: u8, pitch: u8, on: bool) -> NoteEvent<()> {
        if on {
            NoteEvent::NoteOn {
                timing,
                voice_id: None,
                channel,
                note: pitch,
                velocity: 1.0,
            }
        } else {
            NoteEvent::NoteOff {
                timing,
                voice_id: None,
                channel,
                note: pitch,
                velocity: 0.0,
            }
        }
    }

    #[test]
    fn local_keyboard_plays_the_same_samples_as_host_midi_at_the_block_boundary() {
        let (mut local, mut physical) = (initialised(false), initialised(false));
        // A new instance intentionally gets a fresh noise/mismatch seed. Match the persisted
        // seed as loading the same saved patch does before comparing sample identity.
        for p in [&mut local, &mut physical] {
            p.params
                .noise_seed
                .store(0xca72, std::sync::atomic::Ordering::Relaxed);
            p.engine.prepare(48_000.0, p.params.seed());
        }
        let (mut lc, mut pc) = (hostless(&local), hostless(&physical));
        for i in 0..32 {
            if i == 0 || i == 16 {
                let on = i == 0;
                local.local_notes.note(60, on);
                pc.push_event(note(0, 0, 60, on));
            }
            let a = block(&mut local, &mut lc, 256);
            let b = block(&mut physical, &mut pc, 256);
            assert!(a == b, "local and physical MIDI differ in block {i}");
        }
    }

    #[test]
    #[cfg(feature = "standalone-production")]
    fn standalone_invalid_callbacks_and_midi_resets_do_not_acknowledge_ui_tokens() {
        use nih_plug::wrapper::production::Processor;
        use plugin_kit_standalone::{AudioBlock, AudioProcessor, MidiEvent, ProcessConfig};

        for kind in [0, 120, 123] {
            let p = Ca72::default();
            let notes = p.local_notes.clone();
            let mut playing = crate::playing::Playing::default();
            playing.notes = notes.clone();
            let (mut adapter, _) = Processor::new(p, &[]).expect("standalone adapter");
            adapter
                .prepare(ProcessConfig {
                    sample_rate: 48_000.0,
                    max_buffer_frames: 128,
                })
                .expect("prepared");
            let token = playing.panic_standalone();
            let (mut left, mut right) = ([0.0; 128], [0.0; 128]);
            let mut run = |adapter: &mut Processor<Ca72>, panic, midi: &[_]| {
                adapter.process(AudioBlock {
                    left: &mut left,
                    right: &mut right,
                    input_left: &[],
                    input_right: &[],
                    midi,
                    panic,
                });
            };
            if kind == 0 {
                adapter.panic_token(Some(token));
                adapter.process(AudioBlock {
                    left: &mut [0.0],
                    right: &mut [],
                    input_left: &[],
                    input_right: &[],
                    midi: &[],
                    panic: true,
                });
            } else {
                run(
                    &mut adapter,
                    false,
                    &[MidiEvent {
                        source: 1,
                        data: [0xb0, kind, 0],
                        len: 3,
                    }],
                );
            }
            playing.sync();
            playing.press(69);
            // No new metadata is supplied: the invalid call must have consumed its tag.
            run(&mut adapter, kind == 0, &[]);
            assert_eq!(
                notes.held(),
                [0, 0],
                "invalid/generic/CC reset cannot finish the UI request"
            );
            adapter.panic_token(Some(token));
            run(&mut adapter, true, &[]);
            assert_eq!(
                notes.held(),
                [0, 1 << 5],
                "matching boundary admits fresh input"
            );
        }
    }

    #[test]
    #[cfg(feature = "standalone-production")]
    fn repeated_standalone_panics_wait_for_their_own_captured_boundary() {
        use nih_plug::wrapper::production::Processor;
        use plugin_kit_standalone::{AudioBlock, AudioProcessor, ProcessConfig};

        // Old captured UI request, generic reset, and coalesced requests respectively.
        for preceding in [Some(1), Some(0), None] {
            let p = Ca72::default();
            let notes = p.local_notes.clone();
            let mut playing = crate::playing::Playing::default();
            playing.notes = notes.clone();
            let (mut adapter, _) = Processor::new(p, &[]).expect("standalone adapter");
            adapter
                .prepare(ProcessConfig {
                    sample_rate: 48_000.0,
                    max_buffer_frames: 128,
                })
                .expect("prepared");
            let (mut left, mut right) = ([0.0; 128], [0.0; 128]);
            let mut run = |token| {
                adapter.panic_token(token);
                adapter.process(AudioBlock {
                    left: &mut left,
                    right: &mut right,
                    input_left: &[],
                    input_right: &[],
                    midi: &[],
                    panic: true,
                });
            };
            let first = playing.panic_standalone();
            let second = playing.panic_standalone();
            playing.press(69);
            let fresh_epoch = notes.generation();
            if let Some(which) = preceding {
                run((which == 1).then_some(first));
                assert_eq!(
                    notes.held(),
                    [0, 0],
                    "old/generic capture cannot acknowledge newer UI input"
                );
            }
            run(Some(second));
            assert_eq!(
                notes.generation(),
                fresh_epoch,
                "matching capture cannot age fresh input"
            );
            assert_eq!(
                notes.held(),
                [0, 1 << 5],
                "fresh note survives its own boundary"
            );
        }
    }

    #[test]
    #[cfg(feature = "standalone-production")]
    fn standalone_ui_and_core_panics_coalesce_at_the_same_or_next_boundary() {
        use nih_plug::wrapper::production::Processor;
        use plugin_kit_standalone::{AudioBlock, AudioProcessor, MidiEvent, ProcessConfig};

        for split_boundary in [false, true] {
            let p = Ca72::default();
            let notes = p.local_notes.clone();
            let mut playing = crate::playing::Playing::default();
            playing.notes = notes.clone();
            let (mut adapter, _) = Processor::new(p, &[]).expect("standalone adapter");
            adapter
                .prepare(ProcessConfig {
                    sample_rate: 48_000.0,
                    max_buffer_frames: 128,
                })
                .expect("prepared");
            let (mut left, mut right) = ([0.0; 128], [0.0; 128]);
            let mut run = |panic, token, midi: &[_]| {
                adapter.panic_token(token);
                adapter.process(AudioBlock {
                    left: &mut left,
                    right: &mut right,
                    input_left: &[],
                    input_right: &[],
                    midi,
                    panic,
                });
            };
            let physical = |on| MidiEvent {
                source: 1,
                data: [if on { 0x90 } else { 0x80 }, 60, 100],
                len: 3,
            };
            playing.press(59);
            run(false, None, &[physical(true)]);
            assert_eq!(notes.held(), [(1 << 59) | (1 << 60), 0]);
            // The editor marks the paired request and changes the UI epoch. The core's
            // AudioBlock panic can belong to this callback or the following one.
            let token = playing.panic_standalone();
            playing.press(69);
            let fresh_epoch = notes.generation();
            if split_boundary {
                run(false, None, &[]);
                assert_eq!(
                    notes.held(),
                    [1 << 60, 0],
                    "fresh local input waits for hardware reset"
                );
            }
            run(true, Some(token), &[]);
            assert_eq!(
                notes.generation(),
                fresh_epoch,
                "the two panic paths share one UI epoch"
            );
            assert_eq!(
                notes.held(),
                [0, 1 << 5],
                "fresh note survives paired panic"
            );
            playing.release_pointer();
            run(false, None, &[physical(true)]);
            assert_eq!(
                notes.held(),
                [1 << 60, 0],
                "same physical pitch works after ownership reset"
            );
            run(false, None, &[physical(false)]);
            assert_eq!(
                notes.held(),
                [0, 0],
                "physical release does not strand a note"
            );
            playing.press(67);
            run(false, None, &[]);
            playing.press(70);
            run(true, None, &[]);
            assert_eq!(
                notes.held(),
                [0, 0],
                "external standalone panic clears local input too"
            );
            adapter.deactivate();
        }
    }

    #[test]
    fn keyboard_source_overlap_overflow_panic_and_reset_clear_correct_notes() {
        let mut p = initialised(false);
        let mut c = hostless(&p);
        c.push_event(note(31, 0, 60, true));
        c.push_event(note(99, 1, 60, true));
        p.local_notes.note(60, true);
        block(&mut p, &mut c, 128);
        assert_eq!(p.local_notes.held(), [1 << 60, 0]);
        p.local_notes.release();
        c.push_event(note(31, 0, 60, false));
        block(&mut p, &mut c, 128);
        assert_eq!(p.local_notes.held(), [1 << 60, 0], "channel 1 remains held");
        c.push_event(note(63, 1, 60, false));
        block(&mut p, &mut c, 128);
        assert_eq!(p.local_notes.held(), [0, 0]);
        p.local_notes.note(67, true);
        block(&mut p, &mut c, 128);
        for _ in 0..local_notes::CAPACITY {
            p.local_notes.note(67, true);
        }
        assert!(!p.local_notes.note(67, false));
        block(&mut p, &mut c, 128);
        assert_eq!(
            p.local_notes.held(),
            [0, 0],
            "overflow released the sounding local note"
        );
        p.local_notes.note(64, true);
        p.local_notes.panic();
        block(&mut p, &mut c, 128);
        assert_eq!(
            p.local_notes.held(),
            [0, 0],
            "panic invalidates old queued attacks"
        );
        p.local_notes.panic();
        p.local_notes.note(69, true);
        block(&mut p, &mut c, 128);
        assert_eq!(
            p.local_notes.held(),
            [0, 1 << 5],
            "a fresh note after panic remains playable"
        );
        p.local_notes.note(60, true);
        block(&mut p, &mut c, 128);
        p.reset();
        assert_eq!(p.local_notes.held(), [0, 0]);
        c.push_event(NoteEvent::NoteOn {
            timing: 3,
            voice_id: None,
            channel: 0,
            note: 60,
            velocity: 0.0,
        });
        block(&mut p, &mut c, 128);
        assert_eq!(p.local_notes.held(), [0, 0], "velocity zero is a release");
        // CA-72's reserved CC120/123 have always been global, irrespective of channel.
        // Keep that compatibility behavior and invalidate local attacks alongside the engine.
        for controller in [ALL_NOTES_OFF, ALL_SOUND_OFF] {
            c.push_event(note(0, 2, 60, true));
            c.push_event(note(0, 3, 67, true));
            p.local_notes.note(64, true);
            block(&mut p, &mut c, 128);
            assert_ne!(p.local_notes.held(), [0, 0]);
            c.push_event(cc(31, 9, controller, 0));
            block(&mut p, &mut c, 128);
            assert_eq!(
                p.local_notes.held(),
                [0, 0],
                "CC{controller} stays a global reset"
            );
        }
    }

    #[test]
    fn the_hosts_midi_reaches_the_wheels_on_any_channel() {
        let mut p = Ca72::default();
        let mut c = hostless(&p);
        p.midi(
            NoteEvent::MidiPitchBend {
                timing: 0,
                channel: 3,
                value: 0.0,
            },
            &mut c,
        );
        p.midi(
            NoteEvent::MidiCC {
                timing: 0,
                channel: 9,
                cc: MODULATION_WHEEL,
                value: 1.0,
            },
            &mut c,
        );
        let (bend, modulation) = p.engine.wheels();
        assert!((bend + 2.0 / PITCH_WHEEL_SEMITONES).abs() < 1e-9, "{bend}");
        assert_eq!(modulation, 1.0);
        p.midi(
            NoteEvent::MidiCC {
                timing: 0,
                channel: 0,
                cc: RESET_CONTROLLERS,
                value: 0.0,
            },
            &mut c,
        );
        assert_eq!(p.engine.wheels(), (0.0, 0.0));
        assert!(c.reported.is_empty());
    }

    /// All Sound Off silences, All Notes Off releases (decisions.md R18).
    #[test]
    fn all_sound_off_and_all_notes_off_are_told_apart() {
        let cc = |cc| {
            event_of(NoteEvent::MidiCC {
                timing: 0,
                channel: 0,
                cc,
                value: 0.0,
            })
        };
        assert_eq!(cc(ALL_SOUND_OFF), Some(Event::AllSoundOff));
        assert_eq!(cc(ALL_NOTES_OFF), Some(Event::AllNotesOff));
        assert_eq!(cc(7), None);
    }

    /// The side chain's channels mixed, a channel shorter than the block (one the host did
    /// not supply may be) and a sample that is not finite taken as 0 (R18).
    #[test]
    fn the_side_chain_is_read_within_its_channels_and_finite() {
        let (mut a, mut b) = ([1.0, f32::NAN, 2.0, f32::INFINITY], [3.0f32]);
        let side: [&mut [f32]; 2] = [&mut a, &mut b];
        let mixed: Vec<f32> = (0..4).map(|i| mono(&side, i)).collect();
        assert_eq!(mixed, [2.0, 0.0, 1.0, 0.0]);
    }

    /// A session 0.1 saved opens translated into 0.2.0's calibration: its EMPHASIS where 0.2.0's
    /// law sets what 0.1's did, LEVEL trimmed and DRIVE at none (decisions.md R-TRANSLATE); one
    /// saved by 0.2.0 opens as it was saved.
    #[test]
    fn a_session_of_0_1_opens_translated_and_one_of_0_2_as_saved() {
        let state = |version: &str| PluginState {
            version: version.to_owned(),
            params: [("emphasis", 5.0f32), ("tune", 0.5)]
                .iter()
                .map(|&(k, v)| (k.to_owned(), ParamValue::F32(v)))
                .collect(),
            fields: Default::default(),
        };
        let f32_of = |s: &PluginState, id: &str| match s.params.get(id) {
            Some(ParamValue::F32(v)) => Some(*v),
            _ => None,
        };
        let mut old = state("0.1.5");
        Ca72::filter_state(&mut old);
        let want = compat::translated("emphasis", 5.0).unwrap() as f32;
        assert_eq!(f32_of(&old, "emphasis"), Some(want));
        assert!((want - 5.0).abs() > 0.1, "EMPHASIS's law changed: {want}");
        assert_eq!(f32_of(&old, "tune"), Some(0.5));
        assert_eq!(f32_of(&old, "level"), Some(compat::LEVEL_FROM_0_1 as f32));
        assert_eq!(f32_of(&old, "drive"), Some(0.0));
        let mut new = state("0.2.0");
        Ca72::filter_state(&mut new);
        assert_eq!(f32_of(&new, "emphasis"), Some(5.0));
        assert_eq!(f32_of(&new, "level"), None);
    }

    /// A session saved before ENTROPY, with ANALOG's amount, opens with ENTROPY at its
    /// default, not at what the instance had (R14, R18); one saved since keeps its own.
    #[test]
    fn an_old_session_opens_with_entropy_at_its_default() {
        let state = |params: &[(&str, f32)]| PluginState {
            version: String::new(),
            params: params
                .iter()
                .map(|&(k, v)| (k.to_owned(), ParamValue::F32(v)))
                .collect(),
            fields: Default::default(),
        };
        let entropy = |s: &PluginState| match s.params.get("entropy") {
            Some(ParamValue::F32(v)) => Some(*v),
            _ => None,
        };
        let mut old = state(&[("analog", 70.0), ("cutoff", 3.0)]);
        Ca72::filter_state(&mut old);
        assert_eq!(entropy(&old), Some(0.0));
        assert!(!old.params.contains_key("analog"));
        let mut new = state(&[("entropy", 40.0)]);
        Ca72::filter_state(&mut new);
        assert_eq!(entropy(&new), Some(40.0));
    }

    /// A session saved before NOTE PRIORITY and RETRIGGER opens with LOW (its index, as nih-plug
    /// sets an enum by one) and RETRIGGER off, the instrument's own, not at what the instance
    /// had (R50); one saved since keeps its own.
    #[test]
    fn an_old_session_opens_with_note_priority_low_and_no_retrigger() {
        let filtered_as = |params: Vec<(String, ParamValue)>, id: &str| {
            let mut s = PluginState {
                version: String::new(),
                params: params.into_iter().collect(),
                fields: Default::default(),
            };
            Ca72::filter_state(&mut s);
            s.params.remove(id)
        };
        let filtered = |params| filtered_as(params, "note_priority");
        assert!(matches!(
            filtered_as(vec![], "retrigger"),
            Some(ParamValue::Bool(false))
        ));
        assert!(matches!(
            filtered_as(
                vec![("retrigger".to_owned(), ParamValue::Bool(true))],
                "retrigger"
            ),
            Some(ParamValue::Bool(true))
        ));
        let low = params::NotePriority::Low.to_index() as i32;
        assert!(matches!(
            filtered(vec![("cutoff".to_owned(), ParamValue::F32(3.0))]),
            Some(ParamValue::I32(i)) if i == low
        ));
        let high = (
            "note_priority".to_owned(),
            ParamValue::String("high".to_owned()),
        );
        assert!(matches!(
            filtered(vec![high]),
            Some(ParamValue::String(s)) if s == "high"
        ));
    }

    /// A session saved with SCATTER's EDGES (no longer offered: R49), or with a placement
    /// not understood, opens with EVEN (its index, as nih-plug sets an enum by one), not at what
    /// the instance had (R18); one saved with EVEN or CENTER keeps its own.
    #[test]
    fn a_session_saved_with_edges_opens_with_even() {
        let filtered = |id: &str| {
            let mut s = PluginState {
                version: String::new(),
                params: [("placement".to_owned(), ParamValue::String(id.to_owned()))].into(),
                fields: Default::default(),
            };
            Ca72::filter_state(&mut s);
            s.params.remove("placement")
        };
        let even = params::Scatter::Even.to_index() as i32;
        for id in ["edges", "sideways"] {
            assert!(
                matches!(filtered(id), Some(ParamValue::I32(i)) if i == even),
                "{id}"
            );
        }
        for id in ["even", "centre"] {
            assert!(
                matches!(filtered(id), Some(ParamValue::String(s)) if s == id),
                "{id}"
            );
        }
        assert_eq!(
            params::Scatter::from_index(even as usize),
            params::Scatter::Even
        );
    }

    /// One block of `len` samples processed as a host calls `process`, with the events pushed to
    /// `c` (stereo, no side chain): its left channel.
    #[allow(unsafe_code)]
    fn block(p: &mut Ca72, c: &mut TestProcessContext<Ca72>, len: usize) -> Vec<f32> {
        let (mut l, mut r) = (vec![0.0f32; len], vec![0.0f32; len]);
        let mut buffer = Buffer::default();
        // SAFETY: both slices are `len` long and live until `process` has returned.
        unsafe {
            buffer.set_slices(len, |s| {
                s.clear();
                s.push(&mut l);
                s.push(&mut r);
            });
        }
        let mut aux = AuxiliaryBuffers {
            inputs: &mut [],
            outputs: &mut [],
        };
        p.process(&mut buffer, &mut aux, c);
        drop(buffer);
        l
    }

    /// A control change as a host gives it (channel 0 to 15, the 7-bit value as 0..1).
    fn cc(timing: u32, channel: u8, cc: u8, value: u8) -> NoteEvent<()> {
        NoteEvent::MidiCC {
            timing,
            channel,
            cc,
            value: f32::from(value) / 127.0,
        }
    }

    fn learnable(id: &str) -> usize {
        learn::index(id).expect("learnable")
    }

    fn on(channel: u8, cc: u8) -> learn::Cc {
        learn::Cc { channel, cc }
    }

    /// A learned controller sets its parameter at its event's sample, through the host (the
    /// host told once, at that time; a value the parameter already has, not again), with no
    /// editor open; the parameter is at the new value at once.
    #[test]
    fn a_learned_controller_sets_its_parameter_at_its_sample_through_the_host() {
        let mut p = initialised(false);
        let mut c = hostless(&p);
        p.params.midi_map.assign(learnable("cutoff"), on(0, 74));
        c.push_event(cc(100, 0, 74, 127));
        c.push_event(cc(200, 0, 74, 127));
        block(&mut p, &mut c, 256);
        assert_eq!(p.params.cutoff.value(), 5.0);
        assert_eq!(c.reported, [(p.params.cutoff.as_ptr(), 1.0, 100)]);
        c.push_event(cc(10, 0, 74, 0));
        block(&mut p, &mut c, 256);
        assert_eq!(p.params.cutoff.value(), -5.0);
        assert_eq!(
            c.reported.last(),
            Some(&(p.params.cutoff.as_ptr(), 0.0, 10))
        );
    }

    /// While learning, the next learnable controller is caught and changes nothing; once the
    /// editor has assigned it, it sets the parameter.
    #[test]
    fn the_controller_caught_while_learning_changes_nothing() {
        let mut p = initialised(false);
        let mut c = hostless(&p);
        let map = Arc::clone(&p.params.midi_map);
        map.arm(learnable("emphasis"));
        c.push_event(cc(0, 3, 20, 127));
        c.push_event(cc(64, 3, 20, 100));
        block(&mut p, &mut c, 128);
        assert_eq!(p.params.emphasis.value(), 0.0, "the sound unchanged");
        assert!(c.reported.is_empty());
        assert_eq!(
            map.poll().map(|a| (a.param, a.cc)),
            Some((learnable("emphasis"), on(3, 20)))
        );
        c.push_event(cc(5, 3, 20, 127));
        block(&mut p, &mut c, 128);
        assert_eq!(p.params.emphasis.value(), 10.0);
    }

    /// The reserved controllers keep what they did (the modulation wheel moves the wheel, reset
    /// all controllers resets it) and are not learned; a controller on another channel than the
    /// one learned does nothing.
    #[test]
    fn the_reserved_controllers_keep_their_paths_and_channels_are_told_apart() {
        let mut p = initialised(false);
        let mut c = hostless(&p);
        let map = Arc::clone(&p.params.midi_map);
        map.assign(learnable("cutoff"), on(0, 74));
        map.arm(learnable("glide"));
        for reserved in [
            0, 1, 6, 32, 38, 96, 97, 98, 99, 100, 101, 122, 124, 125, 126, 127,
        ] {
            c.push_event(cc(0, 0, reserved, 127));
        }
        block(&mut p, &mut c, 64);
        assert_eq!(p.engine.wheels().1, 1.0, "CC 1 is the modulation wheel");
        assert_eq!(
            map.armed(),
            Some(learnable("glide")),
            "nothing reserved was learned"
        );
        assert_eq!(map.refused(), Some(127));
        c.push_event(cc(0, 0, 121, 0));
        block(&mut p, &mut c, 64);
        assert_eq!(p.engine.wheels(), (0.0, 0.0), "CC 121 reset the wheels");
        map.cancel();
        c.push_event(cc(0, 1, 74, 127));
        block(&mut p, &mut c, 64);
        assert_eq!(
            p.params.cutoff.value(),
            0.0,
            "CC 74 on channel 2 is not channel 1's"
        );
        assert!(c.reported.is_empty());
    }

    /// A switch, a selector and VOICES from their controllers' values.
    #[test]
    fn switches_selectors_and_voices_take_their_controllers_values() {
        let mut p = initialised(false);
        let mut c = hostless(&p);
        let map = Arc::clone(&p.params.midi_map);
        map.assign(learnable("osc2_on"), on(0, 20));
        map.assign(learnable("osc1_range"), on(0, 21));
        map.assign(learnable("voices"), on(0, 22));
        c.push_event(cc(0, 0, 20, 64));
        c.push_event(cc(0, 0, 21, 0));
        c.push_event(cc(0, 0, 22, 127));
        block(&mut p, &mut c, 64);
        assert!(p.params.osc2_on.value());
        assert_eq!(p.params.osc1_range.value(), params::Footage::Lo);
        assert_eq!(p.params.voices.value(), 10);
        c.push_event(cc(0, 0, 20, 63));
        c.push_event(cc(0, 0, 21, 127));
        c.push_event(cc(0, 0, 22, 0));
        block(&mut p, &mut c, 64);
        assert!(!p.params.osc2_on.value());
        assert_eq!(p.params.osc1_range.value(), params::Footage::R2);
        assert_eq!(p.params.voices.value(), 2);
    }

    /// A knob a learned controller moves is at its new value at once (the host's, the editor's,
    /// the state's), and the voices glide there over `DEZIP` (10 ms), set every `DEZIP_STEP`
    /// samples; anything else setting it meanwhile (the host's automation) ends the glide there.
    #[test]
    fn a_knob_a_controller_moves_glides_in_the_voices_alone() {
        let mut p = initialised(false);
        let mut c = hostless(&p);
        p.params.midi_map.assign(learnable("cutoff"), on(0, 74));
        let at_rest = p.controls().panel.cutoff;
        assert_eq!(at_rest, p.params.controls().panel.cutoff);
        c.push_event(cc(0, 0, 74, 127));
        block(&mut p, &mut c, 128);
        let voices = p.controls().panel.cutoff;
        let target = p.params.controls().panel.cutoff;
        assert_eq!(target, 1.0, "the parameter at once");
        assert!(
            voices > at_rest && voices < target,
            "{at_rest} {voices} {target}"
        );
        // 480 samples at 48 kHz: there, exactly.
        block(&mut p, &mut c, 352);
        assert_eq!(p.controls(), p.params.controls());
        assert!(!p.dezip.moving());
        // Automation meanwhile: no more glide.
        c.push_event(cc(0, 0, 74, 0));
        block(&mut p, &mut c, 64);
        assert!(p.dezip.moving());
        assert!(c.automate(p.params.cutoff.as_ptr(), 0.75));
        block(&mut p, &mut c, 64);
        assert!(!p.dezip.moving());
        assert_eq!(p.controls(), p.params.controls());
        assert_eq!(p.params.cutoff.value(), 2.5);
        // Switches and selectors do not glide.
        p.params.midi_map.assign(learnable("osc1_range"), on(0, 21));
        c.push_event(cc(0, 0, 21, 127));
        block(&mut p, &mut c, 64);
        assert!(!p.dezip.moving());
        assert_eq!(p.controls(), p.params.controls());
    }

    /// Why a knob a learned controller moves glides (decisions.md R34), measured, by hand
    /// (`cargo test --release -p ca72-plugin --lib a_controllers_steps -- --ignored
    /// --nocapture`): a held A2 on a sawtooth, a knob turned in a quarter of a second three ways:
    /// by the host's automation every 32 samples (the reference), by a 7-bit controller with the
    /// voices following each step at once, and by the same controller as the plug-in plays it
    /// (gliding). The energy above 8 kHz, where the note has little, and the whole, against the
    /// reference's; each knob from and to on its dial, with EMPHASIS where given.
    #[test]
    #[ignore]
    fn a_controllers_steps() {
        use ca72_analysis::fft::{hann, power};
        const RATE: usize = 48_000;
        const STEP: usize = 32;
        let render = |id: &str, from: f64, to: f64, emphasis: f32, how: usize| -> Vec<f32> {
            let mut p = initialised(false);
            let mut c = hostless(&p);
            if how == 1 {
                // (The voices following each step at once: glides one sample long.)
                p.dezip.prepare(1.0 / learn::DEZIP);
            }
            let i = learnable(id);
            p.params.midi_map.assign(i, on(0, 74));
            let params = Arc::clone(&p.params);
            let knob = learn::knob_param(&params, i).expect("a knob");
            let ptr = knob.as_ptr();
            let travel = |v: f64| knob.preview_normalized(v as f32);
            c.automate(params.emphasis.as_ptr(), emphasis / 10.0);
            c.automate(params.contour_amount.as_ptr(), 0.0);
            c.automate(ptr, travel(from));
            c.push_event(NoteEvent::NoteOn {
                timing: 0,
                voice_id: None,
                channel: 0,
                note: 45,
                velocity: 1.0,
            });
            for _ in 0..(RATE / 4 / STEP) {
                block(&mut p, &mut c, STEP);
            }
            let mut out = Vec::new();
            let mut sent = None;
            for k in 0..(RATE / 2 / STEP) {
                let t = ((k * STEP) as f64 / RATE as f64 / 0.25).min(1.0);
                let n = travel(from + (to - from) * t);
                if how == 0 {
                    c.automate(ptr, n);
                } else {
                    let v = (f64::from(n) * 127.0).round() as u8;
                    if sent != Some(v) {
                        c.push_event(cc(0, 0, 74, v));
                        sent = Some(v);
                    }
                }
                out.extend(block(&mut p, &mut c, STEP));
            }
            out
        };
        let band = |x: &[f32]| -> (f64, f64) {
            let n = 2048;
            let w = hann(n);
            let (mut high, mut all) = (0.0, 0.0);
            for frame in x.windows(n).step_by(n / 2) {
                let f: Vec<f64> = frame.iter().map(|&s| f64::from(s)).collect();
                for (k, e) in power(&f, &w).iter().enumerate() {
                    all += e;
                    if k as f64 * RATE as f64 / n as f64 >= 8_000.0 {
                        high += e;
                    }
                }
            }
            (high, all)
        };
        let db = |a: f64, b: f64| 10.0 * (a / b).log10();
        for (id, from, to, emphasis) in [
            ("cutoff", -1.0, 2.0, 7.0),
            ("cutoff", -1.0, 2.0, 9.5),
            ("volume", 10.0, 3.0, 0.0),
            ("osc1_volume", 8.0, 2.0, 0.0),
            ("emphasis", 2.0, 9.0, 0.0),
            ("tune", -1.0, 1.0, 0.0),
        ] {
            let emphasis = if id == "emphasis" {
                from as f32
            } else {
                emphasis
            };
            let smooth = render(id, from, to, emphasis, 0);
            let (hs, a0) = band(&smooth);
            let mut line = format!(
                "{id} {from} to {to} (EMPHASIS {emphasis}): reference above 8 kHz {:.1} dB of its whole",
                db(hs, a0)
            );
            for (name, how) in [("stepped", 1), ("glided", 2)] {
                let (h, a) = band(&render(id, from, to, emphasis, how));
                line += &format!(
                    "; {name} {:+.2} dB above 8 kHz, {:+.2} dB all",
                    db(h, hs),
                    db(a, a0)
                );
            }
            println!("{line}");
        }
    }

    /// With no controller learned, `process` plays as it did before MIDI Learn (decisions.md
    /// R34), to the bit: the code before replayed here on the engine itself (its controls set at
    /// the block's start, runs of at most 128 samples between events, each event at its
    /// sample), with notes, the wheels, controllers not learned and reserved ones, POLY off and
    /// on (rendered offline: every voice waited for).
    #[test]
    fn with_nothing_learned_the_sound_is_as_before() {
        use crate::engine::Engine;
        let _one = BUDGET_TESTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        const LEN: usize = 256;
        let before =
            |e: &mut Engine, controls: &crate::engine::Controls, events: &[NoteEvent<()>]| {
                e.set(controls);
                e.set_deadline(None);
                let (mut l, mut r) = (vec![0.0f32; LEN], vec![0.0f32; LEN]);
                let mut it = events.iter().peekable();
                let mut i = 0;
                while i < LEN {
                    while let Some(ev) = it.peek() {
                        if ev.timing() as usize > i {
                            break;
                        }
                        if let Some(x) = event_of(**ev) {
                            e.event(x);
                        }
                        it.next();
                    }
                    let end = it
                        .peek()
                        .map_or(LEN, |ev| ev.timing() as usize)
                        .min(LEN)
                        .min(i + 128);
                    e.render(&[], &mut l[i..end], &mut r[i..end]);
                    i = end;
                }
                for ev in it {
                    if let Some(x) = event_of(*ev) {
                        e.event(x);
                    }
                }
                e.end_block(LEN);
                l
            };
        for poly in [false, true] {
            let mut p = initialised(poly);
            let params: Arc<dyn Params> = p.params.clone();
            let mut c = TestProcessContext::new(params, 48_000.0, ProcessMode::Offline);
            let controls = p.params.controls();
            let mut e = Engine::new();
            e.set(&controls);
            e.prepare(48_000.0, p.params.seed());
            let mut heard = 0.0;
            for b in 0..120u32 {
                let note = |timing, note: u8, on: bool| {
                    if on {
                        NoteEvent::NoteOn {
                            timing,
                            voice_id: None,
                            channel: 0,
                            note,
                            velocity: 1.0,
                        }
                    } else {
                        NoteEvent::NoteOff {
                            timing,
                            voice_id: None,
                            channel: 0,
                            note,
                            velocity: 0.0,
                        }
                    }
                };
                let mut events = vec![];
                if b % 12 == 0 {
                    events.push(note(3, 45 + (b / 12 % 7) as u8, true));
                    events.push(note(3, 52 + (b / 12 % 5) as u8, true));
                }
                if b % 12 == 9 {
                    events.push(note(17, 45 + (b / 12 % 7) as u8, false));
                }
                events.push(cc(40, (b % 16) as u8, 74, (b * 7 % 128) as u8));
                events.push(cc(41, 0, 1, (b * 5 % 128) as u8));
                events.push(NoteEvent::MidiPitchBend {
                    timing: 90,
                    channel: 0,
                    value: 0.5 + 0.4 * ((b as f32) * 0.3).sin(),
                });
                events.push(cc(200, 0, 7, 100));
                if b % 50 == 49 {
                    events.push(cc(250, 0, 121, 0));
                    events.push(cc(255, 0, 123, 0));
                }
                for ev in &events {
                    c.push_event(*ev);
                }
                let now = block(&mut p, &mut c, LEN);
                let was = before(&mut e, &controls, &events);
                heard += was.iter().map(|x| f64::from(x.abs())).sum::<f64>();
                assert!(
                    now.iter()
                        .zip(&was)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "POLY {poly}, block {b}: not the same samples"
                );
            }
            assert!(c.reported.is_empty());
            assert!(heard > 1.0, "POLY {poly}: silence compared");
        }
    }

    /// The host's automation and a learned controller on one parameter: whichever sets it last,
    /// in time, holds it. At one sample the automation is set first (the wrappers split a block
    /// at an automation point and set it before the run's events: PATCHES.md, change 10), so the
    /// controller's value holds from there until the next automation point.
    #[test]
    fn automation_and_a_controller_on_one_parameter_take_turns_in_time() {
        let mut p = initialised(false);
        let mut c = hostless(&p);
        p.params.midi_map.assign(learnable("emphasis"), on(0, 71));
        // A run from an automation point (set first), with the controller at its first sample.
        c.automate(p.params.emphasis.as_ptr(), 0.2);
        c.push_event(cc(0, 0, 71, 127));
        block(&mut p, &mut c, 32);
        assert_eq!(p.params.emphasis.value(), 10.0);
        // The next automation point, later: it holds.
        c.automate(p.params.emphasis.as_ptr(), 0.3);
        block(&mut p, &mut c, 32);
        assert_eq!(p.params.emphasis.value(), 3.0);
        assert_eq!(p.controls(), p.params.controls());
        // The controller again, later in a run: from its sample.
        c.push_event(cc(16, 0, 71, 0));
        block(&mut p, &mut c, 32);
        assert_eq!(p.params.emphasis.value(), 0.0);
        assert_eq!(c.reported.last().map(|r| r.2), Some(16));
    }

    /// Two instances, each its own assignments: one's controller does nothing to the other.
    #[test]
    fn each_instance_has_its_own_assignments() {
        let (mut a, mut b) = (initialised(false), initialised(false));
        let (mut ca, mut cb) = (hostless(&a), hostless(&b));
        a.params.midi_map.assign(learnable("cutoff"), on(0, 74));
        b.params.midi_map.assign(learnable("emphasis"), on(0, 74));
        for (p, c) in [(&mut a, &mut ca), (&mut b, &mut cb)] {
            c.push_event(cc(0, 0, 74, 127));
            block(p, c, 64);
        }
        assert_eq!(
            (a.params.cutoff.value(), a.params.emphasis.value()),
            (5.0, 0.0)
        );
        assert_eq!(
            (b.params.cutoff.value(), b.params.emphasis.value()),
            (0.0, 10.0)
        );
        b.params.midi_map.arm(learnable("glide"));
        assert_eq!(a.params.midi_map.armed(), None);
    }

    /// The assignments are saved with the session and come back with it; a session saved before
    /// MIDI Learn, loaded into an instance with assignments, leaves it none; one whose table is
    /// not understood, none either. An armed learning is not saved.
    #[test]
    fn a_session_holds_its_assignments_and_an_old_one_none() {
        let state = |fields: BTreeMap<String, String>| PluginState {
            version: String::new(),
            params: Default::default(),
            fields,
        };
        let a = Ca72::default();
        a.params.midi_map.assign(learnable("cutoff"), on(0, 74));
        a.params.midi_map.arm(learnable("glide"));
        let saved = a.params.serialize_fields();
        assert!(
            saved
                .get(learn::STATE_KEY)
                .is_some_and(|t| !t.contains("glide"))
        );
        let b = Ca72::default();
        let mut s = state(saved);
        Ca72::filter_state(&mut s);
        b.params.deserialize_fields(&s.fields);
        assert_eq!(
            b.params.midi_map.assignment(learnable("cutoff")),
            Some(on(0, 74))
        );
        assert_eq!(b.params.midi_map.armed(), None);
        // Older, without one: none.
        let mut old = state(BTreeMap::from([(
            "preset".to_owned(),
            "\"Bass\"".to_owned(),
        )]));
        Ca72::filter_state(&mut old);
        b.params.deserialize_fields(&old.fields);
        assert!(b.params.midi_map.assignments().iter().all(Option::is_none));
        // Not understood: none.
        b.params.midi_map.assign(learnable("cutoff"), on(0, 74));
        let mut bad = state(BTreeMap::from([(
            learn::STATE_KEY.to_owned(),
            "[1, 2".to_owned(),
        )]));
        Ca72::filter_state(&mut bad);
        b.params.deserialize_fields(&bad.fields);
        assert!(b.params.midi_map.assignments().iter().all(Option::is_none));
    }

    /// The plug-in tests that use the process's budget of workers, one at a time.
    static BUDGET_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct Init;
    impl InitContext<Ca72> for Init {
        fn plugin_api(&self) -> PluginApi {
            PluginApi::Clap
        }
        fn execute(&self, _: ()) {}
        fn set_latency_samples(&self, _: u32) {}
        fn set_current_voice_capacity(&self, _: u32) {}
    }

    /// A plug-in, POLY as `poly`, initialised as a host does at 48 kHz in blocks of 256.
    fn initialised(poly: bool) -> Ca72 {
        let config = BufferConfig {
            sample_rate: 48_000.0,
            min_buffer_size: None,
            max_buffer_size: 256,
            process_mode: ProcessMode::Realtime,
        };
        let mut p = Ca72::with_params(Ca72Params {
            poly: BoolParam::new("Poly", poly),
            ..Ca72Params::default()
        });
        assert!(p.initialize(&Ca72::AUDIO_IO_LAYOUTS[0], &config, &mut Init));
        p
    }

    /// Waits up to 10 s for `f` (bounded by what happens, not by time: it fails only if the
    /// helper never gets there).
    fn until(what: &str, mut f: impl FnMut() -> bool) {
        let t = std::time::Instant::now();
        while !f() {
            assert!(t.elapsed() < std::time::Duration::from_secs(10), "{what}");
            std::thread::sleep(std::time::Duration::from_micros(500));
        }
    }

    /// The plug-in's workers through its life, out of the process's budget (decisions.md
    /// R21): initialised with POLY off, it holds none; POLY switched on, its helper (R23),
    /// asked as at the end of a block, starts this machine's share, taken at a block's start;
    /// deactivated, they are back in the budget; initialised with POLY on, it holds them at
    /// once, and gives them back when dropped.
    #[test]
    fn the_workers_follow_poly_through_the_plug_ins_life() {
        let _one = BUDGET_TESTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let held = || crate::pool::budget_held();
        // (A machine of 1 or 2 processors has none to give, one of 3 to 7 one: R11, R39.
        // CI's have 3 or 4.)
        let most = crate::engine::default_workers();
        let mut p = initialised(false);
        assert_eq!((p.engine.workers(), held()), (0, 0));
        // (POLY as `process` reads it at a block's start.)
        let c = crate::engine::Controls {
            poly: true,
            ..p.params.controls()
        };
        p.engine.set(&c);
        p.engine.set_deadline(None);
        p.ask_helper();
        until("no workers from the helper", || {
            p.engine.set_deadline(None);
            p.engine.workers() == most
        });
        assert_eq!(held(), most);
        p.deactivate();
        assert_eq!((p.engine.workers(), held()), (0, 0));
        let q = initialised(true);
        assert_eq!((q.engine.workers(), held()), (most, most));
        drop(q);
        assert_eq!(held(), 0);
    }

    /// The crash R23 fixes: a plug-in dropped (the host's destroy) while its helper is in a
    /// round of mending and serving waits for it, and once the drop returns no thread of the
    /// plug-in's runs: the helper has ended and the voices' workers have stopped, their share
    /// back in the budget, so the host may unload the plug-in.
    #[test]
    fn a_plug_in_dropped_while_its_helper_works_leaves_no_thread_running() {
        let _one = BUDGET_TESTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let held = || crate::pool::budget_held();
        let p = initialised(true);
        assert_eq!(held(), p.engine.workers());
        let watch = p.helper.as_ref().map(Helper::watch).unwrap();
        watch.hold(true);
        if let Some(h) = &p.helper {
            h.ask(MEND | SERVE);
        }
        until("the helper's round never began", || watch.held());
        let (tx, rx) = std::sync::mpsc::channel();
        let w = watch.clone();
        let dropper = std::thread::spawn(move || {
            drop(p);
            tx.send((w.ended(), held())).unwrap();
        });
        // (The drop must not return while the round runs: given a moment to, it has not.)
        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(200))
                .is_err(),
            "the drop returned while the helper's round ran"
        );
        watch.hold(false);
        let (ended, workers) = rx.recv().unwrap();
        assert!(ended, "the drop returned before the helper ended");
        assert_eq!(workers, 0, "the drop returned before the workers stopped");
        dropper.join().unwrap();
    }

    /// The VST3 as a JUCE host plays it (Sandyne, decisions.md R31): set up for blocks of up
    /// to 1920 samples, the side chain deactivated (the host's instrument has no inputs), then
    /// blocks of 480 with that bus's two channels given as null pointers. The blocks play,
    /// every output sample written and finite; 0.1.0 and 0.1.1 read the side chain from
    /// address 0, an access violation.
    #[test]
    #[allow(unsafe_code)]
    fn a_vst3_host_that_deactivates_the_side_chain_has_its_blocks_played() {
        use nih_plug::wrapper::vst3::{Wrapper, vst3_sys};
        use vst3_sys::base::{IPluginBase, IUnknown, kResultOk};
        use vst3_sys::vst::{
            AudioBusBuffers, BusDirections, IAudioProcessor, IComponent, MediaTypes, ProcessData,
            ProcessModes, ProcessSetup, SymbolicSampleSizes,
        };
        let _one = BUDGET_TESTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        const LEN: usize = 480;
        let realtime = ProcessModes::kRealtime as i32;
        let float32 = SymbolicSampleSizes::kSample32 as i32;

        let w = Box::leak(Wrapper::<Ca72>::new());
        let mut side = [std::ptr::null_mut::<f32>(); 2];
        let (mut left, mut right) = (vec![f32::NAN; LEN], vec![f32::NAN; LEN]);
        let mut out = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut inputs = AudioBusBuffers {
            num_channels: 2,
            silence_flags: u64::MAX,
            buffers: side.as_mut_ptr().cast(),
        };
        let mut outputs = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            buffers: out.as_mut_ptr().cast(),
        };
        // SAFETY: the calls a host makes, in its order, on an instance it then releases; the
        // buffers outlive them, and the rest of the block's data is null (no events, no
        // parameter changes, no transport), which the wrapper takes as absent.
        unsafe {
            assert_eq!(w.initialize(std::ptr::null_mut()), kResultOk);
            let setup = ProcessSetup {
                process_mode: realtime,
                symbolic_sample_size: float32,
                max_samples_per_block: 1920,
                sample_rate: 48_000.0,
            };
            assert_eq!(w.setup_processing(&setup), kResultOk);
            // Input bus 0 is the side chain: the plug-in has no main input.
            let (audio, input) = (MediaTypes::kAudio as i32, BusDirections::kInput as i32);
            assert_eq!(w.activate_bus(audio, input, 0, 0), kResultOk);
            assert_eq!(w.set_active(1), kResultOk);
            assert_eq!(w.set_processing(1), kResultOk);

            let mut data: ProcessData = std::mem::zeroed();
            data.process_mode = realtime;
            data.symbolic_sample_size = float32;
            data.num_samples = LEN as i32;
            data.num_inputs = 1;
            data.inputs = &mut inputs;
            data.num_outputs = 1;
            data.outputs = &mut outputs;
            for _ in 0..3 {
                assert_eq!(w.process(&mut data), kResultOk);
            }

            assert_eq!(w.set_processing(0), kResultOk);
            assert_eq!(w.set_active(0), kResultOk);
            assert_eq!(w.terminate(), kResultOk);
            w.release();
        }
        assert!(left.iter().chain(&right).all(|x| x.is_finite()));
    }

    /// AUTO GAIN's curve is the session's (decisions.md R45, the CA-74's R29): saved,
    /// loaded into another instance, and the engine plays with it from the next block; a
    /// session saved before DRIVE, loaded into an instance with a measured curve, leaves it the
    /// average's; one whose curve is not understood, the average's too.
    #[test]
    fn a_session_holds_its_curve_and_an_old_one_the_average() {
        use crate::drive::{AVERAGE, Curve, STATE_KEY, Saved, unmeasured};
        use nih_plug::params::persist::PersistentField;
        let state = |fields: BTreeMap<String, String>| PluginState {
            version: String::new(),
            params: Default::default(),
            fields,
        };
        let measured = Saved {
            sound: 5,
            db: [-1.0, -2.0, -3.0, -4.0],
        };
        let a = Ca72::default();
        a.params.drive_curve.set(measured);
        let saved = a.params.serialize_fields();
        assert_eq!(
            saved.get(STATE_KEY).map(String::as_str),
            Some(r#"{"sound":5,"db":[-1.0,-2.0,-3.0,-4.0]}"#)
        );
        let mut b = Ca72::default();
        let mut s = state(saved);
        Ca72::filter_state(&mut s);
        b.params.deserialize_fields(&s.fields);
        assert_eq!(b.params.drive_curve.saved(), measured);
        let mut c = hostless(&b);
        block(&mut b, &mut c, 64);
        assert_eq!(b.engine.curve(), Curve(measured.db));
        // Older, without one: the average's.
        let mut old = state(BTreeMap::from([(
            "preset".to_owned(),
            "\"Bass\"".to_owned(),
        )]));
        Ca72::filter_state(&mut old);
        b.params.deserialize_fields(&old.fields);
        assert_eq!(b.params.drive_curve.saved(), unmeasured());
        block(&mut b, &mut c, 64);
        assert_eq!(b.engine.curve(), AVERAGE);
        // Not understood: the average's.
        b.params.drive_curve.set(measured);
        let mut bad = state(BTreeMap::from([(STATE_KEY.to_owned(), "[1, 2".to_owned())]));
        Ca72::filter_state(&mut bad);
        b.params.deserialize_fields(&bad.fields);
        assert_eq!(b.params.drive_curve.saved(), unmeasured());
    }
}

#[allow(unsafe_code)]
mod export {
    use super::Ca72;
    use nih_plug::prelude::*;

    nih_export_clap!(Ca72);
    nih_export_vst3!(Ca72);
}
