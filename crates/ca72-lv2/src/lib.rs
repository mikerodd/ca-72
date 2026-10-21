//! The CA-72 as an LV2 plug-in: the engine (`ca72-engine`) behind the LV2 C ABI, for hosts
//! without VST3 or CLAP, Zynthian among them.
//!
//! The plug-in is written against `lv2-sys` (rust-lv2's raw bindings) rather than the safe
//! `lv2` framework: its 63 control ports are one table (`params.rs`) that also generates the
//! bundle's `dsp.ttl`, so the code and the TTL cannot drift. Stereo out, an atom-sequence MIDI
//! input, and an optional stereo EXTERNAL INPUT (the side chain; silent while it is not
//! connected). POLY is off and QUALITY is LO by default (a Raspberry Pi 4). The engine's own
//! helper thread and worker pools are used as the nih-plug plug-in uses them.

mod params;
pub mod ttl;

pub use params::{Control, Kind, PortDef, CONTROL_COUNT, PORTS};

use std::ffi::c_void;
use std::hash::{BuildHasher, Hasher};
use std::os::raw::c_char;
use std::slice;
use std::time::{Duration, Instant};

use ca72_engine::drive::Calibration;
use ca72_engine::engine::{default_workers, Engine, Event, CHUNK, DEADLINE};
use ca72_engine::helper::{Helper, MEND, SERVE};
use lv2_sys::{
    LV2_Atom_Event, LV2_Descriptor, LV2_Feature, LV2_Handle, LV2_URID, LV2_URID_Map,
    LV2_ATOM__Sequence, LV2_BUF_SIZE__maxBlockLength, LV2_BUF_SIZE__nominalBlockLength,
    LV2_MIDI__MidiEvent, LV2_URID__map,
};

use params::controls_from;

/// The plug-in's URI (the bundle's `dsp.ttl` names it too).
pub const PLUGIN_URI: &str = "https://github.com/idlefoundry/ca-72/lv2";

/// The samples the engine is run a stretch at, between MIDI events (the plug-in's `RUN`).
const RUN: usize = CHUNK;

/// The buffer length assumed when the host offers no `buf-size` (a Zynthian's block is small;
/// the value only sizes the fallback buffers and the workers' period).
const DEFAULT_MAX_BLOCK: u32 = 1024;

// Port indices, matching the order the bundle's `dsp.ttl` declares (`bin/ttl.rs`).
pub const PORT_IN_L: u32 = 0;
pub const PORT_IN_R: u32 = 1;
pub const PORT_OUT_L: u32 = 2;
pub const PORT_OUT_R: u32 = 3;
pub const PORT_MIDI: u32 = 4;
pub const PORT_CONTROL_BASE: u32 = 5;
/// Every port's index, control ports after the audio and MIDI ones.
pub const PORT_COUNT: u32 = PORT_CONTROL_BASE + CONTROL_COUNT as u32;

/// One plug-in instance: the engine, its helper, and the host's port buffers.
struct Instance {
    engine: Engine,
    helper: Option<Helper>,
    calibration: Calibration,
    rate: f64,
    /// The host's largest block, from `buf-size` (the workers' period).
    max_block: u32,
    /// The host's buffers, by port index (audio, atom and control alike).
    ports: [*mut c_void; PORT_COUNT as usize],
    /// The URID of `atom:Sequence` and of `midi:MidiEvent` (0: unmapped, no MIDI read).
    seq_urid: LV2_URID,
    midi_urid: LV2_URID,
    /// The side chain mixed to mono for one run.
    ext: [f32; RUN],
    /// Rendered into when a host leaves an output unconnected (it must not).
    fallback_l: Vec<f32>,
    fallback_r: Vec<f32>,
}

impl std::fmt::Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            .field("rate", &self.rate)
            .field("max_block", &self.max_block)
            .finish_non_exhaustive()
    }
}

/// A fresh noise/mismatch seed for a new instance (as the nih-plug plug-in's).
fn fresh_seed() -> u64 {
    std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish()
}

impl Instance {
    /// The controls the host's control ports now hold (a port not yet connected reads its
    /// default). Not on the audio thread (only at `activate`).
    fn controls(&self) -> ca72_engine::engine::Controls {
        let mut v = [0.0f32; CONTROL_COUNT];
        for (i, def) in PORTS.iter().enumerate() {
            let ptr = self.ports[PORT_CONTROL_BASE as usize + i] as *const f32;
            // SAFETY: a connected control port points at one `f32` the host owns; a host that
            // has not connected it yet leaves the pointer null, handled by the default.
            v[i] = if ptr.is_null() {
                def.default
            } else {
                // SAFETY: a connected control port points at one `f32` the host owns.
                unsafe { *ptr }
            };
        }
        controls_from(&v)
    }

    /// POLY's workers and the spares' mending, asked of the helper since the last block: as the
    /// nih-plug plug-in's `ask_helper`. On the audio thread: allocates nothing and does not
    /// wait.
    fn ask_helper(&mut self) {
        let mend = if self.engine.take_mending() { MEND } else { 0 };
        let serve = if self.engine.take_serving() { SERVE } else { 0 };
        if let Some(h) = &self.helper {
            h.ask(mend | serve);
        }
    }
}

/// A mapped URI, or 0.
///
/// # Safety
/// `map` must be the host's `LV2_URID_Map`.
unsafe fn map_uri(map: *const LV2_URID_Map, uri: &[u8]) -> LV2_URID {
    // SAFETY: the caller checked `map` against the `urid:map` feature; `uri` ends in a NUL.
    match unsafe { (*map).map } {
        // SAFETY: as above: `map` and `uri` are the host's map and a NUL-terminated URI.
        Some(f) => unsafe { f((*map).handle, uri.as_ptr() as *const c_char) },
        None => 0,
    }
}

/// The host's features: its `urid:map` and its `buf-size` largest block, if offered.
///
/// # Safety
/// `features` is the host's NULL-terminated `LV2_Feature` array, valid to read.
unsafe fn read_features(features: *const *const LV2_Feature) -> (LV2_URID, LV2_URID, u32) {
    let (mut seq, mut midi, mut max_block) = (0, 0, 0u32);
    if features.is_null() {
        return (seq, midi, max_block);
    }
    let mut i = 0isize;
    loop {
        // SAFETY: `features` is NULL-terminated; index `i` is within it.
        let feat = unsafe { *features.offset(i) };
        if feat.is_null() {
            break;
        }
        // SAFETY: a non-null feature is a valid `LV2_Feature` with a NUL-terminated URI.
        let f = unsafe { &*feat };
        let uri = f.URI;
        if !uri.is_null() {
            // SAFETY: the host's feature URIs are NUL-terminated C strings.
            let s = unsafe { std::ffi::CStr::from_ptr(uri) }.to_bytes();
            if s == &LV2_URID__map[..LV2_URID__map.len() - 1] {
                let map = f.data as *const LV2_URID_Map;
                if !map.is_null() {
                    // SAFETY: `map` is the host's map (checked for null just above).
                    seq = unsafe { map_uri(map, LV2_ATOM__Sequence) };
                    // SAFETY: as above.
                    midi = unsafe { map_uri(map, LV2_MIDI__MidiEvent) };
                }
            } else if s == &LV2_BUF_SIZE__maxBlockLength[..LV2_BUF_SIZE__maxBlockLength.len() - 1]
                && !f.data.is_null()
            {
                // SAFETY: the feature data of `buf-size:maxBlockLength` is a `*const u32`.
                max_block = unsafe { *(f.data as *const u32) };
            } else if s
                == &LV2_BUF_SIZE__nominalBlockLength[..LV2_BUF_SIZE__nominalBlockLength.len() - 1]
                && !f.data.is_null()
                && max_block == 0
            {
                // SAFETY: the feature data of `buf-size:nominalBlockLength` is a
                // `*const u32`.
                max_block = unsafe { *(f.data as *const u32) };
            }
        }
        i += 1;
    }
    (seq, midi, max_block)
}

/// `instantiate`.
unsafe extern "C" fn instantiate(
    _descriptor: *const LV2_Descriptor,
    sample_rate: f64,
    _bundle_path: *const c_char,
    features: *const *const LV2_Feature,
) -> LV2_Handle {
    // SAFETY: the host passes its NULL-terminated feature array.
    let (seq_urid, midi_urid, max_block) = unsafe { read_features(features) };
    let max_block = if max_block == 0 {
        DEFAULT_MAX_BLOCK
    } else {
        max_block.clamp(1, 1 << 16)
    };
    let calibration = Calibration::default();
    let instance = Box::new(Instance {
        engine: Engine::new(),
        helper: None,
        calibration,
        rate: sample_rate,
        max_block,
        ports: [std::ptr::null_mut(); PORT_COUNT as usize],
        seq_urid,
        midi_urid,
        ext: [0.0; RUN],
        fallback_l: vec![0.0; max_block as usize],
        fallback_r: vec![0.0; max_block as usize],
    });
    Box::into_raw(instance) as LV2_Handle
}

/// `connect_port`.
unsafe extern "C" fn connect_port(instance: LV2_Handle, port: u32, data: *mut c_void) {
    if instance.is_null() || port >= PORT_COUNT {
        return;
    }
    // SAFETY: the handle is this plug-in's, from `instantiate`, not yet cleaned up.
    let p = unsafe { &mut *(instance as *mut Instance) };
    p.ports[port as usize] = data;
}

/// `activate`: the engine prepared for the host's rate, the helper started, POLY's workers
/// asked for. Not on the audio thread.
unsafe extern "C" fn activate(instance: LV2_Handle) {
    if instance.is_null() {
        return;
    }
    // SAFETY: the handle is this plug-in's, from `instantiate`, not yet cleaned up.
    let p = unsafe { &mut *(instance as *mut Instance) };
    p.engine.set(&p.controls());
    p.engine.prepare(p.rate, fresh_seed());
    p.engine.calibrate_with(p.calibration.clone());
    if p.helper.is_none() {
        p.helper = Helper::start(
            p.engine.spares(),
            p.engine.crew(),
            p.calibration.clone(),
        )
        .ok();
    }
    let period = Duration::from_secs_f64(f64::from(p.max_block) / p.rate.max(1.0));
    p.engine.start_workers(default_workers(), Some(period));
    p.engine.release_all();
    p.engine.event(Event::ResetControllers);
}

/// `deactivate`: POLY's workers stopped and given back (the host initialises again before it
/// next plays). Not on the audio thread.
unsafe extern "C" fn deactivate(instance: LV2_Handle) {
    if instance.is_null() {
        return;
    }
    // SAFETY: the handle is this plug-in's, from `instantiate`, not yet cleaned up.
    let p = unsafe { &mut *(instance as *mut Instance) };
    p.engine.stop_workers();
}

/// `cleanup`: the helper stopped and joined, the instance freed. Not on the audio thread.
unsafe extern "C" fn cleanup(instance: LV2_Handle) {
    if instance.is_null() {
        return;
    }
    // SAFETY: the handle came from `Box::into_raw` in `instantiate`; this is its only cleanup.
    drop(unsafe { Box::from_raw(instance as *mut Instance) });
}

/// `run`: a block of the host's samples, its MIDI events played sample-accurately. On the
/// audio thread.
unsafe extern "C" fn run(instance: LV2_Handle, sample_count: u32) {
    if instance.is_null() || sample_count == 0 {
        return;
    }
    // SAFETY: the handle is this plug-in's, from `instantiate`, not yet cleaned up; the host
    // calls `run` on one thread at a time.
    let p = unsafe { &mut *(instance as *mut Instance) };
    let n = sample_count as usize;

    p.engine.set(&p.controls());
    let period = n as f64 / p.rate.max(1.0);
    let deadline = Instant::now() + Duration::from_secs_f64(period * DEADLINE);
    p.engine.set_deadline(Some(deadline));

    // The host's buffers, detached as raw pointers so that the engine and the scratch (below)
    // can be borrowed while they are written.
    // SAFETY: the ports are this instance's, connected by the host.
    let (lp, ln) = unsafe { audio_out_ptr(p, PORT_OUT_L, n) };
    // SAFETY: as above.
    let (rp, rn) = unsafe { audio_out_ptr(p, PORT_OUT_R, n) };
    // SAFETY: the side chain is optional: a port left unconnected reads none.
    let (il, iln) = unsafe { audio_in_ptr(p, PORT_IN_L, n) };
    // SAFETY: as above.
    let (ir, irn) = unsafe { audio_in_ptr(p, PORT_IN_R, n) };

    let n = n.min(ln).min(rn);
    if n == 0 {
        p.engine.end_block(sample_count as usize);
        p.ask_helper();
        return;
    }
    // SAFETY: each output buffer is `n` `f32`s (the host's, or a fallback of at least `n`).
    let out_l = unsafe { slice::from_raw_parts_mut(lp, n) };
    // SAFETY: as above.
    let out_r = unsafe { slice::from_raw_parts_mut(rp, n) };
    // SAFETY: a connected side-chain channel is `n` `f32`s; an unconnected one is empty.
    let in_l: &[f32] = if iln == 0 {
        &[]
    } else {
        // SAFETY: `il` is connected and valid for `n` reads.
        unsafe { slice::from_raw_parts(il, n) }
    };
    let in_r: &[f32] = if irn == 0 {
        &[]
    } else {
        // SAFETY: `ir` is connected and valid for `n` reads.
        unsafe { slice::from_raw_parts(ir, n) }
    };
    let side = iln > 0 || irn > 0;

    let mut events = Events::new(
        p.ports[PORT_MIDI as usize],
        p.seq_urid,
        p.midi_urid,
    );

    let mut i = 0usize;
    while i < n {
        // Every event at or before this sample, then the next stretch played.
        while let Some(t) = events.peek() {
            if t > i as i64 {
                break;
            }
            if let Some((_, Some(e))) = events.next() {
                p.engine.event(e);
            }
        }
        let mut end = events.peek().map_or(n, |t| (t.max(0) as usize).clamp(i, n));
        end = end.min(i + RUN);
        if end <= i {
            end = (i + RUN).min(n);
        }
        let len = end - i;
        let ext: &[f32] = if side {
            for k in 0..len {
                p.ext[k] = mono(in_l, in_r, i + k);
            }
            &p.ext[..len]
        } else {
            &[]
        };
        p.engine.render(ext, &mut out_l[i..end], &mut out_r[i..end]);
        i = end;
    }
    // Events past the block reach the voices at the next block's start.
    while let Some((_, Some(e))) = events.next() {
        p.engine.event(e);
    }

    p.engine.end_block(n);
    p.ask_helper();
}

/// The output `port`'s buffer and length: the host's, or the instance's fallback when the host
/// left the port null (which it must not).
///
/// # Safety
/// `p` is a live instance; the returned pointer is valid for the returned length.
unsafe fn audio_out_ptr(p: &mut Instance, port: u32, n: usize) -> (*mut f32, usize) {
    let ptr = p.ports[port as usize] as *mut f32;
    if !ptr.is_null() {
        return (ptr, n);
    }
    let fb = if port == PORT_OUT_L {
        &mut p.fallback_l
    } else {
        &mut p.fallback_r
    };
    let m = n.min(fb.len());
    (fb.as_mut_ptr(), m)
}

/// The input `port`'s buffer and length: null and 0 when the host left it unconnected (the
/// side chain is optional).
///
/// # Safety
/// `p` is a live instance; a non-null pointer is valid for `n` reads.
unsafe fn audio_in_ptr(p: &Instance, port: u32, n: usize) -> (*const f32, usize) {
    let ptr = p.ports[port as usize] as *const f32;
    if ptr.is_null() {
        (std::ptr::null(), 0)
    } else {
        (ptr, n)
    }
}

/// The side chain at sample `i`, its two channels mixed to mono, a non-finite sample as 0.
fn mono(l: &[f32], r: &[f32], i: usize) -> f32 {
    let a = l.get(i).copied().filter(|x| x.is_finite()).unwrap_or(0.0);
    let b = r.get(i).copied().filter(|x| x.is_finite()).unwrap_or(0.0);
    match (l.is_empty(), r.is_empty()) {
        (true, true) => 0.0,
        (true, false) => b,
        (false, true) => a,
        (false, false) => 0.5 * (a + b),
    }
}

/// The host's MIDI as the engine takes it (any channel), if it takes it — the nih-plug
/// plug-in's `event_of`, without MIDI Learn.
fn event_of(bytes: &[u8]) -> Option<Event> {
    let status = *bytes.first()? & 0xf0;
    let data = |k: usize| bytes.get(k).copied().unwrap_or(0) as f32;
    match status {
        0x80 => Some(Event::Note {
            key: bytes.get(1).copied()?,
            on: false,
        }),
        0x90 => Some(Event::Note {
            key: bytes.get(1).copied()?,
            on: data(2) > 0.0,
        }),
        0xe0 => {
            let raw = ((data(2) as u16) << 7) | data(1) as u16;
            let v = f32::from(raw) / 16383.0;
            Some(Event::PitchBend(v * 2.0 - 1.0))
        }
        0xb0 => {
            let cc = bytes.get(1).copied()?;
            let value = data(2) / 127.0;
            match cc {
                1 => Some(Event::Modulation(value)),
                120 => Some(Event::AllSoundOff),
                121 => Some(Event::ResetControllers),
                123 => Some(Event::AllNotesOff),
                _ => None,
            }
        }
        _ => None,
    }
}

/// An iterator over an `atom:Sequence` port's events, by sample frame.
struct Events {
    base: *const u8,
    end: usize,
    off: usize,
    midi_urid: LV2_URID,
}

impl Events {
    /// The events of the sequence at `port`, or none if it is not one (the host maps
    /// `atom:Sequence`, and the block's atom must carry its URID).
    fn new(port: *mut c_void, seq_urid: LV2_URID, midi_urid: LV2_URID) -> Events {
        let empty = Events {
            base: std::ptr::null(),
            end: 0,
            off: 0,
            midi_urid,
        };
        if port.is_null() || seq_urid == 0 {
            return empty;
        }
        // SAFETY: a connected atom port points at an `LV2_Atom_Sequence`.
        let seq = unsafe { &*(port as *const lv2_sys::LV2_Atom_Sequence) };
        if seq.atom.type_ != seq_urid {
            return empty;
        }
        // Events follow the 8-byte atom header and the 8-byte sequence body, for
        // `atom.size - 8` bytes.
        let total = (seq.atom.size as usize).saturating_add(8);
        Events {
            // SAFETY: the sequence buffer is aligned; events start 16 bytes in.
            base: unsafe { (port as *const u8).add(16) },
            end: total,
            off: 0,
            midi_urid,
        }
    }

    /// The next event's sample frame, without consuming it.
    fn peek(&self) -> Option<i64> {
        if self.base.is_null() || self.off + 16 > self.end {
            return None;
        }
        // SAFETY: the event header is 8 bytes of time and an 8-byte atom header, within `end`.
        Some(unsafe { (*(self.base.add(self.off) as *const LV2_Atom_Event)).time.frames })
    }

    /// The next event: its sample frame and, when it is MIDI this engine takes, the event.
    fn next(&mut self) -> Option<(i64, Option<Event>)> {
        if self.base.is_null() || self.off + 16 > self.end {
            return None;
        }
        // SAFETY: as `peek`; the body atom's size bounds the data read.
        let (time, atom, data) = unsafe {
            let ev = self.base.add(self.off) as *const LV2_Atom_Event;
            let atom = (*ev).body;
            let data = self.base.add(self.off + 16);
            ((*ev).time.frames, atom, data)
        };
        let event = if atom.type_ == self.midi_urid && atom.size >= 1 {
            // SAFETY: `atom.size` bytes of MIDI follow the header.
            Some(event_of(unsafe {
                slice::from_raw_parts(data, atom.size as usize)
            }))
        } else {
            None
        }
        .flatten();
        let padded = (atom.size as usize).saturating_add(7) & !7;
        self.off = self.off.saturating_add(16).saturating_add(padded);
        Some((time, event))
    }
}

// The one plug-in. `LV2_Descriptor` holds raw pointers, so it is not `Sync` on its own: this
// wrapper is (the descriptor is immutable and `run` is called on one thread at a time).
struct Descriptor(LV2_Descriptor);

// SAFETY: the descriptor is immutable, its URI a 'static C string, and the LV2 host guarantees
// `run` is called on one thread at a time.
unsafe impl Sync for Descriptor {}
// SAFETY: as `Sync`.
unsafe impl Send for Descriptor {}

static DESCRIPTOR: Descriptor = Descriptor(LV2_Descriptor {
    URI: c"https://github.com/idlefoundry/ca-72/lv2".as_ptr(),
    instantiate: Some(instantiate),
    connect_port: Some(connect_port),
    activate: Some(activate),
    run: Some(run),
    deactivate: Some(deactivate),
    cleanup: Some(cleanup),
    extension_data: None,
});

/// The bundle's entry point (LV2's old discovery API): this plug-in at index 0.
#[unsafe(no_mangle)]
pub extern "C" fn lv2_descriptor(index: u32) -> *const LV2_Descriptor {
    if index == 0 {
        &DESCRIPTOR.0
    } else {
        std::ptr::null()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A note on and its note off through the raw MIDI parser.
    #[test]
    fn midi_notes_parse() {
        assert!(matches!(
            event_of(&[0x90, 60, 100]),
            Some(Event::Note { key: 60, on: true })
        ));
        assert!(matches!(
            event_of(&[0x90, 60, 0]),
            Some(Event::Note { key: 60, on: false })
        ));
        assert!(matches!(
            event_of(&[0x80, 60, 0]),
            Some(Event::Note { key: 60, on: false })
        ));
    }

    /// The wheels, the reserved controllers and the bend's centre.
    #[test]
    fn midi_controllers_parse() {
        assert!(matches!(event_of(&[0xb0, 1, 127]), Some(Event::Modulation(v)) if (v - 1.0).abs() < 1e-6));
        assert!(matches!(event_of(&[0xb0, 120, 0]), Some(Event::AllSoundOff)));
        assert!(matches!(event_of(&[0xb0, 121, 0]), Some(Event::ResetControllers)));
        assert!(matches!(event_of(&[0xb0, 123, 0]), Some(Event::AllNotesOff)));
        assert!(event_of(&[0xb0, 7, 64]).is_none());
        // Bend's centre (0x2000) is 0; full up is 1.
        assert!(matches!(event_of(&[0xe0, 0, 64]), Some(Event::PitchBend(v)) if v.abs() < 1e-4));
        assert!(matches!(event_of(&[0xe0, 127, 127]), Some(Event::PitchBend(v)) if (v - 1.0).abs() < 1e-4));
    }
}

/// A stand-in host: it drives the descriptor's callbacks as a host would, so the whole ABI —
/// instantiate, the atom-sequence MIDI in, the control ports, run and cleanup — is exercised
/// without an LV2 host on the machine.
#[cfg(test)]
mod host {
    use super::*;
    use std::ffi::CStr;

    /// The host's URID map: `atom:Sequence` is 1, `midi:MidiEvent` 2, everything else 3.
    unsafe extern "C" fn map_test(_handle: *mut c_void, uri: *const c_char) -> LV2_URID {
        // SAFETY: the plug-in passes a NUL-terminated URI from a static constant.
        let s = unsafe { CStr::from_ptr(uri) }.to_bytes();
        if s == &LV2_ATOM__Sequence[..LV2_ATOM__Sequence.len() - 1] {
            1
        } else if s == &LV2_MIDI__MidiEvent[..LV2_MIDI__MidiEvent.len() - 1] {
            2
        } else {
            3
        }
    }

    /// An `atom:Sequence` of the given MIDI messages, each at a sample frame.
    fn sequence(events: &[(i64, [u8; 3])]) -> Vec<u8> {
        let mut buf = vec![0u8; 16 + 32 * events.len()];
        // SAFETY: the buffer is 16 bytes at least, aligned for the atom header.
        unsafe {
            let atom = buf.as_mut_ptr() as *mut lv2_sys::LV2_Atom;
            (*atom).size = (buf.len() - 8) as u32;
            (*atom).type_ = 1;
            for (k, (time, midi)) in events.iter().enumerate() {
                let off = 16 + 32 * k;
                let ev = buf.as_mut_ptr().add(off) as *mut LV2_Atom_Event;
                (*ev).time.frames = *time;
                (*ev).body.size = 3;
                (*ev).body.type_ = 2;
                std::ptr::copy_nonoverlapping(
                    midi.as_ptr(),
                    buf.as_mut_ptr().add(off + 16),
                    3,
                );
            }
        }
        buf
    }

    /// The descriptor names the one plug-in, with the URI the bundle's TTL and `PLUGIN_URI`
    /// carry, and no second.
    #[test]
    fn the_descriptor_names_the_uri() {
        let desc = super::lv2_descriptor(0);
        assert!(!desc.is_null());
        // SAFETY: the descriptor's URI is a 'static NUL-terminated C string.
        let uri = unsafe { CStr::from_ptr((*desc).URI) };
        assert_eq!(uri.to_str().unwrap(), PLUGIN_URI);
        assert!(super::lv2_descriptor(1).is_null());
    }

    /// A note through the whole ABI makes sound on the output ports.
    #[test]
    fn a_note_renders_through_the_abi() {
        let map = LV2_URID_Map {
            handle: std::ptr::null_mut(),
            map: Some(map_test),
        };
        let feature = LV2_Feature {
            URI: LV2_URID__map.as_ptr() as *const c_char,
            data: &map as *const _ as *mut c_void,
        };
        let features = [&feature as *const LV2_Feature, std::ptr::null()];

        let desc = &DESCRIPTOR.0;
        let n = 256usize;
        let mut out_l = vec![0.0f32; n];
        let mut out_r = vec![0.0f32; n];
        let mut midi = sequence(&[(0, [0x90, 60, 100])]);
        let mut controls: Vec<f32> = PORTS.iter().map(|p| p.default).collect();

        // SAFETY: every callback is called as the LV2 host would, with this instance.
        unsafe {
            let inst = (desc.instantiate.unwrap())(desc, 48_000.0, std::ptr::null(), features.as_ptr());
            assert!(!inst.is_null());
            let connect = desc.connect_port.unwrap();
            connect(inst, PORT_OUT_L, out_l.as_mut_ptr() as *mut c_void);
            connect(inst, PORT_OUT_R, out_r.as_mut_ptr() as *mut c_void);
            connect(inst, PORT_MIDI, midi.as_mut_ptr() as *mut c_void);
            for i in 0..CONTROL_COUNT {
                connect(
                    inst,
                    PORT_CONTROL_BASE + i as u32,
                    controls.as_mut_ptr().add(i) as *mut c_void,
                );
            }
            (desc.activate.unwrap())(inst);
            let run = desc.run.unwrap();
            let mut peak = 0.0f32;
            for block in 0..16 {
                run(inst, n as u32);
                // Only the first block carries the note; the rest are empty.
                if block == 0 {
                    midi[0..4].copy_from_slice(&8u32.to_ne_bytes());
                }
                peak = peak.max(out_l.iter().fold(0.0f32, |a, x| a.max(x.abs())));
                assert!(out_l.iter().all(|x| x.is_finite()), "block {block} is not finite");
            }
            assert!(peak > 1e-4, "the note made no sound (peak {peak})");
            (desc.deactivate.unwrap())(inst);
            (desc.cleanup.unwrap())(inst);
        }
    }
}
