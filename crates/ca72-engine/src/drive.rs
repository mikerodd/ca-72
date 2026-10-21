//! DRIVE's AUTO GAIN (decisions.md R45, the CA-74's R29): the output brought back down by
//! as much as DRIVE made the sound louder, measured for the sound itself. Off the audio thread,
//! the plug-in's helper plays a few short notes through voices made as the engine's first are,
//! with the sound's panel, without DRIVE and at each of its steps, and compares their loudness
//! (K-weighted, as BS.1770 weighs it): the correction at each step, a curve. The curve is saved
//! with the session, so that the session plays and renders the same again; it is measured again
//! only when the sound is changed in the plug-in's own window (a preset chosen there among the
//! changes), never by the host's automation or a learned controller, so that a render does not
//! depend on when a measurement finished. Until a sound has been measured, the factory
//! presets' average stands in ([`AVERAGE`]). The audio thread only reads the curve, without a
//! lock.
//!
//! The machinery is plugin-kit's (`plugin_kit_stereo::auto_gain`, its K6); what is the
//! instrument's is here: the steps, the notes played, the average, a sound's key, and the
//! voices that measure it ([`Probe`]).

use std::ops::Deref;
use std::sync::Arc;

#[cfg(feature = "nih-plug")]
use nih_plug::params::persist::PersistentField;
use plugin_kit_stereo::auto_gain::{self, Measure};

pub use plugin_kit_stereo::auto_gain::{Curve, KWeighted};

use crate::engine::{Controls, DRIVE_TOP, Probe};

/// The steps of DRIVE the curve is measured at, [`STEP`] dB apart up to [`DRIVE_TOP`]; between
/// them (and from none to the first) it is drawn straight (the CA-74's: measured there at steps
/// half as far apart, its presets' corrections came out no better for twice the work).
pub const STEPS: usize = 4;
pub const STEP: f64 = DRIVE_TOP / STEPS as f64;

/// The notes the measurement plays together, each held [`HELD`] s and heard for [`HEARD`] s:
/// a low, a middle and a high one, as the sound might be played anywhere on the keyboard.
pub const KEYS: [u8; 3] = [36, 55, 72];
pub const HELD: f64 = 0.3;
pub const HEARD: f64 = 0.45;

/// The factory presets' average, measured (`tests/drive_auto.rs`): the correction for a sound
/// not yet measured.
pub const AVERAGE: Curve<STEPS> = Curve([-4.74, -8.43, -10.95, -12.42]);

/// The curve's key in the plug-in's state.
pub const STATE_KEY: &str = "drive_curve";

/// The curve as a session saves it: the sound it was measured for (0: none, the average's) and
/// its corrections.
pub type Saved = auto_gain::Saved<STEPS>;

/// The curve of no sound measured: the average.
pub fn unmeasured() -> Saved {
    Saved {
        sound: 0,
        db: AVERAGE.0,
    }
}

/// The key of a sound as far as DRIVE's loudness goes: the panel, LOCK, ENTROPY (each voice its
/// own parts), FEEDBACK (the voice's output into its own external input) and QUALITY (at LO,
/// the light voices'; at HI keyed as before it, and at ULTRA as at HI: the circuit's model is
/// measured at its real-time setting, which No Compromises differs from far below what the
/// correction can tell, and measuring at No Compromises would take the helper many times as
/// long). Not POLY, UNISON, VOICES, DOUBLE, SPREAD or INNER: a
/// note's voices each have their own filter, so a chord, a unison or a pair sum what one voice
/// does (their levels are their own trims'). Never 0.
pub fn sound(c: &Controls) -> u64 {
    // (FNV-1a over the debug form: every field of the panel, the floats to the bit.)
    let text = if c.potato {
        format!("{:?}", (c.panel, c.lock, c.entropy, c.feedback, "LO"))
    } else {
        format!("{:?}", (c.panel, c.lock, c.entropy, c.feedback))
    };
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h | 1
}

impl Measure for Probe {
    type Sound = Controls;

    fn new(c: &Controls, rate: f64) -> Probe {
        Probe::new(c, rate)
    }

    fn energy(&self, db: f64) -> f64 {
        Probe::energy(self, db)
    }
}

/// The curve, shared by the editor (which asks for a sound to be measured), the plug-in's
/// helper thread (which measures it), the audio thread (which reads it) and the host (which
/// saves and loads it with the session): the kit's, for this instrument's sounds.
#[derive(Debug, Clone)]
pub struct Calibration(Arc<auto_gain::Calibration<Probe, STEPS>>);

impl Default for Calibration {
    fn default() -> Self {
        Calibration(Arc::new(auto_gain::Calibration::new(DRIVE_TOP, AVERAGE)))
    }
}

impl Deref for Calibration {
    type Target = auto_gain::Calibration<Probe, STEPS>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Calibration {
    /// The sound `c` wanted: measured unless the curve is already its, the helper woken. From
    /// the editor, once a change made there is done.
    pub fn ask(&self, c: &Controls) {
        self.0.ask(sound(c), c);
    }

    /// How many hold it (the tests: whether the helper let it go).
    #[cfg(test)]
    pub(crate) fn holders(&self) -> usize {
        Arc::strong_count(&self.0)
    }
}

/// The curve saved and loaded with the session: a session's replaces the instance's (one
/// saved before DRIVE has none, and is given the average by `filter_state`). For nih-plug's
/// `#[persist]` field (`features = ["nih-plug"]`); the LV2 plug-in keeps the curve itself.
#[cfg(feature = "nih-plug")]
impl<'a> PersistentField<'a, Saved> for Calibration {
    fn set(&self, new_value: Saved) {
        self.load(&new_value);
    }

    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&Saved) -> R,
    {
        f(&self.saved())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sound is the panel's, LOCK's, ENTROPY's and FEEDBACK's: asked for once, and not when
    /// the curve is already its; DRIVE, LEVEL, SPREAD and POLY leave it the same sound.
    #[test]
    fn a_sound_is_asked_for_once_and_not_when_it_is_the_curves() {
        let c = Calibration::default();
        let controls = Controls::default();
        c.ask(&controls);
        assert!(c.waiting());
        c.load(&Saved {
            sound: sound(&controls),
            db: [0.0; STEPS],
        });
        let mut other = controls;
        other.panel.cutoff = 0.2;
        assert_ne!(sound(&other), sound(&controls));
        c.ask(&other);
        c.ask(&controls);
        assert!(!c.waiting());
        let mut louder = controls;
        (louder.drive, louder.level, louder.spread, louder.poly) = (12.0, -6.0, 1.0, true);
        louder.inner = 0.5;
        assert_eq!(sound(&louder), sound(&controls));
        let mut fed = controls;
        fed.feedback = 0.5;
        assert_ne!(sound(&fed), sound(&controls), "FEEDBACK is the sound's");
        let ultra = Controls {
            ultra: true,
            ..controls
        };
        assert_eq!(sound(&ultra), sound(&controls), "ULTRA keeps HI's curve");
    }

    /// A sound measured by its voices: the curve is the sound's, every correction a cut. (At the
    /// voices' lowest rate: a debug build's voice is slow.)
    #[test]
    fn a_sound_is_measured_by_its_voices() {
        let c = Calibration::default();
        assert_eq!(c.saved(), unmeasured());
        c.set_rate(crate::engine::MIN_VOICE_RATE);
        let mut other = Controls::default();
        other.panel.cutoff = 0.2;
        c.ask(&other);
        c.measure_now();
        let s = c.saved();
        assert_eq!(s.sound, sound(&other));
        assert!(s.db.iter().all(|d| d.is_finite() && *d < 0.0), "{s:?}");
    }

    /// The steps as the kit draws the curve between them: DRIVE's top over [`STEPS`].
    #[test]
    fn the_curve_is_drawn_between_drives_steps() {
        assert_eq!(STEP, 6.0);
        assert!((AVERAGE.at(STEP, DRIVE_TOP) - f64::from(AVERAGE.0[0])).abs() < 1e-12);
        assert!((AVERAGE.at(DRIVE_TOP, DRIVE_TOP) - f64::from(AVERAGE.0[3])).abs() < 1e-12);
    }
}
