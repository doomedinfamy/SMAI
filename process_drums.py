from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import soundfile as sf
    import librosa
    from scipy.signal import butter, sosfilt, lfilter
    import pandas as pd
except ImportError as e:
    sys.exit(f"Missing dependency ({e}). "
             f"pip install numpy scipy soundfile librosa pandas")


DRUM_COLS = ["gain_drums", "eq_drums_low", "eq_drums_mid",
             "eq_drums_high", "comp_drums", "pan_drums"]

LOW_HZ, MID_HZ = 250.0, 4000.0


# --------------------------------------------------------------------------- #
# I/O + drum isolation
# --------------------------------------------------------------------------- #
def load_audio(path, sr=44100):
    y, sr = librosa.load(path, sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])          # -> (2, n) stereo
    return y, sr


def isolate_drums(path, sr=44100):
    """Return a stereo drum signal (2, n)."""
    try:
        import demucs  # noqa: F401
        return _demucs_drums(path, sr)
    except Exception:
        print("demucs not available - using percussive approximation.")
        y, sr = load_audio(path, sr)
        perc = np.stack([librosa.effects.percussive(y[0]),
                         librosa.effects.percussive(y[1])])
        return perc, sr


def _demucs_drums(path, sr):
    import subprocess, tempfile, glob
    outdir = tempfile.mkdtemp(prefix="demucs_")
    print("Separating drums with demucs...")
    subprocess.run([sys.executable, "-m", "demucs", "--two-stems=drums",
                    "-o", outdir, path], check=True)
    base = os.path.splitext(os.path.basename(path))[0]
    hits = glob.glob(os.path.join(outdir, "*", base, "drums.wav"))
    if not hits:
        raise RuntimeError("demucs produced no drums stem")
    y, _ = librosa.load(hits[0], sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y, sr


# --------------------------------------------------------------------------- #
# Envelope helper
# --------------------------------------------------------------------------- #
def envelope(x, sr, attack_ms, release_ms):
    """One-pole attack/release envelope follower on |x|."""
    a_att = np.exp(-1.0 / (sr * attack_ms / 1000.0 + 1e-9))
    a_rel = np.exp(-1.0 / (sr * release_ms / 1000.0 + 1e-9))
    env = np.zeros_like(x)
    prev = 0.0
    ax = np.abs(x)
    for i in range(len(x)):
        coeff = a_att if ax[i] > prev else a_rel
        prev = coeff * prev + (1.0 - coeff) * ax[i]
        env[i] = prev
    return env


# --------------------------------------------------------------------------- #
# 1. Aggressive noise gate
# --------------------------------------------------------------------------- #
def noise_gate(x, sr, threshold_db=-45.0, attack_ms=1.0, hold_ms=15.0,
               release_ms=60.0, floor_db=-80.0):
    """Hard, fast gate. Below threshold the signal is attenuated to floor."""
    thr = 10 ** (threshold_db / 20.0)
    floor = 10 ** (floor_db / 20.0)
    env = envelope(x, sr, attack_ms=0.5, release_ms=5.0)
    open_mask = env > thr

    hold = int(sr * hold_ms / 1000.0)
    gain = np.full(len(x), floor)
    count = 0
    for i in range(len(x)):
        if open_mask[i]:
            count = hold
        if count > 0:
            gain[i] = 1.0
            count -= 1
    # Smooth the gate gain so it doesn't click.
    a_att = np.exp(-1.0 / (sr * attack_ms / 1000.0 + 1e-9))
    a_rel = np.exp(-1.0 / (sr * release_ms / 1000.0 + 1e-9))
    g = np.zeros_like(gain)
    prev = floor
    for i in range(len(gain)):
        coeff = a_att if gain[i] > prev else a_rel
        prev = coeff * prev + (1.0 - coeff) * gain[i]
        g[i] = prev
    return x * g


# --------------------------------------------------------------------------- #
# 2. Transient shaper
# --------------------------------------------------------------------------- #
def transient_shaper(x, sr, attack=1.6, sustain=0.7):
    """
    Boost/cut attack and sustain. attack/sustain are multipliers:
      attack  > 1 adds punch, < 1 softens hits.
      sustain > 1 adds body,  < 1 dries/tightens.
    """
    fast = envelope(x, sr, attack_ms=1.0, release_ms=25.0)
    slow = envelope(x, sr, attack_ms=15.0, release_ms=180.0)
    diff = fast - slow                      # >0 during attack transient
    trans = np.clip(diff, 0, None)
    sust = np.clip(slow, 0, None)
    denom = fast + 1e-9
    gain = (attack * trans + sustain * sust) / denom
    gain = np.clip(gain, 0.0, 4.0)
    return x * gain


# --------------------------------------------------------------------------- #
# 3. Volume shaper / LFO ducking
# --------------------------------------------------------------------------- #
def volume_shaper_duck(x, sr, tempo=120.0, depth=0.5, shape="pump",
                       subdiv=1.0):
    """
    Tempo-synced gain LFO. `depth` 0..1 = how deep the duck is.
    shape='pump' gives the sidechain saw (duck then recover each beat);
    shape='sine' gives a smooth tremolo.
    subdiv scales the rate (1=quarter notes, 2=eighths, 0.5=half notes).
    """
    beat_sec = 60.0 / max(tempo, 1e-6) / max(subdiv, 1e-6)
    n = x.shape[-1]
    t = np.arange(n) / sr
    phase = (t % beat_sec) / beat_sec        # 0..1 within each beat
    if shape == "sine":
        lfo = 0.5 * (1 + np.cos(2 * np.pi * phase))     # 1 at beat, dips mid
    else:  # pump: 0 at beat start (ducked), rises back to 1
        lfo = phase ** 0.6
    gain = 1.0 - depth * (1.0 - lfo)
    return x * gain


# --------------------------------------------------------------------------- #
# 4. Correction application (level / EQ / comp / pan)
# --------------------------------------------------------------------------- #
def apply_gain_db(x, db):
    return x * (10 ** (db / 20.0))


def three_band_eq(x, sr, low_db, mid_db, high_db):
    """Simple 3-band EQ via crossover filters, boosts/cuts in dB."""
    sos_low = butter(2, LOW_HZ / (sr / 2), btype="low", output="sos")
    sos_high = butter(2, MID_HZ / (sr / 2), btype="high", output="sos")
    sos_mid = butter(2, [LOW_HZ / (sr / 2), MID_HZ / (sr / 2)],
                     btype="band", output="sos")
    low = sosfilt(sos_low, x)
    mid = sosfilt(sos_mid, x)
    high = sosfilt(sos_high, x)
    return (low * 10 ** (low_db / 20.0)
            + mid * 10 ** (mid_db / 20.0)
            + high * 10 ** (high_db / 20.0))


def compressor(x, sr, amount, ratio_max=8.0, attack_ms=5.0, release_ms=80.0):
    """`amount` 0..~10 (dataset comp scale) -> soft-knee downward compression."""
    if amount <= 0:
        return x
    ratio = 1.0 + (ratio_max - 1.0) * min(amount / 10.0, 1.0)
    env = envelope(x, sr, attack_ms, release_ms)
    env_db = 20 * np.log10(env + 1e-9)
    thr_db = -24.0
    over = np.clip(env_db - thr_db, 0, None)
    gain_db = -over * (1 - 1 / ratio)
    makeup = 10 ** ((amount * 0.6) / 20.0)      # crude makeup gain
    return x * 10 ** (gain_db / 20.0) * makeup


def apply_pan(stereo, pan):
    """Equal-power pan. pan -1=L .. +1=R applied to a stereo (2,n) signal."""
    angle = (pan + 1) * 0.25 * np.pi          # 0..pi/2
    l_gain, r_gain = np.cos(angle), np.sin(angle)
    mono = stereo.mean(axis=0)
    return np.stack([mono * l_gain * np.sqrt(2),
                     mono * r_gain * np.sqrt(2)])


# --------------------------------------------------------------------------- #
# Correction loading
# --------------------------------------------------------------------------- #
def load_corrections(path, song_id):
    df = pd.read_csv(path)
    if "song_id" in df.columns and song_id in df["song_id"].values:
        row = df.loc[df["song_id"] == song_id].iloc[0]
    else:
        row = df.iloc[song_id]
    return {c: float(row[c]) for c in DRUM_COLS if c in df.columns}


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def process(drums, sr, corr, tempo, gate_db, duck_depth, duck_shape,
            attack, sustain, subdiv):
    out = np.zeros_like(drums)
    # Per-channel: gate -> transient shape -> LFO duck -> gain -> EQ -> comp
    for ch in range(drums.shape[0]):
        x = drums[ch]
        x = noise_gate(x, sr, threshold_db=gate_db)
        x = transient_shaper(x, sr, attack=attack, sustain=sustain)
        x = volume_shaper_duck(x, sr, tempo=tempo, depth=duck_depth,
                               shape=duck_shape, subdiv=subdiv)
        x = apply_gain_db(x, corr.get("gain_drums", 0.0))
        x = three_band_eq(x, sr,
                          corr.get("eq_drums_low", 0.0),
                          corr.get("eq_drums_mid", 0.0),
                          corr.get("eq_drums_high", 0.0))
        x = compressor(x, sr, corr.get("comp_drums", 0.0))
        out[ch] = x
    # Pan across the stereo field.
    out = apply_pan(out, corr.get("pan_drums", 0.0))
    # Guard against clipping.
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.0:
        out = out / peak * 0.98
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Apply gate/transient/LFO-duck + corrections to drums")
    ap.add_argument("audio", help="drum stem, or full song with --from-song")
    ap.add_argument("--from-song", action="store_true",
                    help="separate drums from a full mix first")
    ap.add_argument("--corrections",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "corrected_songs.csv"),
                    help="corrections CSV from apply_corrections.py")
    ap.add_argument("--song-id", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tempo", type=float, default=120.0)
    ap.add_argument("--subdiv", type=float, default=1.0,
                    help="LFO rate: 1=quarter, 2=eighth, 0.5=half")
    ap.add_argument("--gate-db", type=float, default=-45.0)
    ap.add_argument("--duck-depth", type=float, default=0.5)
    ap.add_argument("--duck-shape", default="pump", choices=["pump", "sine"])
    ap.add_argument("--attack", type=float, default=1.6,
                    help="transient attack multiplier (>1 = punchier)")
    ap.add_argument("--sustain", type=float, default=0.7,
                    help="transient sustain multiplier (<1 = tighter)")
    args = ap.parse_args()

    if not os.path.exists(args.audio):
        sys.exit(f"Audio not found: {args.audio}")

    # Correction values (fall back to neutral if file missing).
    if os.path.exists(args.corrections):
        corr = load_corrections(args.corrections, args.song_id)
        print(f"Loaded drum corrections (song {args.song_id}): {corr}")
    else:
        corr = {c: 0.0 for c in DRUM_COLS}
        print("No corrections file - applying techniques with neutral levels.")

    # Get drum signal.
    if args.from_song:
        drums, sr = isolate_drums(args.audio)
    else:
        drums, sr = load_audio(args.audio)

    print(f"Processing drums: tempo={args.tempo} gate={args.gate_db}dB "
          f"duck={args.duck_depth} ({args.duck_shape}) "
          f"transient a={args.attack}/s={args.sustain}")

    out = process(drums, sr, corr, args.tempo, args.gate_db, args.duck_depth,
                  args.duck_shape, args.attack, args.sustain, args.subdiv)

    out_path = args.out or (os.path.splitext(args.audio)[0] + "_drums_processed.wav")
    sf.write(out_path, out.T, sr)
    print(f"Wrote processed drums -> {out_path}")


if __name__ == "__main__":
    main()
