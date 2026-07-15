from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Optional, better loudness (LUFS) if available
try:
    import pyloudnorm as pyln

    _HAVE_PYLN = True
except Exception:
    _HAVE_PYLN = False

STEMS = ["vocals", "drums", "bass", "other"]
SR = 44100  # analysis sample rate


# --------------------------------------------------------------------------- #
# 1. Stem separation
# --------------------------------------------------------------------------- #
def separate(song: Path, out_root: Path, model: str) -> dict[str, Path]:
    """Run Demucs and return {stem_name: wav_path}. Skips if already done."""
    track = song.stem
    stem_dir = out_root / model / track

    if all((stem_dir / f"{s}.wav").exists() for s in STEMS):
        print(f"  stems already present for '{track}', skipping Demucs")
    else:
        print(f"  running Demucs ({model}) on '{song.name}' ...")
        cmd = [
            sys.executable, "-m", "demucs",
            "-n", model,
            "-o", str(out_root),
            str(song),
        ]
        subprocess.run(cmd, check=True)

    paths = {s: stem_dir / f"{s}.wav" for s in STEMS}
    missing = [s for s, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Demucs did not produce these stems: {missing} (looked in {stem_dir})"
        )
    return paths


# --------------------------------------------------------------------------- #
# 2. Per-stem measurements
# --------------------------------------------------------------------------- #
def loudness_lufs(y: np.ndarray, sr: int) -> float:
    """Integrated loudness in LUFS (pyloudnorm) or dBFS RMS fallback."""
    if _HAVE_PYLN:
        meter = pyln.Meter(sr)
        try:
            return float(meter.integrated_loudness(y))
        except Exception:
            pass
    rms = np.sqrt(np.mean(y**2)) + 1e-12
    return float(20 * np.log10(rms))  # dBFS


def spectral_color(y: np.ndarray, sr: int) -> tuple[float, float, float]:
    """
    Summarise energy distribution as an RGB colour.
    Split the spectrum into low / mid / high bands and use each band's
    energy share as R / G / B.
    """
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band_energy = S.sum(axis=1)  # energy per frequency bin

    low = band_energy[freqs < 250].sum()
    mid = band_energy[(freqs >= 250) & (freqs < 4000)].sum()
    high = band_energy[freqs >= 4000].sum()
    total = low + mid + high + 1e-12

    return (low / total, mid / total, high / total)


def analyse_stem(path: Path) -> dict:
    y, sr = librosa.load(path, sr=SR, mono=True)
    energy = float(np.sum(y**2))
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr // 2),
        ref=np.max,
    )
    return {
        "y": y,
        "sr": sr,
        "energy": energy,
        "loudness": loudness_lufs(y, sr),
        "peak": float(np.max(np.abs(y)) + 1e-12),
        "color": spectral_color(y, sr),
        "centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        "mel_db": mel,
    }


# --------------------------------------------------------------------------- #
# 3. Combined output
# --------------------------------------------------------------------------- #
def render(track: str, data: dict[str, dict], out_png: Path, out_csv: Path):
    total_energy = sum(d["energy"] for d in data.values()) + 1e-12
    for d in data.values():
        d["loud_pct"] = 100.0 * d["energy"] / total_energy

    # ---- one combined figure -------------------------------------------- #
    fig = plt.figure(figsize=(14, 4 * len(STEMS)))
    gs = GridSpec(len(STEMS), 3, width_ratios=[5, 1, 1], figure=fig, hspace=0.5)
    fig.suptitle(f"Stem mixing analysis — {track}", fontsize=16, y=0.995)

    for row, s in enumerate(STEMS):
        d = data[s]

        # spectrogram
        ax = fig.add_subplot(gs[row, 0])
        librosa.display.specshow(
            d["mel_db"], sr=d["sr"], x_axis="time", y_axis="mel",
            fmax=d["sr"] // 2, ax=ax, cmap="magma",
        )
        ax.set_title(f"{s}  —  spectrogram", loc="left", fontsize=11)

        # loudness % bar
        axb = fig.add_subplot(gs[row, 1])
        axb.bar([0], [d["loud_pct"]], color=(0.2, 0.5, 0.9))
        axb.set_ylim(0, 100)
        axb.set_xticks([])
        axb.set_title("loudness %", fontsize=10)
        axb.text(0, d["loud_pct"] + 2, f"{d['loud_pct']:.1f}%",
                 ha="center", fontsize=10)

        # spectral colour swatch
        axc = fig.add_subplot(gs[row, 2])
        axc.imshow([[d["color"]]], aspect="auto")
        axc.set_xticks([]); axc.set_yticks([])
        r, g, b = d["color"]
        axc.set_title("spectral color", fontsize=10)
        axc.set_xlabel(f"low {r*100:.0f}%\nmid {g*100:.0f}%\nhigh {b*100:.0f}%",
                       fontsize=8)

    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png.name}")

    # ---- raw numbers ---------------------------------------------------- #
    import csv

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "loudness_pct", "loudness_lufs_or_dbfs",
                    "peak", "spectral_centroid_hz",
                    "color_low", "color_mid", "color_high"])
        for s in STEMS:
            d = data[s]
            r, g, b = d["color"]
            w.writerow([s, round(d["loud_pct"], 2), round(d["loudness"], 2),
                        round(d["peak"], 4), round(d["centroid"], 1),
                        round(r, 3), round(g, 3), round(b, 3)])
    print(f"  wrote {out_csv.name}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def process(song: Path, sep_root: Path, out_dir: Path, model: str):
    print(f"\n=== {song.name} ===")
    stem_paths = separate(song, sep_root, model)
    data = {s: analyse_stem(p) for s, p in stem_paths.items()}
    render(
        song.stem, data,
        out_dir / f"{song.stem}_mixing.png",
        out_dir / f"{song.stem}_mixing.csv",
    )


def main():
    ap = argparse.ArgumentParser(description="Demucs stem separation + mixing analysis")
    ap.add_argument("songs", nargs="+", help="input audio files (mp3/wav/...)")
    ap.add_argument("--out", default="analysis", help="output folder for images/csv")
    ap.add_argument("--sep", default="separated", help="folder for Demucs stems")
    ap.add_argument("--model", default="htdemucs", help="Demucs model name")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    sep_root = Path(args.sep); sep_root.mkdir(parents=True, exist_ok=True)

    if not _HAVE_PYLN:
        print("note: pyloudnorm not installed — loudness reported as dBFS RMS.")

    for pattern in args.songs:
        # expand wildcards relative to cwd; otherwise treat as a direct path
        if any(ch in pattern for ch in "*?["):
            matches = sorted(Path().glob(pattern))
        else:
            matches = [Path(pattern)]
        if not matches:
            print(f"skip: no files match '{pattern}'")
        for song in matches:
            if song.exists():
                process(song, sep_root, out_dir, args.model)
            else:
                print(f"skip: {song} not found")

    print("\nDone.")


if __name__ == "__main__":
    main()
