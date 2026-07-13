

import os
import sys
import json
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save files
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import yt_dlp


def load_song(path, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr
playlist_url="https://www.youtube.com/watch?v=Br3KkvgMAZY&list=RDBr3KkvgMAZY&start_radio=1"



def download_playlist(playlist_url, output_folder="songs"):
    os.makedirs(output_folder, exist_ok=True)
    downloaded_files = []

    def hook(d):
        if d['status'] == 'finished':
            # 'filename' here is pre-conversion; final file will have .mp3 after postprocessing
            base = os.path.splitext(d['filename'])[0]
            downloaded_files.append(base + ".mp3")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_folder}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'progress_hooks': [hook],
        'ignoreerrors': True,   # skip private/unavailable videos instead of crashing
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([playlist_url])

    return downloaded_files


songs = download_playlist(playlist_url)
print(f"Downloaded {len(songs)} songs")


def split_into_segments(y, num_segments=5):
    """Split waveform y into num_segments equal-length chunks (last chunk
    absorbs any leftover samples so nothing gets dropped)."""
    total_samples = len(y)
    seg_len = total_samples // num_segments
    segments = []
    for i in range(num_segments):
        start = i * seg_len
        end = (start + seg_len) if i < num_segments - 1 else total_samples
        segments.append(y[start:end])
    return segments


# --------------------------------------------------------------------------
# 3. Spectrogram generation
# --------------------------------------------------------------------------
def generate_spectrogram(segment, sr, save_path, title="Spectrogram"):
    """Compute a mel spectrogram for a segment and save it as a PNG."""
    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(6, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return S_db


# --------------------------------------------------------------------------
# 4. Volume (RMS) -> percentage -> color array
# --------------------------------------------------------------------------
def volume_percentage_colors(segment, sr, hop_length=512, frame_length=2048,
                              colormap="plasma"):
    """
    Compute the volume (RMS energy) of a segment over time, express each
    value as a percentage of the loudest moment in that segment (0-100),
    and map each percentage to a color using a matplotlib colormap.

    Returns:
        percentages: 1D numpy array, one value per time frame (0-100)
        hex_colors:  list of hex color strings, same length as percentages
        rgba_colors: list of (r, g, b, a) tuples (0-1 floats)
    """
    rms = librosa.feature.rms(y=segment, frame_length=frame_length,
                               hop_length=hop_length)[0]

    max_rms = np.max(rms)
    if max_rms <= 0:
        max_rms = 1e-9  # avoid divide-by-zero on silent segments

    percentages = (rms / max_rms) * 100.0

    cmap = matplotlib.colormaps.get_cmap(colormap)
    norm = mcolors.Normalize(vmin=0, vmax=100)

    rgba_colors = [cmap(norm(p)) for p in percentages]
    hex_colors = [mcolors.to_hex(c) for c in rgba_colors]

    return percentages, hex_colors, rgba_colors


# --------------------------------------------------------------------------
# 5. Put it all together for a list of songs
# --------------------------------------------------------------------------
def process_songs(song_paths, output_dir="segment_outputs", num_segments=5,
                   sr=22050, colormap="plasma"):
    """
    Process a list of song file paths.

    For each song, creates a subfolder in output_dir containing:
      - segment_N_spectrogram.png  for each of the num_segments segments
      - results.json (per-song) with volume percentages + colors

    Returns a dict keyed by song name with all results (also handy for
    using the data directly in Python without reading the JSON back).
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for song_path in song_paths:
        if not os.path.isfile(song_path):
            print(f"  [skip] File not found: {song_path}")
            continue

        song_name = os.path.splitext(os.path.basename(song_path))[0]
        song_out_dir = os.path.join(output_dir, song_name)
        os.makedirs(song_out_dir, exist_ok=True)

        print(f"Processing '{song_name}'...")
        y, loaded_sr = load_song(song_path, sr=sr)
        segments = split_into_segments(y, num_segments=num_segments)

        song_result = {"num_segments": num_segments, "segments": []}

        for idx, seg in enumerate(segments):
            seg_num = idx + 1
            spec_path = os.path.join(song_out_dir,
                                      f"segment_{seg_num}_spectrogram.png")
            generate_spectrogram(seg, loaded_sr, spec_path,
                                  title=f"{song_name} - Segment {seg_num}/{num_segments}")

            percentages, hex_colors, _ = volume_percentage_colors(
                seg, loaded_sr, colormap=colormap
            )

            song_result["segments"].append({
                "segment_index": seg_num,
                "spectrogram_path": spec_path,
                "duration_seconds": round(len(seg) / loaded_sr, 3),
                "volume_percentages": [round(float(p), 2) for p in percentages],
                "volume_colors_hex": hex_colors,
            })
            print(f"  segment {seg_num}/{num_segments} done "
                  f"({len(percentages)} volume samples)")

        all_results[song_name] = song_result

        # Save a per-song JSON file too, so results survive independently
        with open(os.path.join(song_out_dir, "results.json"), "w") as f:
            json.dump(song_result, f, indent=2)

    # Combined JSON for everything
    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python song_segment_analyzer.py <song1> <song2> ...")
        ssongs = ["head.mp3", "9.mp3","fre.mp3"]

    song_list = sys.argv[1:]
    results = process_songs(song_list)
    print(f"\nDone. Results saved in ./segment_outputs/")