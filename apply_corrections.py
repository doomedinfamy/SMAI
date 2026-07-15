#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

import mixing_model as mm


def parse_targets(pairs):
    """Turn ['perceptual_quality=95', 'clarity=0.8'] into a dict."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            sys.exit(f"Bad --targets entry '{p}', expected col=value")
        col, val = p.split("=", 1)
        if col not in mm.TARGET_COLS:
            sys.exit(f"Unknown target '{col}'. Choose from {mm.TARGET_COLS}")
        out[col] = float(val)
    return out


def main():
    ap = argparse.ArgumentParser(description="Correct song mixing values to hit predicted-quality goals")
    ap.add_argument("--csv", default=mm.DEFAULT_CSV, help="input dataset CSV")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "corrected_songs.csv"),
        help="output CSV path")
    ap.add_argument("--model", default="rf", choices=["rf", "ridge"])
    ap.add_argument("--goal", default="maximize", choices=["maximize", "match"])
    ap.add_argument("--target", default="perceptual_quality",
                    choices=mm.TARGET_COLS,
                    help="target column to maximize (goal=maximize)")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="col=value pairs to match (goal=match)")
    ap.add_argument("--max-step", type=float, default=3.0,
                    help="max change allowed per parameter")
    ap.add_argument("--candidates", type=int, default=4000,
                    help="random candidates searched per song")
    ap.add_argument("--limit", type=int, default=0,
                    help="only process first N songs (0 = all)")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"CSV not found: {args.csv}")

    # ---- Load or train the forward model --------------------------------- #
    if os.path.exists(mm.DEFAULT_MODEL_PATH):
        print(f"Loading model from {mm.DEFAULT_MODEL_PATH}")
        model, feature_cols = mm.load_model()
        _, X, _, _ = mm.load_data(args.csv)
    else:
        print("No saved model found - training a fresh one...")
        model, feature_cols, X, _ = mm.train(args.csv, kind=args.model)
        mm.save_model(model, feature_cols)

    # ---- Decide the goal ------------------------------------------------- #
    match_targets = None
    if args.goal == "match":
        match_targets = parse_targets(args.targets)
        if not match_targets:
            # Default: match the best song's quality in the dataset.
            df_full = pd.read_csv(args.csv)
            best_row = df_full.loc[df_full["perceptual_quality"].idxmax()]
            match_targets = {c: float(best_row[c]) for c in mm.TARGET_COLS}
            print(f"No --targets given; matching best song's quality: "
                  f"{ {k: round(v,2) for k,v in match_targets.items()} }")

    rows_to_do = X if args.limit <= 0 else X.iloc[:args.limit]
    print(f"Correcting {len(rows_to_do)} songs "
          f"(goal={args.goal}, max_step={args.max_step})...")

    # ---- Correct each song ----------------------------------------------- #
    records = []
    for idx, (_, song) in enumerate(rows_to_do.iterrows()):
        params = song.to_dict()
        before = mm.predict_quality(model, feature_cols, params)

        result = mm.suggest_correction(
            model, feature_cols, params, X,
            maximize=args.target,
            targets=match_targets,
            n_candidates=args.candidates,
            max_step=args.max_step,
        )
        after = result["predicted_quality"]
        corrected = result["suggested_params"]

        rec = {"song_id": idx}
        rec.update(corrected)  # the modified mixing values
        for c in mm.TARGET_COLS:
            rec[f"pred_before_{c}"] = round(before[c], 4)
            rec[f"pred_after_{c}"] = round(after[c], 4)
        records.append(rec)

        if (idx + 1) % 25 == 0 or idx + 1 == len(rows_to_do):
            print(f"  ...{idx + 1}/{len(rows_to_do)} done")

    out_df = pd.DataFrame(records)
    out_df.to_csv(args.out, index=False)

    # ---- Summary --------------------------------------------------------- #
    key = args.target
    gain = (out_df[f"pred_after_{key}"] - out_df[f"pred_before_{key}"]).mean()
    print()
    print("=" * 60)
    print(f"Wrote {len(out_df)} corrected songs -> {args.out}")
    print(f"Average predicted {key} change: {gain:+.2f}")
    print(f"  before mean: {out_df[f'pred_before_{key}'].mean():.2f}")
    print(f"  after  mean: {out_df[f'pred_after_{key}'].mean():.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
