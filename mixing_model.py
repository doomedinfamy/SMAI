#!/usr/bin/env python3
"""
mixing_model.py
================
Machine-learning pipeline for the music-mixing dataset.

The dataset (`music_mixing_dataset.csv`) has 28 numeric columns:

    24 INPUT features  -> the mixing parameters you dial in per stem
        gain_{vocals,drums,bass,instruments}                (dB)
        eq_{stem}_{low,mid,high}                            (dB)
        comp_{stem}                                         (compression amount)
        pan_{stem}                                          (-1 = L ... +1 = R)

     4 TARGET outputs  -> the perceived quality of the resulting mix
        loudness            (LUFS-ish, negative dB scale)
        clarity             (0..1)
        spectral_balance    (0..1)
        perceptual_quality  (0..100 overall score)

What this script does
---------------------
1.  PROFILE   - prints shape, missing values, and summary stats.
2.  TRAIN     - fits a multi-output regressor (mixing params -> 4 quality
                targets) and reports R2 / MAE / RMSE per target, plus
                cross-validated scores.
3.  PREDICT   - `predict_quality()` scores an "upcoming song" from its
                mixing settings.
4.  CORRECT   - `suggest_correction()` searches for the parameter tweaks
                that best push a mix toward a target quality (e.g. maximise
                perceptual_quality), using the trained forward model.
5.  SAVE/LOAD - persists the trained pipeline to `mixing_model.joblib`.

Run it directly to profile, train, evaluate, and see a worked example:

    python mixing_model.py
    python mixing_model.py --csv music_mixing_dataset.csv --model rf

Dependencies: pandas, numpy, scikit-learn, joblib
    pip install pandas numpy scikit-learn joblib
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import joblib


# --------------------------------------------------------------------------- #
# Column definitions
# --------------------------------------------------------------------------- #
TARGET_COLS = ["loudness", "clarity", "spectral_balance", "perceptual_quality"]

DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "music_mixing_dataset.csv")
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "mixing_model.joblib")


# --------------------------------------------------------------------------- #
# Data loading / profiling
# --------------------------------------------------------------------------- #
def load_data(csv_path: str):
    """Load the CSV and split into feature frame X and target frame y."""
    df = pd.read_csv(csv_path)
    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected target columns missing from CSV: {missing}")
    feature_cols = [c for c in df.columns if c not in TARGET_COLS]
    X = df[feature_cols].copy()
    y = df[TARGET_COLS].copy()
    return df, X, y, feature_cols


def profile(df: pd.DataFrame, feature_cols, y: pd.DataFrame) -> None:
    """Print the crucial ML facts about the dataset."""
    print("=" * 70)
    print("DATASET PROFILE")
    print("=" * 70)
    print(f"Rows (samples)   : {len(df)}")
    print(f"Columns total    : {df.shape[1]}")
    print(f"Input features   : {len(feature_cols)}")
    print(f"Target outputs   : {len(TARGET_COLS)}  -> {TARGET_COLS}")
    print()

    n_missing = int(df.isna().sum().sum())
    n_dupes = int(df.duplicated().sum())
    print(f"Missing values   : {n_missing}")
    print(f"Duplicate rows   : {n_dupes}")
    print()

    print("Feature summary (min / mean / max / std):")
    desc = df[feature_cols].describe().T[["min", "mean", "max", "std"]]
    print(desc.round(3).to_string())
    print()

    print("Target summary (min / mean / max / std):")
    tdesc = y.describe().T[["min", "mean", "max", "std"]]
    print(tdesc.round(3).to_string())
    print()

    print("Strongest feature -> target correlations (|r|):")
    corr = pd.concat([df[feature_cols], y], axis=1).corr()
    for t in TARGET_COLS:
        s = corr[t].drop(TARGET_COLS).abs().sort_values(ascending=False).head(3)
        pairs = ", ".join(f"{k} ({corr[t][k]:+.2f})" for k in s.index)
        print(f"  {t:<18}: {pairs}")
    print()


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_model(kind: str = "rf") -> Pipeline:
    """
    Build a scaling + multi-output regression pipeline.

    kind = 'rf'    -> RandomForest (non-linear, robust default)
    kind = 'ridge' -> Ridge linear regression (fast, interpretable baseline)
    """
    if kind == "ridge":
        base = Ridge(alpha=1.0)
    elif kind == "rf":
        base = RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            n_jobs=2, random_state=42,
        )
    else:
        raise ValueError("model kind must be 'rf' or 'ridge'")

    # Scale features; wrap regressor for multi-output; scale targets too so
    # the four very-differently-ranged targets are learned evenly.
    regressor = TransformedTargetRegressor(
        regressor=MultiOutputRegressor(base),
        transformer=StandardScaler(),
    )
    return Pipeline([("scaler", StandardScaler()), ("model", regressor)])


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def evaluate(model: Pipeline, X_test, y_test) -> None:
    """Print per-target regression metrics on a held-out test set."""
    preds = model.predict(X_test)
    preds = np.asarray(preds).reshape(len(y_test), -1)
    print("=" * 70)
    print("HELD-OUT TEST METRICS")
    print("=" * 70)
    print(f"{'target':<20}{'R2':>8}{'MAE':>10}{'RMSE':>10}")
    for i, t in enumerate(TARGET_COLS):
        yt = y_test.iloc[:, i].values
        yp = preds[:, i]
        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        print(f"{t:<20}{r2:>8.3f}{mae:>10.3f}{rmse:>10.3f}")
    print()


def cross_validate(model: Pipeline, X, y, folds: int = 5) -> None:
    """Report cross-validated R2 (averaged across the 4 targets)."""
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
    print(f"{folds}-fold CV R2 (mean over targets): "
          f"{scores.mean():.3f} +/- {scores.std():.3f}")
    print()


def train(csv_path: str, kind: str = "rf", test_size: float = 0.2):
    """Full training routine. Returns (model, feature_cols, X, y)."""
    df, X, y, feature_cols = load_data(csv_path)
    profile(df, feature_cols, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    model = build_model(kind)
    model.fit(X_train, y_train)

    evaluate(model, X_test, y_test)
    cross_validate(model, X, y)

    return model, feature_cols, X, y


# --------------------------------------------------------------------------- #
# Inference: score an upcoming song
# --------------------------------------------------------------------------- #
def predict_quality(model: Pipeline, feature_cols, params: dict) -> dict:
    """
    Predict the 4 quality targets for a new song's mixing settings.

    `params` is a dict of {feature_name: value}. Any missing feature
    defaults to 0.0 (a neutral setting).
    """
    row = {c: float(params.get(c, 0.0)) for c in feature_cols}
    X_new = pd.DataFrame([row], columns=feature_cols)
    pred = np.asarray(model.predict(X_new)).reshape(-1)
    return {t: float(pred[i]) for i, t in enumerate(TARGET_COLS)}


# --------------------------------------------------------------------------- #
# Correction: what should I change to hit a target quality?
# --------------------------------------------------------------------------- #
def suggest_correction(model, feature_cols, current_params: dict, X_ref,
                       maximize="perceptual_quality", targets=None,
                       n_candidates=4000, max_step=3.0, seed=42) -> dict:
    """
    Search for small parameter tweaks that improve a mix.

    Strategy: random local search around the current settings using the
    trained forward model as an oracle (no gradients needed, works with any
    regressor). Returns the best-found parameter set, the resulting predicted
    quality, and the per-parameter deltas.

    maximize : target column to push as high as possible (e.g. perceptual_quality)
    targets  : optional dict {col: desired_value} to instead match specific
               target values (minimises squared error to them). Overrides
               `maximize` when provided.
    max_step : maximum absolute change allowed per parameter (keeps tweaks
               musically sensible).
    """
    rng = np.random.default_rng(seed)
    base = np.array([float(current_params.get(c, 0.0)) for c in feature_cols])

    # Per-feature bounds taken from the observed data range, so suggestions
    # stay within realistic mixing values.
    lo = X_ref[feature_cols].min().values
    hi = X_ref[feature_cols].max().values

    def score(param_matrix):
        pm = pd.DataFrame(param_matrix, columns=feature_cols)
        preds = np.asarray(model.predict(pm))
        preds = preds.reshape(len(param_matrix), -1)
        pred_df = {t: preds[:, i] for i, t in enumerate(TARGET_COLS)}
        if targets:
            err = np.zeros(len(param_matrix))
            for col, want in targets.items():
                j = TARGET_COLS.index(col)
                err += (preds[:, j] - want) ** 2
            return -err  # higher is better
        return pred_df[maximize]

    # Candidate tweaks: base + bounded random deltas.
    deltas = rng.uniform(-max_step, max_step, size=(n_candidates, len(base)))
    candidates = np.clip(base + deltas, lo, hi)
    candidates = np.vstack([base, candidates])  # include "do nothing"

    scores = score(candidates)
    best_idx = int(np.argmax(scores))
    best = candidates[best_idx]

    best_params = {c: float(best[i]) for i, c in enumerate(feature_cols)}
    best_quality = predict_quality(model, feature_cols, best_params)
    deltas_out = {c: float(best[i] - base[i]) for i, c in enumerate(feature_cols)
                  if abs(best[i] - base[i]) > 1e-6}

    return {
        "suggested_params": best_params,
        "predicted_quality": best_quality,
        "changes": dict(sorted(deltas_out.items(),
                               key=lambda kv: -abs(kv[1]))),
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_model(model, feature_cols, path: str = DEFAULT_MODEL_PATH) -> None:
    joblib.dump({"model": model, "feature_cols": feature_cols,
                 "target_cols": TARGET_COLS}, path)
    print(f"Saved model -> {path}")


def load_model(path: str = DEFAULT_MODEL_PATH):
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_cols"]


# --------------------------------------------------------------------------- #
# CLI / demo
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Music-mixing ML pipeline")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="path to dataset CSV")
    ap.add_argument("--model", default="rf", choices=["rf", "ridge"],
                    help="regressor type")
    ap.add_argument("--no-save", action="store_true",
                    help="do not write the .joblib model")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"CSV not found: {args.csv}")

    model, feature_cols, X, y = train(args.csv, kind=args.model)

    if not args.no_save:
        save_model(model, feature_cols)

    # ---- Worked example: score a new song, then suggest a correction ----
    print("=" * 70)
    print("EXAMPLE: score an upcoming song")
    print("=" * 70)
    sample = X.iloc[0].to_dict()   # pretend this is a new mix
    q = predict_quality(model, feature_cols, sample)
    print("Predicted quality:")
    for k, v in q.items():
        print(f"  {k:<18}: {v:.2f}")
    print()

    print("=" * 70)
    print("EXAMPLE: suggest corrections to maximise perceptual_quality")
    print("=" * 70)
    fix = suggest_correction(model, feature_cols, sample, X,
                             maximize="perceptual_quality")
    print(f"Predicted perceptual_quality: "
          f"{q['perceptual_quality']:.2f} -> "
          f"{fix['predicted_quality']['perceptual_quality']:.2f}")
    print("Top suggested parameter changes:")
    for k, v in list(fix["changes"].items())[:8]:
        print(f"  {k:<22}: {v:+.2f}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
