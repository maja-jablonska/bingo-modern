"""
Data preparation & validation for the stellar-age BNN.

Works with the catalog you ALREADY have (no new stars required). It:
  1. Cleans labels   - drops unphysical / clipped ages that create the high-age wall.
  2. Engineers [C/N] - the strongest age indicator for giants, derived from C_FE - N_FE
                       with a properly propagated error.
  3. Normalizes      - recomputes (x-mu)/sigma using TRAIN-ONLY statistics, so there is
                       no train->test leakage. Error columns are scaled by the same sigma
                       (xerr/sigma), matching the convention train_bnn.py expects.
  4. Re-balances     - adds an inverse-frequency `train_weight` per logAge bin so the
                       sparse young/old tails count as much as the central hump during
                       SVI. No rows are duplicated or invented.
  5. Validates       - prints a report: balance, label noise, train/test ID overlap,
                       feature coverage, and what was dropped and why.

Outputs train/test CSVs that drop straight into train_bnn.py (plus a norm_stats.json
so the same normalization can be reapplied to future data).

Usage:
    python prepare_dataset.py
Then in train_bnn.py, add 'C_N_NORM' to `all_possible_cols` to use the new feature,
and (optionally) wire `train_weight` into a WeightedRandomSampler -- see notes at bottom.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

RAW_TRAIN = "./train_data/AllTrainedNorm_dr17.csv"
RAW_TEST = "./test_data/TestOriginalNorm_dr17.csv"

OUT_TRAIN = "./train_data/AllTrainedNorm_dr17_clean.csv"
OUT_TEST = "./test_data/TestOriginalNorm_dr17_clean.csv"
OUT_STATS = "./BNN_targeted_output/norm_stats.json"

ID_COL = "APOGEE_ID"

# Base features that exist as raw columns in the catalog. For each FOO we expect
# raw `FOO` and error `FOO_ERR`; the script produces FOO_NORM / FOO_ERR_NORM.
BASE_FEATURES = ["LOGG", "TEFF", "MG_FE", "FE_H", "C_FE", "N_FE"]

# Targets
TARGET = "logAge"
TARGET_ERR = "logAgeErr"

# --- Label cleaning ---------------------------------------------------------
UNPHYSICAL_AGE_GYR = 13.8            # older than the Universe -> treat as saturated
CAP_AGE_GYR = 14.0                   # saturated stars are kept and set to this age
INFLATE_CAPPED_ERR = False           # if True, enlarge logAgeErr of capped stars so the
                                     # likelihood treats their age as a soft lower bound

# --- Re-balancing -----------------------------------------------------------
N_AGE_BINS = 12                      # bins across the logAge range for inverse-freq weights
WEIGHT_CLIP = 10.0                   # cap weight ratio so a near-empty bin can't dominate

# ============================================================================
# Helpers
# ============================================================================


def load(path):
    print(f"Loading {path} ...")
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith((".hdf5", ".h5")):
        from astropy.table import Table
        df = Table.read(path).to_pandas()
    else:
        raise ValueError(f"Unsupported file format: {path}")
    print(f"  {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_labels(df, name):
    """Drop NaNs, unphysical ages, and (optionally) the clipped age ceiling."""
    n0 = len(df)
    report = {}

    # Required columns present?
    need = [ID_COL, TARGET, TARGET_ERR, "age"] + \
           BASE_FEATURES + [f"{f}_ERR" for f in BASE_FEATURES]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"[{name}] missing required columns: {missing}")

    # Drop rows with NaN/inf in any feature, error, or target
    cols = BASE_FEATURES + [f"{f}_ERR" for f in BASE_FEATURES] + [TARGET, TARGET_ERR]
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=cols)
    report["dropped_nan"] = before - len(df)

    # Cap unphysical ages instead of dropping them: saturated stars (older than the
    # Universe) are kept but pinned to CAP_AGE_GYR, and logAge is recomputed to match
    # (the catalog uses logAge = log10(age[Gyr])).
    cap_log = float(np.log10(CAP_AGE_GYR))
    saturated = df["age"] > UNPHYSICAL_AGE_GYR
    report["capped_unphysical_age"] = int(saturated.sum())
    report["cap_age_gyr"] = CAP_AGE_GYR
    df.loc[saturated, "age"] = CAP_AGE_GYR
    df.loc[saturated, TARGET] = cap_log
    if INFLATE_CAPPED_ERR:
        # Treat the cap as a soft lower bound: widen the label error upward so SVI does
        # not over-trust the exact 14 Gyr value. Floor of 0.3 dex is ~factor-2 in age.
        df.loc[saturated, TARGET_ERR] = np.maximum(df.loc[saturated, TARGET_ERR], 0.3)

    # Non-positive errors are unusable for the uncertainty model
    before = len(df)
    df = df[(df[TARGET_ERR] > 0)]
    for f in BASE_FEATURES:
        df = df[df[f"{f}_ERR"] > 0]
    report["dropped_bad_errors"] = before - len(df)

    print(f"[{name}] cleaning: {n0} -> {len(df)} rows  {report}")
    return df.reset_index(drop=True), report


def derive_features(df):
    """Add [C/N] and its propagated error from C_FE, N_FE."""
    df = df.copy()
    df["C_N"] = df["C_FE"] - df["N_FE"]
    # Independent errors add in quadrature for a difference.
    df["C_N_ERR"] = np.sqrt(df["C_FE_ERR"] ** 2 + df["N_FE_ERR"] ** 2)
    return df


def fit_norm_stats(train_df, features):
    """Standardization stats from TRAIN ONLY: x_norm=(x-mu)/sigma, xerr_norm=xerr/sigma."""
    stats = {}
    for f in features:
        mu = float(train_df[f].mean())
        sigma = float(train_df[f].std(ddof=0))
        if sigma == 0 or not np.isfinite(sigma):
            sigma = 1.0
        stats[f] = {"mean": mu, "std": sigma}
    return stats


def apply_norm(df, features, stats):
    df = df.copy()
    for f in features:
        mu, sigma = stats[f]["mean"], stats[f]["std"]
        df[f"{f}_NORM"] = (df[f] - mu) / sigma
        df[f"{f}_ERR_NORM"] = df[f"{f}_ERR"] / sigma
    return df


def add_sample_weights(df, n_bins, clip):
    """Inverse-frequency weights per logAge bin -> flat effective age distribution."""
    df = df.copy()
    lo, hi = df[TARGET].min(), df[TARGET].max()
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(df[TARGET].values, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    counts[counts == 0] = np.nan
    inv = np.nanmean(counts) / counts          # ~1 for average bin, >1 for sparse bins
    inv = np.clip(np.nan_to_num(inv, nan=clip), 1.0 / clip, clip)
    w = inv[idx]
    df["train_weight"] = w / w.mean()          # normalize to mean 1
    return df, edges, counts


# ============================================================================
# Validation report
# ============================================================================


def text_hist(values, edges, label):
    counts, _ = np.histogram(values, bins=edges)
    peak = max(counts.max(), 1)
    print(f"\n  {label} (logAge histogram, n={len(values)}):")
    for i in range(len(counts)):
        bar = "#" * int(40 * counts[i] / peak)
        print(f"    [{edges[i]:+.2f},{edges[i+1]:+.2f})  {counts[i]:4d} {bar}")


def validate(train_df, test_df, features, edges):
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    # Balance
    text_hist(train_df[TARGET].values, edges, "TRAIN")
    text_hist(test_df[TARGET].values, edges, "TEST")

    # Effective (weighted) train balance
    counts, _ = np.histogram(train_df[TARGET].values, bins=edges,
                             weights=train_df["train_weight"].values)
    print("\n  TRAIN effective balance after weighting (should be ~flat):")
    peak = max(counts.max(), 1e-9)
    for i in range(len(counts)):
        bar = "#" * int(40 * counts[i] / peak)
        print(f"    [{edges[i]:+.2f},{edges[i+1]:+.2f})  {counts[i]:7.1f} {bar}")

    # Label noise
    print(f"\n  Label noise (logAgeErr): mean={train_df[TARGET_ERR].mean():.3f} "
          f"median={train_df[TARGET_ERR].median():.3f} dex")
    const = train_df[TARGET_ERR].nunique() == 1
    print(f"  logAgeErr constant fill value? {'YES - SUSPECT' if const else 'no'}")

    # Train/test leakage by ID
    overlap = set(train_df[ID_COL]) & set(test_df[ID_COL])
    print(f"\n  Train/test {ID_COL} overlap: {len(overlap)} "
          f"{'<-- LEAKAGE, remove these' if overlap else '(clean)'}")

    # Feature coverage
    print(f"\n  Features used ({len(features)}): {features}")
    print("  Any NaN remaining in NORM features? "
          f"{train_df[[f + '_NORM' for f in features]].isna().any().any()}")


# ============================================================================
# Main
# ============================================================================


def main():
    train_raw = load(RAW_TRAIN)
    test_raw = load(RAW_TEST)

    train, train_rep = clean_labels(train_raw, "train")
    test, test_rep = clean_labels(test_raw, "test")

    # Remove any test stars that also appear in train (leakage)
    overlap = set(train[ID_COL]) & set(test[ID_COL])
    if overlap:
        print(f"Removing {len(overlap)} leaked IDs from the test set")
        test = test[~test[ID_COL].isin(overlap)].reset_index(drop=True)

    train = derive_features(train)
    test = derive_features(test)

    features = BASE_FEATURES + ["C_N"]

    # Normalize using TRAIN statistics only, then apply to both
    stats = fit_norm_stats(train, features)
    train = apply_norm(train, features, stats)
    test = apply_norm(test, features, stats)

    # Re-balance training set via weights (test left untouched)
    train, edges, _ = add_sample_weights(train, N_AGE_BINS, WEIGHT_CLIP)

    validate(train, test, features, edges)

    # Save
    Path(OUT_STATS).parent.mkdir(exist_ok=True)
    train.to_csv(OUT_TRAIN, index=False)
    test.to_csv(OUT_TEST, index=False)
    with open(OUT_STATS, "w") as fh:
        json.dump({"features": features, "stats": stats,
                   "train_report": train_rep, "test_report": test_rep}, fh, indent=2)

    print("\n" + "=" * 60)
    print(f"Wrote {OUT_TRAIN}  ({len(train)} stars)")
    print(f"Wrote {OUT_TEST}   ({len(test)} stars)")
    print(f"Wrote {OUT_STATS}")
    print("=" * 60)
    print("""
Next steps in train_bnn.py:
  1. Point train_path / test_path at the *_clean.csv files.
  2. Add 'C_N_NORM' to `all_possible_cols` in load_astronomical_data().
  3. (Optional) use the new `train_weight` column to fight age imbalance:
       w = torch.FloatTensor(df['train_weight'].values)
       sampler = torch.utils.data.WeightedRandomSampler(w, len(w), replacement=True)
       loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
     (drop shuffle=True when you pass a sampler).
""")


if __name__ == "__main__":
    main()
