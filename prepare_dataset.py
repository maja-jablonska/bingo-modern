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


def _first_col(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns for {what} found: {candidates}. "
                   f"Available: {sorted(df.columns.tolist())}")


def load_willett_merged(path, target_state="rgb", impostor_cut=True,
                        selection_file=None):
    """Load the all-missions merged catalog (merged_with_ages_raw.parquet) and
    return a dataframe in the BNN column convention (BASE_FEATURES + errors,
    logAge/logAgeErr/age, APOGEE_ID, SNR), ready for clean_labels().

    Selection mirrors AnniesLasso/notebooks/train_rgb_wilett_all_missions.ipynb:
      1. finite Willett age (age_L)
      2. ASPCAP sentinel rows dropped (raw_* < -100; isfinite does not catch them)
      3. RelAge_L == False (unreliable reference age) dropped
      4. young alpha-rich impostors dropped (age_L < 4 Gyr, mg_fe > 0.15,
         c_n > -0.2): predominantly mass-transfer/merger products whose
         reference ages are wrong (set impostor_cut=False to keep them)
      5. evolutionary state: seismic EvoState (1=RGB, 2=HeB/RC; Kepler only).
         K2/TESS rows have no EvoState -- the Cannon notebook classifies them
         from the normalized spectra, which this loader cannot do. Pass
         ``selection_file`` (a parquet/csv with the ID column + boolean
         ``is_rgb``, persisted from that notebook) to reuse its selection;
         otherwise only seismically labeled stars survive.

    Spectra columns (wavelength/flux/ivar) are never read.
    """
    import pyarrow.parquet as pq
    spectra_cols = {"wavelength", "flux", "ivar"}
    names = pq.ParquetFile(path).schema_arrow.names
    df = pd.read_parquet(path, columns=[c for c in names if c not in spectra_cols])
    n0 = len(df)

    df = df[np.isfinite(df["age_L"])]

    # Abundance ratios from the Astra raw_*_h convention (X/Fe = X/H - Fe/H).
    for out, xh in [("mg_fe", "raw_mg_h"), ("c_fe", "raw_c_h"), ("n_fe", "raw_n_h")]:
        if out not in df.columns:
            df[out] = df[xh] - df["raw_fe_h"]
    if "c_n" not in df.columns:
        df["c_n"] = df["raw_c_h"] - df["raw_n_h"]

    sentinel_cols = [c for c in ("raw_teff", "raw_logg", "raw_fe_h", "raw_mg_h",
                                 "raw_c_h", "raw_n_h") if c in df.columns]
    sentinel = (df[sentinel_cols] < -100).any(axis=1)
    unreliable = ~df["RelAge_L"].astype(bool)
    impostor = ((df["age_L"] < 4) & (df["mg_fe"] > 0.15) & (df["c_n"] > -0.2)
                if impostor_cut else pd.Series(False, index=df.index))
    drop = sentinel | unreliable | impostor
    print(f"[merged catalog] {n0} rows -> {len(df)} with finite age_L; dropping "
          f"sentinel={int(sentinel.sum())}, RelAge_L=False={int(unreliable.sum())}, "
          f"impostors={int(impostor.sum())} (overlapping) -> {int((~drop).sum())} remain")
    df = df[~drop].reset_index(drop=True)

    id_col = _first_col(df, ["APOGEE_ID", "apogee_id", "sdss_id", "GAIADR3_ID"], "star ID")
    state_code = {"rgb": 1, "rc": 2}[target_state]
    if selection_file is not None:
        sel = (pd.read_parquet(selection_file) if str(selection_file).endswith(".parquet")
               else pd.read_csv(selection_file))
        sel_id = _first_col(sel, [id_col, "APOGEE_ID", "sdss_id"], "selection-file ID")
        keep_ids = set(sel.loc[sel["is_rgb"].astype(bool), sel_id])
        keep = df[id_col].isin(keep_ids)
        print(f"[state] selection file {selection_file}: {int(keep.sum())} of "
              f"{len(df)} rows kept")
    else:
        evo = pd.to_numeric(df.get("EvoState"), errors="coerce").fillna(0).astype(int)
        keep = evo == state_code
        n_missing = int((evo == 0).sum())
        print(f"[state] seismic EvoState=={state_code} only: {int(keep.sum())} of "
              f"{len(df)} rows kept ({n_missing} rows have no seismic state -- "
              "mostly K2/TESS; pass selection_file to include their classified subset)")
    if "source" in df.columns:
        print(df.loc[keep, "source"].value_counts().to_string())
    df = df[keep].reset_index(drop=True)

    # Map to the BNN column convention consumed by the rest of this module.
    out = pd.DataFrame()
    out[ID_COL] = df[id_col].astype(str)
    snr = next((c for c in ("snr", "SNR", "snr_x", "raw_snr") if c in df.columns), None)
    out["SNR"] = df[snr] if snr else 1.0
    out["TEFF"] = df["raw_teff"]
    out["TEFF_ERR"] = df[_first_col(df, ["raw_e_teff", "e_teff"], "Teff error")]
    out["LOGG"] = df["raw_logg"]
    out["LOGG_ERR"] = df[_first_col(df, ["raw_e_logg", "e_logg"], "logg error")]
    out["FE_H"] = df["raw_fe_h"]
    fe_err = df[_first_col(df, ["raw_e_fe_h", "e_fe_h"], "Fe/H error")]
    out["FE_H_ERR"] = fe_err
    for feat, xh, exh in [("MG_FE", "raw_mg_h", ["raw_e_mg_h", "e_mg_fe"]),
                          ("C_FE", "raw_c_h", ["raw_e_c_h", "e_c_fe"]),
                          ("N_FE", "raw_n_h", ["raw_e_n_h", "e_n_fe"])]:
        out[feat] = df[xh] - df["raw_fe_h"]
        ecol = _first_col(df, exh, f"{feat} error")
        # raw_e_*_h errors combine with Fe/H in quadrature for the ratio;
        # e_*_fe columns are already ratio errors.
        out[f"{feat}_ERR"] = (np.sqrt(df[ecol] ** 2 + fe_err ** 2)
                              if ecol.startswith("raw_e_") else df[ecol])

    out["age"] = df["age_L"]
    out[TARGET] = np.log10(df["age_L"])
    if "log_age_L_err" in df.columns:
        out[TARGET_ERR] = df["log_age_L_err"]
    elif any(c in df.columns for c in ("e_age_L", "age_L_err")):
        e = df[_first_col(df, ["e_age_L", "age_L_err"], "age error")]
        out[TARGET_ERR] = e / (df["age_L"] * np.log(10.0))
    elif any(c in df.columns for c in ("age_16_L", "age_L_16")):
        lo = df[_first_col(df, ["age_16_L", "age_L_16"], "age 16th pct")]
        hi = df[_first_col(df, ["age_84_L", "age_L_84"], "age 84th pct")]
        out[TARGET_ERR] = (np.log10(hi) - np.log10(lo)) / 2.0
    else:
        age_like = [c for c in df.columns if "age" in c.lower()]
        raise KeyError("No Willett age-uncertainty column found. Age-related "
                       f"columns available: {age_like}")
    return out


def clean_labels(df, name, drop_saturated=False):
    """Drop NaNs, unphysical ages, and (optionally) the clipped age ceiling.

    ``drop_saturated=True`` removes stars older than UNPHYSICAL_AGE_GYR outright.
    ``False`` (legacy) keeps them pinned at CAP_AGE_GYR, which piles a large
    fraction of the sample onto a single fabricated label value — prefer True
    unless you model the cap as a censored observation.
    """
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

    saturated = df["age"] > UNPHYSICAL_AGE_GYR
    if drop_saturated:
        # Ages older than the Universe are measurement failures, not old stars.
        report["dropped_unphysical_age"] = int(saturated.sum())
        df = df[~saturated]
    else:
        # Legacy: keep saturated stars pinned to CAP_AGE_GYR, with logAge recomputed
        # to match (the catalog uses logAge = log10(age[Gyr])).
        cap_log = float(np.log10(CAP_AGE_GYR))
        report["capped_unphysical_age"] = int(saturated.sum())
        report["cap_age_gyr"] = CAP_AGE_GYR
        df.loc[saturated, "age"] = CAP_AGE_GYR
        df.loc[saturated, TARGET] = cap_log
        if INFLATE_CAPPED_ERR:
            # Treat the cap as a soft lower bound: widen the label error upward so SVI
            # does not over-trust the exact 14 Gyr value. 0.3 dex is ~factor-2 in age.
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
    """Add [C/N] and Salaris-corrected [M/H], each with propagated errors."""
    df = df.copy()
    df["C_N"] = df["C_FE"] - df["N_FE"]
    # Independent errors add in quadrature for a difference.
    df["C_N_ERR"] = np.sqrt(df["C_FE_ERR"] ** 2 + df["N_FE_ERR"] ** 2)
    # Salaris et al. (1993) alpha-corrected global metallicity, with [Mg/Fe] as
    # the alpha proxy: [M/H] = [Fe/H] + log10(0.638 * 10^[a/Fe] + 0.362).
    # Captures the combined metallicity+alpha effect on the giant branch more
    # directly than [Fe/H] and [Mg/Fe] separately.
    alpha_term = 0.638 * 10.0 ** df["MG_FE"]
    df["M_H"] = df["FE_H"] + np.log10(alpha_term + 0.362)
    # d[M/H]/d[Fe/H] = 1;  d[M/H]/d[Mg/Fe] = alpha_term / (alpha_term + 0.362)
    dmg = alpha_term / (alpha_term + 0.362)
    df["M_H_ERR"] = np.sqrt(df["FE_H_ERR"] ** 2 + (dmg * df["MG_FE_ERR"]) ** 2)
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
