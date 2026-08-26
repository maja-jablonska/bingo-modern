#!/usr/bin/env python
"""
Out-of-fold purity/completeness of the RGB/RC classifier against SEISMIC
truth, at several operating points.

``classify_rgb_rc.ipynb`` reports its three-way band metrics on the full
labelled sample, whose labels are ``EvoState_pred`` -- itself an upstream
prediction, so the quoted ~97% RGB purity is partly circular. The gold subset
(stars with a seismic consensus ``EvoState``) is the only place where truth is
independent of any classifier, and purity there is materially lower.

This script reproduces that number, and prints what the training-side operating
point of AnniesLasso's classifier (accept at p > 0.9, i.e. p(RC) < 0.1) would
buy on the same stars -- the harmonization the two pipelines are missing.

Usage:  python check_classifier_purity.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# must stay in sync with classify_bulge.py / classify_rgb_rc.ipynb
RGB_LABELS = "train_data/rgb_clean_labels.parquet"
RC_LABELS = "train_data/rc_clean_labels.parquet"
SEED = 42
FEATS = ["raw_teff", "raw_logg", "raw_fe_h", "mg_fe", "c_fe", "n_fe", "c_n"]
HGB_PARAMS = dict(max_leaf_nodes=15, l2_regularization=1.0, learning_rate=0.06,
                  random_state=SEED)

# (label, p(RC) low threshold, p(RC) high threshold)
OPERATING_POINTS = [
    ("three-way 0.3/0.7  [bulge]", 0.3, 0.7),
    ("p(RC) < 0.1        [train-side p > 0.9]", 0.1, 0.9),
    ("hard 0.5", 0.5, 0.5),
]


def load():
    rgb = pd.read_parquet(RGB_LABELS).assign(is_rc=0)
    rc = pd.read_parquet(RC_LABELS).assign(is_rc=1)
    conflicts = set(rgb.APOGEE_ID) & set(rc.APOGEE_ID)
    df = pd.concat([rgb, rc], ignore_index=True)
    df = df[~df.APOGEE_ID.isin(conflicts)].reset_index(drop=True)
    df["c_n"] = df["c_fe"] - df["n_fe"]
    return df.dropna(subset=FEATS).reset_index(drop=True)


def main():
    df = load()
    X, y = df[FEATS].values, df.is_rc.values
    gold = df["EvoState"].isin([1.0, 2.0]).values
    y_seismic = (df["EvoState"].values == 2.0).astype(int)
    print("n=%d  gold (seismic EvoState)=%d" % (len(df), gold.sum()))

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p_rc = cross_val_predict(HistGradientBoostingClassifier(**HGB_PARAMS),
                             X, y, cv=cv, method="predict_proba")[:, 1]

    for title, mask, truth in [
            ("FULL sample -- labels are EvoState_pred (partly circular)",
             np.ones(len(y), bool), y),
            ("GOLD subset -- labels are seismic EvoState (independent)",
             gold, y_seismic)]:
        print("\n%s:" % title)
        for name, lo, hi in OPERATING_POINTS:
            decided = (p_rc < lo) | (p_rc > hi)
            selected = mask & decided & (p_rc < lo)      # accepted as RGB
            purity = (truth[selected] == 0).mean() if selected.any() else np.nan
            completeness = selected[mask & (truth == 0)].mean()
            print("  %-42s decided=%5.1f%%  RGB n=%5d  purity=%.3f  "
                  "completeness=%.3f"
                  % (name, 100 * decided[mask].mean(), selected.sum(),
                     purity, completeness))


if __name__ == "__main__":
    main()
