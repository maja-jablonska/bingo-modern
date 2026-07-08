"""
Classify a bulge catalog into RGB / RC with the spectroscopic classifier from
classify_rgb_rc.ipynb, streaming the catalog in chunks so arbitrarily large
files never have to fit in memory.

Loads the saved classifier bundle (RGB_RC_classifier/rgb_rc_classifier.pkl) if
present and loadable; otherwise (or with --retrain) trains it from
train_data/{rgb,rc}_clean_labels.parquet — identical data, params, and seed to
the notebook, so results match. Ambiguous stars (thr_lo <= P(RC) <= thr_hi)
are dropped; only the RGB and RC parquets are written.

Column handling: each needed quantity is resolved against the catalog's actual
schema (raw_teff or teff, mg_fe or raw_mg_fe, ...). [X/Fe] abundances missing
from the catalog are derived as [X/H] - [Fe/H] when an _h column exists, with
errors combined in quadrature. A missing ID column is synthesized from the row
number. Only genuinely unresolvable classifier features are a hard error, and
the message lists the catalog's columns.

Usage:
    python classify_bulge.py --bulge /path/to/bulge_catalog.parquet
    python classify_bulge.py --bulge catalog.csv --out-dir ./RGB_RC_classifier \
        --batch-rows 500000 --retrain
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

# --- must stay in sync with classify_rgb_rc.ipynb -------------------------
RGB_LABELS = "train_data/rgb_clean_labels.parquet"
RC_LABELS = "train_data/rc_clean_labels.parquet"
SEED = 42
FEATS = ["raw_teff", "raw_logg", "raw_fe_h", "mg_fe", "c_fe", "n_fe", "c_n"]
THR_LO, THR_HI = 0.3, 0.7
HGB_PARAMS = dict(max_leaf_nodes=15, l2_regularization=1.0, learning_rate=0.06,
                  random_state=SEED)

# Canonical (training) column -> the FOO/FOO_ERR schema the age pipelines expect.
FEATURE_MAP = {
    "APOGEE_ID": "APOGEE_ID",
    "raw_teff": "TEFF", "raw_e_teff": "TEFF_ERR",
    "raw_logg": "LOGG", "raw_e_logg": "LOGG_ERR",
    "raw_fe_h": "FE_H", "raw_e_fe_h": "FE_H_ERR",
    "mg_fe": "MG_FE", "e_mg_fe": "MG_FE_ERR",
    "c_fe": "C_FE", "e_c_fe": "C_FE_ERR",
    "n_fe": "N_FE", "e_n_fe": "N_FE_ERR",
}
# ---------------------------------------------------------------------------

# Direct alternatives per canonical column, first hit wins.
DIRECT_ALTS = {
    "APOGEE_ID": ["APOGEE_ID", "apogee_id", "sdss_id", "GAIADR3_ID", "gaiadr3_id",
                  "gaia_dr3_source_id", "source_id"],
    "raw_teff": ["raw_teff", "teff"],
    "raw_e_teff": ["raw_e_teff", "e_teff"],
    "raw_logg": ["raw_logg", "logg"],
    "raw_e_logg": ["raw_e_logg", "e_logg"],
    "raw_fe_h": ["raw_fe_h", "fe_h"],
    "raw_e_fe_h": ["raw_e_fe_h", "e_fe_h"],
    "mg_fe": ["mg_fe", "raw_mg_fe"],
    "e_mg_fe": ["e_mg_fe", "raw_e_mg_fe"],
    "c_fe": ["c_fe", "raw_c_fe"],
    "e_c_fe": ["e_c_fe", "raw_e_c_fe"],
    "n_fe": ["n_fe", "raw_n_fe"],
    "e_n_fe": ["e_n_fe", "raw_e_n_fe"],
}
# Fallback [X/H] columns for deriving [X/Fe] = [X/H] - [Fe/H].
XH_ALTS = {
    "mg_fe": ["mg_h", "raw_mg_h"],
    "c_fe": ["c_h", "raw_c_h"],
    "n_fe": ["n_h", "raw_n_h"],
}
EH_ALTS = {
    "e_mg_fe": ["e_mg_h", "raw_e_mg_h"],
    "e_c_fe": ["e_c_h", "raw_e_c_h"],
    "e_n_fe": ["e_n_h", "raw_e_n_h"],
}
# Columns the classifier itself needs values for; unresolvable => hard error.
# (Missing errors or ID only degrade the output, so they just warn.)
CRITICAL = ["raw_teff", "raw_logg", "raw_fe_h", "mg_fe", "c_fe", "n_fe"]


def train_bundle(out_path):
    rgb = pd.read_parquet(RGB_LABELS).assign(is_rc=0)
    rc = pd.read_parquet(RC_LABELS).assign(is_rc=1)
    conflicts = set(rgb.APOGEE_ID) & set(rc.APOGEE_ID)
    df = pd.concat([rgb, rc], ignore_index=True)
    df = df[~df.APOGEE_ID.isin(conflicts)].reset_index(drop=True)
    df["c_n"] = df["c_fe"] - df["n_fe"]
    df = df.dropna(subset=FEATS).reset_index(drop=True)

    clf = HistGradientBoostingClassifier(**HGB_PARAMS)
    clf.fit(df[FEATS].values, df["is_rc"].values)

    bundle = {
        "model": clf,
        "features": FEATS,
        "thr_lo": THR_LO,
        "thr_hi": THR_HI,
        "domain_1_99": {f: (float(np.percentile(df[f], 1)),
                            float(np.percentile(df[f], 99))) for f in FEATS},
        "sklearn_version": sklearn.__version__,
        "n_train": len(df),
        "label_convention": "P(RC): is_rc=1 for red clump, 0 for RGB",
    }
    out_path.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"Trained on {len(df)} stars "
          f"({(df.is_rc == 0).sum()} RGB / {(df.is_rc == 1).sum()} RC), "
          f"saved {out_path}")
    return bundle


def get_bundle(bundle_path, retrain):
    if not retrain and bundle_path.exists():
        try:
            bundle = joblib.load(bundle_path)
            print(f"Loaded {bundle_path} "
                  f"(trained on {bundle['n_train']} stars with "
                  f"sklearn {bundle['sklearn_version']}; running {sklearn.__version__})")
            return bundle
        except Exception as e:  # e.g. sklearn version mismatch across machines
            print(f"Could not load {bundle_path} ({e}); retraining instead")
    return train_bundle(bundle_path)


def build_plan(names):
    """Resolve every canonical column against the catalog schema.

    Returns (plan, read_cols): plan maps canonical name -> recipe tuple,
    read_cols is the set of catalog columns actually needed.
    """
    nameset = set(names)
    plan, read_cols, derived, missing = {}, set(), [], []

    for target, alts in DIRECT_ALTS.items():
        col = next((c for c in alts if c in nameset), None)
        if col is not None:
            plan[target] = ("direct", col)
            read_cols.add(col)
            continue
        xh = next((c for c in XH_ALTS.get(target, []) if c in nameset), None)
        if xh is not None:
            plan[target] = ("xh_minus_feh", xh)
            read_cols.add(xh)
            derived.append(f"{target} = {xh} - [Fe/H]")
            continue
        eh = next((c for c in EH_ALTS.get(target, []) if c in nameset), None)
        if eh is not None:
            plan[target] = ("quad_feh_err", eh)
            read_cols.add(eh)
            derived.append(f"{target} = sqrt({eh}^2 + e_fe_h^2)")
            continue
        plan[target] = ("missing",)
        missing.append(target)

    unresolvable = [t for t in CRITICAL if plan[t][0] == "missing"]
    if unresolvable:
        raise KeyError(
            f"cannot resolve classifier features {unresolvable} from the catalog "
            f"(no direct or [X/H] fallback column). Catalog columns:\n  "
            + "\n  ".join(sorted(names)))

    if derived:
        print("Derived columns: " + "; ".join(derived))
    for t in missing:
        if t == "APOGEE_ID":
            print("No ID column found -- synthesizing row_<N> IDs")
        else:
            print(f"WARNING: no source for {t} -- writing NaN "
                  f"({FEATURE_MAP[t]} in the output; the age pipelines need it)")
    return plan, sorted(read_cols)


def canonicalize(chunk, plan, row_offset):
    """Build the canonical-schema frame for one chunk."""
    out = pd.DataFrame(index=chunk.index)
    # [Fe/H] and its error first: the derivations below need them.
    for target in ["raw_fe_h", "raw_e_fe_h"]:
        if plan[target][0] == "direct":
            out[target] = chunk[plan[target][1]]
        else:
            out[target] = np.nan
    for target, recipe in plan.items():
        kind = recipe[0]
        if kind == "direct":
            out[target] = chunk[recipe[1]]
        elif kind == "xh_minus_feh":
            out[target] = chunk[recipe[1]] - out["raw_fe_h"]
        elif kind == "quad_feh_err":
            out[target] = np.sqrt(chunk[recipe[1]] ** 2 + out["raw_e_fe_h"] ** 2)
        elif target == "APOGEE_ID":
            out[target] = [f"row_{i}" for i in
                           range(row_offset, row_offset + len(chunk))]
        else:
            out[target] = np.nan
    out["c_n"] = out["c_fe"] - out["n_fe"]
    return out


def get_schema_names(path):
    if path.suffix == ".parquet":
        return pq.ParquetFile(path).schema_arrow.names
    return list(pd.read_csv(path, nrows=0).columns)


def iter_chunks(path, read_cols, batch_rows):
    """Yield DataFrame chunks with only read_cols, never the full catalog."""
    if path.suffix == ".parquet":
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=read_cols):
            yield batch.to_pandas()
    else:
        yield from pd.read_csv(path, usecols=read_cols, chunksize=batch_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--bulge", required=True, help="bulge catalog (.parquet or .csv)")
    ap.add_argument("--out-dir", default="./RGB_RC_classifier", type=Path)
    ap.add_argument("--batch-rows", default=500_000, type=int,
                    help="rows classified per chunk (default 500k)")
    ap.add_argument("--retrain", action="store_true",
                    help="retrain even if a saved bundle exists")
    args = ap.parse_args()

    bulge_path = Path(args.bulge).expanduser()
    if not bulge_path.exists():
        sys.exit(f"Bulge catalog not found: {bulge_path}")
    args.out_dir.mkdir(exist_ok=True)

    bundle = get_bundle(args.out_dir / "rgb_rc_classifier.pkl", args.retrain)
    thr_lo, thr_hi = bundle["thr_lo"], bundle["thr_hi"]
    feats = bundle["features"]

    plan, read_cols = build_plan(get_schema_names(bulge_path))

    keep = list(FEATURE_MAP.values()) + ["p_rc", "evo_class", "in_domain"]
    float_cols = [c for c in keep if c not in ("APOGEE_ID", "evo_class", "in_domain")]
    out_paths = {cls: args.out_dir / f"bulge_{cls.lower()}_bnn.parquet"
                 for cls in ("RGB", "RC")}
    writers = {}
    stats = {"rows": 0, "nan": 0, "ambiguous": 0, "RGB": 0, "RC": 0, "in_domain": 0}

    try:
        for raw_chunk in iter_chunks(bulge_path, read_cols, args.batch_rows):
            chunk = canonicalize(raw_chunk, plan, stats["rows"])
            n_in = len(chunk)
            stats["rows"] += n_in
            chunk = chunk.dropna(subset=feats)
            stats["nan"] += n_in - len(chunk)
            if chunk.empty:
                continue

            p_rc = bundle["model"].predict_proba(chunk[feats].values)[:, 1]
            chunk["p_rc"] = p_rc
            chunk["evo_class"] = np.select(
                [p_rc < thr_lo, p_rc > thr_hi], ["RGB", "RC"], default="ambiguous")
            chunk["in_domain"] = np.all(
                [(chunk[f] >= lo) & (chunk[f] <= hi)
                 for f, (lo, hi) in bundle["domain_1_99"].items()], axis=0)
            stats["ambiguous"] += int((chunk["evo_class"] == "ambiguous").sum())
            stats["in_domain"] += int(chunk["in_domain"].sum())

            out = chunk.rename(columns=FEATURE_MAP)
            for cls in ("RGB", "RC"):
                sel = out.loc[out["evo_class"] == cls, keep].copy()
                stats[cls] += len(sel)
                if sel.empty:
                    continue
                sel["APOGEE_ID"] = sel["APOGEE_ID"].astype(str)
                sel[float_cols] = sel[float_cols].astype("float64")
                table = pa.Table.from_pandas(sel, preserve_index=False)
                if cls not in writers:
                    writers[cls] = pq.ParquetWriter(out_paths[cls], table.schema)
                writers[cls].write_table(table.cast(writers[cls].schema))
    finally:
        for w in writers.values():
            w.close()

    # A class that never appeared still gets a (valid, empty) output file.
    for cls, path in out_paths.items():
        if cls not in writers:
            pd.DataFrame(columns=keep).to_parquet(path, index=False)

    classified = stats["RGB"] + stats["RC"]
    print(f"\n{bulge_path.name}: {stats['rows']} rows")
    print(f"  dropped: {stats['nan']} with NaN features, "
          f"{stats['ambiguous']} ambiguous ({thr_lo} <= P(RC) <= {thr_hi})")
    for cls in ("RGB", "RC"):
        print(f"  {out_paths[cls]}: {stats[cls]} stars")
    if classified:
        print(f"  in_domain (within training 1-99 pct ranges): "
              f"{stats['in_domain'] / (classified + stats['ambiguous']):.1%}")


if __name__ == "__main__":
    main()
