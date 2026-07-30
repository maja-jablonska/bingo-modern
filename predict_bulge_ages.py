"""
Apply a trained BINGO BNN age model to a classified bulge sample.

Input:  a parquet from classify_bulge.py (bulge_rgb_bnn.parquet or
        bulge_rc_bnn.parquet) with TEFF/TEFF_ERR-style columns.
Model:  the inference bundle + Pyro param store written by the training
        notebook's final cells (e.g. BNN_targeted_output_RGB/
        bnn_inference_bundle_rgb.pkl) -- run the training notebook first.

Streams the catalog in chunks; per chunk it normalizes with the bundle's
train-only stats, draws posterior predictive samples, combines model
uncertainty and intrinsic scatter exactly as analyze_targeted_results does
(vectorized; observational label error is zero for unlabeled stars), applies
the bundle's scalar calibration factor, and appends to one output parquet.

Usage:
    python predict_bulge_ages.py --input RGB_RC_classifier/bulge_rgb_bnn.parquet \
        --bundle BNN_targeted_output_RGB/bnn_inference_bundle_rgb.pkl
    # RC: same, pointing at the RC parquet and the RC model's bundle.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import pyro
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDiagonalNormal

import prepare_dataset as prep
from train_bnn import BayesianNeuralNetwork, set_seed, device

CARRY_COLS = ["APOGEE_ID", "p_rc", "in_domain"]  # passed through if present


def load_model(bundle_path):
    bundle = joblib.load(bundle_path)
    params_path = bundle_path.parent / bundle["param_store_file"]
    if not params_path.exists():
        sys.exit(f"Param store not found: {params_path} -- run the training notebook")

    pyro.clear_param_store()
    # ParamStore.load uses torch.load's weights_only default (True since
    # torch 2.6), which rejects the pickled constraint objects Pyro stores.
    # The file is our own training artifact, so load it explicitly.
    state = torch.load(str(params_path), map_location=device, weights_only=False)
    pyro.get_param_store().set_state(state)
    model = BayesianNeuralNetwork(**bundle["model_kwargs"]).to(device)
    model.eval()
    # Fresh guide with the default name picks up the loaded params lazily.
    guide = AutoDiagonalNormal(model)
    print(f"Loaded {bundle_path.name}: features={bundle['features']}, "
          f"calibration_factor={bundle['calibration_factor']:.3f}")
    return bundle, model, guide


def predict_chunk(chunk, bundle, model, guide, num_samples, rng, debias=False):
    feats = bundle["features"]
    chunk = prep.derive_features(chunk)
    chunk = prep.apply_norm(chunk, feats, bundle["stats"])

    X = torch.FloatTensor(
        chunk[[f"{f}_NORM" for f in feats]].values.astype(np.float32)).to(device)
    X_err = torch.FloatTensor(
        chunk[[f"{f}_ERR_NORM" for f in feats]].values.astype(np.float32)).to(device)

    predictive = Predictive(
        model, guide=guide, num_samples=num_samples,
        return_sites=["prediction", "model_uncertainty", "intrinsic_scatter"])
    with torch.no_grad():
        s = predictive(X, X_err)
    pred = s["prediction"].cpu().numpy()              # (num_samples, n)
    model_unc = s["model_uncertainty"].cpu().numpy()  # (num_samples, n)
    intrinsic = s["intrinsic_scatter"].cpu().numpy().reshape(num_samples, 1)

    # Same combination as analyze_targeted_results, vectorized, with y_err=0
    # (bulge stars have no age label): every posterior sample scattered by its
    # own model uncertainty and intrinsic scatter.
    total = rng.normal(pred, np.sqrt(model_unc ** 2 + intrinsic ** 2))

    med = np.median(total, axis=0)
    p16, p84 = np.percentile(total, [16, 84], axis=0)

    out = pd.DataFrame({c: chunk[c].values for c in CARRY_COLS if c in chunk})
    # Extrapolation flag on the RAW median, as in the training notebook: a
    # prediction outside the training logAge range left the region the data
    # constrains -- broken, not merely uncertain.
    lo, hi = bundle.get("train_label_range", (-np.inf, np.inf))
    out["flag_outside_train_range"] = (med < lo) | (med > hi)

    if debias:
        # Bundle's affine de-shrinkage from the inverse calibration fit on
        # val: pred' = a*pred + b, sigma' = raw_sigma * sigma_factor. Interval
        # half-widths get the same sigma_factor, around the debiased median.
        db = bundle["affine_debias"]
        a, b, calib = db["a"], db["b"], db["sigma_factor"]
        out["logAge"] = a * med + b
        out["logAgeErr"] = calib * total.std(axis=0)
        out["logAge_16"] = out["logAge"] + calib * (p16 - med)
        out["logAge_84"] = out["logAge"] + calib * (p84 - med)
    else:
        calib = bundle["calibration_factor"]
        out["logAge"] = med
        # Calibrated: interval half-widths scaled around the median by the
        # factor fit on the validation split.
        out["logAgeErr"] = calib * total.std(axis=0)
        out["logAge_16"] = med + calib * (p16 - med)
        out["logAge_84"] = med + calib * (p84 - med)
    out["model_uncertainty"] = model_unc.mean(axis=0)
    out["age"] = 10.0 ** out["logAge"]                 # Gyr
    out["age_16"] = 10.0 ** out["logAge_16"]
    out["age_84"] = 10.0 ** out["logAge_84"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--input", required=True,
                    help="classified parquet from classify_bulge.py")
    ap.add_argument("--bundle", required=True, type=Path,
                    help="bnn_inference_bundle_*.pkl from the training notebook")
    ap.add_argument("--output", default=None,
                    help="output parquet (default: <input stem>_ages.parquet)")
    ap.add_argument("--num-samples", default=1000, type=int,
                    help="posterior samples per star (default 1000)")
    ap.add_argument("--debias", action="store_true",
                    help="apply the bundle's affine de-shrinkage "
                         "(pred' = a*pred + b, sigma' = raw_sigma * "
                         "sigma_factor) instead of the raw calibration")
    ap.add_argument("--batch-rows", default=20_000, type=int,
                    help="stars per chunk (default 20k)")
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    in_path = Path(args.input).expanduser()
    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")
    out_path = (Path(args.output) if args.output
                else in_path.with_name(in_path.stem + "_ages.parquet"))

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    bundle, model, guide = load_model(args.bundle)
    if args.debias and "affine_debias" not in bundle:
        sys.exit("--debias requested but the bundle has no 'affine_debias' "
                 "(legacy bundle; retrain or drop the flag)")

    pf = pq.ParquetFile(in_path)
    writer, n_done = None, 0
    try:
        for batch in pf.iter_batches(batch_size=args.batch_rows):
            chunk = batch.to_pandas()
            res = predict_chunk(chunk, bundle, model, guide, args.num_samples,
                                rng, debias=args.debias)
            table = pa.Table.from_pandas(res, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table.cast(writer.schema))
            n_done += len(res)
            print(f"  {n_done} stars done "
                  f"(chunk median age {res['age'].median():.2f} Gyr, "
                  f"median logAgeErr {res['logAgeErr'].median():.3f} dex)")
    finally:
        if writer is not None:
            writer.close()

    print(f"\nWrote {n_done} stars to {out_path}")


if __name__ == "__main__":
    main()
