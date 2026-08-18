#!/usr/bin/env python
"""Train/test the bingo BNN on the homogenized shared dataset (stardata).

Consumes a dataset directory built by ``stardata.build_dataset`` so the BNN
sees exactly the same stars and split as the Cannon and Lux (its row is each
star's best-SNR observation, ``is_primary``). Trains with val-based early
stopping (as sweep_rgb_bnn.run_config does) and — unlike the sweep — evaluates
on the shared held-out TEST split, writing ``targeted_prediction_summary.csv``
in the notebook schema (APOGEE_ID + source + base columns) so
``plot_label_diagnostics.py`` / ``stardiag.load_bingo`` consume it directly.

Usage:
    python run_bnn_shared.py --dataset-dir <dir> [--outdir BNN_shared_output]
        [--hidden-dim 64] [--initial-lr 0.0025] [--weight-clip 10]
        [--batch-size 128] [--iterations 20000] [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

for _c in (HERE.parent / "stardiag", Path.home() / "code" / "stardiag",
           Path.home() / "scr_mk27" / "stardiag"):
    if (_c / "stardata.py").exists():
        sys.path.insert(0, str(_c))
        break
else:
    sys.exit("stardiag checkout (stardata.py) not found next to this repo")
import stardata  # noqa: E402
import prepare_dataset as prep  # noqa: E402

# Same catalogue labels the spectral methods train on: the BNN cannot see
# pixels, but it must not be denied information Cannon/Lux have, or the
# comparison measures the feature list rather than the method.
FEATURES = prep.BASE_FEATURES + ["C_N", "M_H", "LOGG_SEISMIC", "AL_FE"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--outdir", default="BNN_shared_output")
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--initial-lr", type=float, default=0.0025)
    ap.add_argument("--weight-clip", type=float, default=10.0,
                    help="only used with --sample-weights inverse-frequency")
    ap.add_argument("--sample-weights", default="none",
                    choices=["none", "inverse-frequency"],
                    help="'inverse-frequency' flattens the training age "
                         "distribution. The Cannon and Lux train on the "
                         "natural (old-heavy) prior, so flattening only the "
                         "BNN gives it a different amount of shrinkage and "
                         "shows up as star-by-star disagreement. Default "
                         "'none' keeps all three priors identical.")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--intrinsic-prior", type=float, nargs=2,
                    default=(0.1, 0.3))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input-err-systematics", default="on",
                    choices=["on", "off"],
                    help="add stardata.LABEL_SYST in quadrature to the input "
                         "feature errors. The catalogue's quoted errors are "
                         "formal precisions (median [Fe/H] error 0.001 dex), "
                         "so with 'off' the BNN is told its inputs are "
                         "essentially exact and the intrinsic_scatter term "
                         "absorbs the unmodelled input noise.")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run for wiring checks")
    args = ap.parse_args()

    import torch
    from train_bnn import (set_seed, BayesianNeuralNetwork, train_smooth_bnn,
                           get_targeted_posterior_samples, device)

    stars, manifest = stardata.load_stars(args.dataset_dir)
    frame = stardata.to_bingo_frame(
        stars, err_systematics=args.input_err_systematics == "on")
    frame, report = prep.clean_labels(frame, "shared", drop_saturated=True)
    if report.get("n_dropped", 0):
        print(f"warning: clean_labels dropped rows on a shared dataset "
              f"(should be 0): {report}")
    frame = prep.derive_features(frame)

    # clean_labels only guards prep.BASE_FEATURES, so the labels added here
    # to match the spectral methods would otherwise reach SVI unchecked
    needed = [c for f in FEATURES for c in (f, f"{f}_ERR")]
    missing = [c for c in needed
               if c not in frame.columns
               or not np.isfinite(frame[c].to_numpy(float)).all()]
    if missing:
        sys.exit(f"features absent or non-finite in the shared dataset: "
                 f"{missing} — rebuild the dataset with --rebuild")

    parts = {s: frame[frame["split"] == s].reset_index(drop=True)
             for s in ("train", "val", "test")}
    print({k: len(v) for k, v in parts.items()})

    stats = prep.fit_norm_stats(parts["train"], FEATURES)
    parts = {k: prep.apply_norm(v, FEATURES, stats) for k, v in parts.items()}
    if args.sample_weights == "inverse-frequency":
        parts["train"], edges, _ = prep.add_sample_weights(
            parts["train"], prep.N_AGE_BINS, args.weight_clip)
    else:
        parts["train"] = parts["train"].assign(train_weight=1.0)

    feat_cols = [f"{f}_NORM" for f in FEATURES]
    err_cols = [f"{f}_ERR_NORM" for f in FEATURES]
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32),
                                  device=device)

    def tensors(d):
        return (t(d[feat_cols].values), t(d[err_cols].values),
                t(d[prep.TARGET].values), t(d[prep.TARGET_ERR].values))

    Xt, Xe, yt, ye = tensors(parts["train"])
    Xv, Xve, yv, yve = tensors(parts["val"])
    Xs, Xse, ys, yse = tensors(parts["test"])
    wt = t(parts["train"]["train_weight"].values)

    set_seed(args.seed)
    model = BayesianNeuralNetwork(
        input_dim=len(FEATURES), hidden_dim=args.hidden_dim,
        use_skip_connections=True, use_empirical_output_bias=True,
        use_leaky_relu=True, y_mean=float(yt.mean()), y_std=float(yt.std()),
        intrinsic_scatter_prior=args.intrinsic_prior[0],
        intrinsic_scatter_prior_logstd=args.intrinsic_prior[1],
    ).to(device)

    guide, losses = train_smooth_bnn(
        model, Xt, Xe, yt, ye,
        num_iterations=200 if args.smoke else args.iterations,
        initial_lr=args.initial_lr, lr_decay_per_epoch=0.995,
        batch_size=args.batch_size, seed=args.seed,
        w_train=wt, minibatch_scale=True,
        X_val=Xv, X_err_val=Xve, y_val=yv, y_err_val=yve,
        early_stopping=not args.smoke, patience=8, eval_every=5,
        restore_best=True)

    n_samp = 100 if args.smoke else args.num_samples
    samples, mean_pred, model_unc, intrinsic = get_targeted_posterior_samples(
        model, guide, Xs, Xse, yse, num_samples=n_samp)

    ys_np, yse_np = ys.cpu().numpy(), yse.cpu().numpy()
    pred_median = np.median(samples, axis=0)
    model_unc_mean = np.mean(model_unc, axis=0)
    intrinsic_mean = float(np.mean(intrinsic))
    total_unc = np.sqrt(model_unc_mean ** 2 + intrinsic_mean ** 2
                        + yse_np ** 2)
    residual = ys_np - pred_median            # bingo's own sign convention

    test = parts["test"]
    summary = pd.DataFrame({
        "row_id": test["row_id"].values,
        "APOGEE_ID": test["APOGEE_ID"].values,
        "source": test.get("source"),
        "split": "test",
        "is_primary": True,
        "observational_uncertainty": yse_np,
        "model_uncertainty": model_unc_mean,
        "intrinsic_scatter": intrinsic_mean,
        "total_predictive_uncertainty": total_unc,
        "pred_median": pred_median,
        "pred_mean_only": np.asarray(mean_pred).mean(axis=0)
        if np.ndim(mean_pred) > 1 else np.asarray(mean_pred),
        "true_age": ys_np,
        "residual": residual,
        "normalized_residual": residual / total_unc,
    })

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "targeted_prediction_summary.csv", index=False)

    import pyro
    pyro.get_param_store().save(str(outdir / "targeted_bnn_params.pth"))

    bias = float(np.median(residual))
    sig = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    run_info = {
        "dataset_dir": str(args.dataset_dir),
        "config": {k: getattr(args, k) for k in
                   ("hidden_dim", "initial_lr", "weight_clip",
                    "sample_weights", "input_err_systematics", "batch_size",
                    "iterations", "seed", "smoke")},
        "features": FEATURES,
        "n": {k: int(len(v)) for k, v in parts.items()},
        "epochs_run": len(losses),
        "test_bias": bias, "test_scatter": sig,
        "test_rms": float(np.sqrt(np.mean(residual ** 2))),
        "mean_intrinsic_scatter": intrinsic_mean,
    }
    (outdir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"test bias={bias:+.4f} scatter={sig:.4f} -> "
          f"{outdir / 'targeted_prediction_summary.csv'}")


if __name__ == "__main__":
    main()
