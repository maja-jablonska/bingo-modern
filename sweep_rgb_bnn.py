#!/usr/bin/env python3
"""Hyperparameter sweep for the RGB stellar-age BNN, one config per process.

Mirrors train_pipeline_rgb.ipynb sections 3-6 exactly (same SEED, same
stratified split, same cleaning), trains one grid config, and evaluates on the
VAL split only -- test is never touched, so the winner can be promoted into the
notebook and evaluated on test exactly once there. No checkpoints are written;
each run emits a small JSON with config + val metrics.

Usage:
  python3 sweep_rgb_bnn.py --list                 # show the grid
  python3 sweep_rgb_bnn.py --index 7              # run config 7 (or $PBS_ARRAY_INDEX)
  python3 sweep_rgb_bnn.py --collect              # rank finished results
  python3 sweep_rgb_bnn.py --index 0 --smoke      # 1-minute smoke run

Submit the full sweep on Gadi with sweep_rgb_bnn.pbs.
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import prepare_dataset as prep

# --- must match train_pipeline_rgb.ipynb ------------------------------------
DATASET = os.environ.get("SWEEP_DATASET", os.path.expanduser(
    "~/scr_mk27/bulge-ages-and-orbits/data/merged_with_ages_raw.parquet"))
# RGB selection persisted by the Cannon notebook (adds classifier-accepted
# K2/TESS stars). Resolution of SWEEP_SELECTION_FILE:
#   unset/empty -> the default file if it exists, else seismic-only
#   "none"      -> FORCE seismic-only (Kepler EvoState==1), even if the
#                  default file exists -- use this to sweep the Kepler sample
#   a path      -> that file
_DEFAULT_SELECTION = os.path.expanduser(
    "~/scr_mk27/bulge-ages-and-orbits/data/rgb_selection_all_missions.parquet")
_sel_env = os.environ.get("SWEEP_SELECTION_FILE", "")
if _sel_env.lower() == "none":
    SELECTION_FILE = None
elif _sel_env:
    SELECTION_FILE = _sel_env
else:
    SELECTION_FILE = (_DEFAULT_SELECTION if os.path.exists(_DEFAULT_SELECTION)
                      else None)
SEED = 42
TEST_FRAC = 0.20
VAL_FRAC = 0.15
MAX_LOGAGE_ERR = 0.3   # dex; drop stars with less-precise Willett ages (None = off)
NUM_ITERATIONS = 20000
ES_PATIENCE = 8
ES_EVAL_EVERY = 5
FEATURES = prep.BASE_FEATURES + ["C_N", "M_H"]

# --- the grid: 3 x 2 x 2 x 2 x 2 = 48 configs --------------------------------
GRID = [dict(hidden_dim=h, initial_lr=lr, weight_clip=c, batch_size=b,
             intrinsic_prior=ip)
        for h, lr, c, b, ip in itertools.product(
            [32, 64, 128],
            [0.001, 0.0025],
            [10.0, 30.0],
            [32, 128],
            [(0.1, 0.3), (0.05, 0.2)])]


def load_and_split():
    """Load, select, clean, and split exactly like the notebook (sections 3-4)."""
    df = prep.load_willett_merged(DATASET, target_state="rgb",
                                  selection_file=SELECTION_FILE)
    df = (df.sort_values("SNR", ascending=False)
            .drop_duplicates(prep.ID_COL, keep="first")
            .reset_index(drop=True))
    df, _ = prep.clean_labels(df, "full catalog", drop_saturated=True)
    df = df.reset_index(drop=True)
    if MAX_LOGAGE_ERR is not None:
        df = df[df[prep.TARGET_ERR] <= MAX_LOGAGE_ERR].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    strata = pd.qcut(df[prep.TARGET], q=10, labels=False, duplicates="drop")
    train_idx, val_idx, test_idx = [], [], []
    for b in np.unique(strata):
        members = rng.permutation(np.flatnonzero(strata == b))
        n_test = int(round(TEST_FRAC * len(members)))
        n_val = int(round(VAL_FRAC * len(members)))
        test_idx.extend(members[:n_test])
        val_idx.extend(members[n_test:n_test + n_val])
        train_idx.extend(members[n_test + n_val:])
    train = df.iloc[sorted(train_idx)].reset_index(drop=True)
    val = df.iloc[sorted(val_idx)].reset_index(drop=True)
    # test is deliberately not returned: the sweep must never see it
    return train, val


def run_config(cfg, outdir, smoke=False):
    import torch
    from train_bnn import (set_seed, BayesianNeuralNetwork, train_smooth_bnn,
                           get_targeted_posterior_samples, device)

    train, val = load_and_split()
    train = prep.derive_features(train)
    val = prep.derive_features(val)
    stats = prep.fit_norm_stats(train, FEATURES)
    train = prep.apply_norm(train, FEATURES, stats)
    val = prep.apply_norm(val, FEATURES, stats)
    train, edges, _ = prep.add_sample_weights(
        train, prep.N_AGE_BINS, cfg["weight_clip"])

    feat_cols = [f"{f}_NORM" for f in FEATURES]
    err_cols = [f"{f}_ERR_NORM" for f in FEATURES]
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)
    Xt, Xe = t(train[feat_cols].values), t(train[err_cols].values)
    yt, ye = t(train[prep.TARGET].values), t(train[prep.TARGET_ERR].values)
    wt = t(train["train_weight"].values)
    Xv, Xve = t(val[feat_cols].values), t(val[err_cols].values)
    yv, yve = t(val[prep.TARGET].values), t(val[prep.TARGET_ERR].values)

    set_seed(SEED)
    model = BayesianNeuralNetwork(
        input_dim=len(FEATURES),
        hidden_dim=cfg["hidden_dim"],
        use_skip_connections=True,
        use_empirical_output_bias=True,
        use_leaky_relu=True,
        y_mean=float(yt.mean()), y_std=float(yt.std()),
        intrinsic_scatter_prior=cfg["intrinsic_prior"][0],
        intrinsic_scatter_prior_logstd=cfg["intrinsic_prior"][1],
    ).to(device)

    guide, losses = train_smooth_bnn(
        model, Xt, Xe, yt, ye,
        num_iterations=200 if smoke else NUM_ITERATIONS,
        initial_lr=cfg["initial_lr"],
        lr_decay_per_epoch=0.995,
        batch_size=cfg["batch_size"],
        seed=SEED,
        w_train=wt, minibatch_scale=True,
        X_val=Xv, X_err_val=Xve, y_val=yv, y_err_val=yve,
        early_stopping=not smoke, patience=ES_PATIENCE,
        eval_every=ES_EVAL_EVERY, restore_best=True,
    )

    # Val-only evaluation (test stays untouched).
    samples, mean_pred, model_unc, intrinsic = get_targeted_posterior_samples(
        model, guide, Xv, Xve, yve, num_samples=100 if smoke else 1000)
    pred = np.median(samples, axis=0)
    yv_np = yv.cpu().numpy()
    resid = yv_np - pred
    total_unc = np.sqrt(np.mean(model_unc, axis=0) ** 2
                        + np.mean(intrinsic) ** 2
                        + yve.cpu().numpy() ** 2)
    z = resid / total_unc
    val_hist = getattr(model, "val_losses_", [])
    result = {
        "config": {**cfg, "intrinsic_prior": list(cfg["intrinsic_prior"])},
        "seed": SEED, "features": FEATURES, "smoke": smoke,
        "n_train": int(len(train)), "n_val": int(len(val)),
        "best_val_elbo": float(min(v for _, v in val_hist)) if val_hist else None,
        "best_epoch": int(getattr(model, "best_epoch_", -1)),
        "epochs_run": len(losses),
        "val_corr": float(np.corrcoef(yv_np, pred)[0, 1]),
        "val_rms": float(np.sqrt(np.mean(resid ** 2))),
        "val_slope": float(np.polyfit(yv_np, pred, 1)[0]),
        "val_z_robust_width": float(1.4826 * np.median(np.abs(z - np.median(z)))),
        "mean_intrinsic_scatter": float(np.mean(intrinsic)),
    }
    return result, losses, val_hist


def log_to_wandb(index, result, losses, val_hist, outdir):
    """One wandb run per sweep config: hyperparameters as run config, the
    per-epoch train/val ELBO curves, and the final val metrics as the summary.

    Gadi compute nodes have no internet, so runs default to OFFLINE mode
    (override with WANDB_MODE=online off-cluster). Sync from a login node:
        wandb sync <outdir>/wandb/offline-run-*
    """
    try:
        import wandb
    except ImportError:
        print("wandb not installed -- skipping logging (pip install wandb)")
        return
    cfg = result["config"]
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "bingo-rgb-bnn-sweep"),
        name=(f"cfg{index:03d}-h{cfg['hidden_dim']}-lr{cfg['initial_lr']}"
              f"-clip{cfg['weight_clip']:g}-b{cfg['batch_size']}"
              f"-ip{cfg['intrinsic_prior'][0]:g}"),
        config={"index": index, **cfg, "seed": result["seed"],
                "features": result["features"], "smoke": result["smoke"]},
        mode=os.environ.get("WANDB_MODE", "offline"),
        dir=str(outdir),
    )
    val_by_epoch = dict(val_hist)
    for epoch, loss in enumerate(losses):
        rec = {"train/elbo": loss}
        if epoch in val_by_epoch:
            rec["val/elbo"] = val_by_epoch[epoch]
        run.log(rec, step=epoch)
    run.summary.update({k: v for k, v in result.items()
                        if k not in ("config", "features")})
    run.finish()
    print(f"wandb run logged ({run.settings.mode} mode): {run.name}")


def collect(outdir):
    rows = []
    for f in sorted(Path(outdir).glob("result_*.json")):
        r = json.loads(f.read_text())
        rows.append({"file": f.name, **r["config"],
                     "val_elbo": r["best_val_elbo"], "corr": r["val_corr"],
                     "rms": r["val_rms"], "slope": r["val_slope"],
                     "z_width": r["val_z_robust_width"],
                     "intr_scatter": r["mean_intrinsic_scatter"],
                     "best_ep": r["best_epoch"]})
    if not rows:
        print(f"No result_*.json files in {outdir}")
        return
    t = pd.DataFrame(rows).sort_values("val_elbo")
    print(f"{len(t)}/{len(GRID)} configs finished; ranked by best val ELBO "
          "(lower = better). Promote the winner into the notebook and evaluate "
          "on test there, once.")
    print(t.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int,
                    default=int(os.environ.get("PBS_ARRAY_INDEX", -1)))
    ap.add_argument("--outdir",
                    default=os.environ.get("SWEEP_OUTDIR", "sweep_results"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (200 iterations) to verify the plumbing")
    ap.add_argument("--no-wandb", action="store_true",
                    help="skip wandb logging for this run")
    args = ap.parse_args()

    if args.list:
        for i, cfg in enumerate(GRID):
            print(f"{i:3d}  {cfg}")
        return
    if args.collect:
        collect(args.outdir)
        return
    if not 0 <= args.index < len(GRID):
        sys.exit(f"--index must be 0..{len(GRID) - 1} "
                 f"(got {args.index}); use --list to see the grid")

    cfg = GRID[args.index]
    print(f"Running config {args.index}: {cfg}")
    print(f"Selection: {SELECTION_FILE or 'seismic-only (Kepler EvoState==1)'} | "
          f"outdir: {args.outdir}")
    result, losses, val_hist = run_config(cfg, args.outdir, smoke=args.smoke)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    out = outdir / f"result_{args.index:03d}.json"
    out.write_text(json.dumps({"index": args.index,
                               "selection_file": SELECTION_FILE,
                               **result}, indent=2))
    print(f"Wrote {out}")
    print(json.dumps(result, indent=2))
    if not args.no_wandb:
        log_to_wandb(args.index, result, losses, val_hist, outdir)


if __name__ == "__main__":
    main()
