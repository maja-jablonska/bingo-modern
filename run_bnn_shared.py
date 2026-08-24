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
# Spectroscopic LOGG (in BASE_FEATURES), not LOGG_SEISMIC. The seismic value
# is an INPUT here, so requiring it would confine the BNN to stars with
# asteroseismology -- which excludes every large APOGEE sample.
FEATURES = prep.BASE_FEATURES + ["C_N", "M_H", "AL_FE"]


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
    ap.add_argument("--apply-to", default=None,
                    help="parquet of ASPCAP parameters for stars OUTSIDE the "
                         "labelled sample. Now that LOGG_SEISMIC is out of "
                         "FEATURES the BNN needs no asteroseismology, so it "
                         "can run wherever the catalogue parameters exist. "
                         "Stars beyond the training parameter range are "
                         "flagged in_training_range=False.")
    ap.add_argument("--apply-out", default=None)
    ap.add_argument("--logg-seismic-from", default=None,
                    help="parquet of spectral predictions supplying "
                         "LOGG_SEISMIC in place of the seismic value. The "
                         "BNN takes logg_seismic as an INPUT, so without "
                         "numax it cannot run at all; Cannon and Lux instead "
                         "PREDICT it from the spectrum. Substituting their "
                         "prediction cascades the two and unlocks the BNN "
                         "off the seismic sample -- at the cost of "
                         "propagating the spectral model's error into the "
                         "BNN's input. Point this at out-of-sample "
                         "predictions or the test becomes circular.")
    ap.add_argument("--err-inflation", type=float, default=1.0,
                    help="scale the assumed input sigma_x. LatentNN's "
                         "correction strength is set by sigma_x, and "
                         "over-stating it over-corrects (Ting 2026 sec 7.4). "
                         "The abundance systematics are upper bounds, so "
                         "values below 1 test whether the observed slope of "
                         "~1.04 is over-correction from an inflated sigma_x.")
    ap.add_argument("--latent-inputs", default="off", choices=["on", "off"],
                    help="LatentNN (Ting 2026, arXiv:2512.23138): optimise a "
                         "latent true value per input alongside the weights, "
                         "instead of feeding the network the noisy "
                         "observations. Corrects attenuation bias, which "
                         "compresses the predicted range by "
                         "1/(1+(sigma_x/sigma_range)^2). On this sample the "
                         "age-carrying features [C/N] and [Al/Fe] sit at "
                         "SNR_x ~ 2.35, so the predicted attenuation (~0.85) "
                         "is about the size of the BNN's measured residual "
                         "shrinkage (0.91).")
    ap.add_argument("--var-prior-scale", type=float, default=None,
                    help="prior width for the VARIANCE network's weights "
                         "(default: prior-scale * 0.3). The per-star sigma "
                         "currently spans only p90/p10 = 1.47, i.e. the model "
                         "is near-homoscedastic, which no tail shape can "
                         "calibrate; loosening this is the candidate fix.")
    ap.add_argument("--likelihood", default="student_t",
                    choices=["normal", "student_t"],
                    help="the RGB age residuals are strongly leptokurtic "
                         "(excess kurtosis ~17), so a Normal has to widen "
                         "sigma to cover the tails and then over-covers the "
                         "core (85% inside 1 sigma against a nominal 68%). "
                         "'student_t' absorbs the tails in its shape. NOTE "
                         "this also robustifies TRAINING: outlying stars are "
                         "down-weighted, so the fit changes, not just the "
                         "reported uncertainty.")
    ap.add_argument("--student-t-df", type=float, default=None,
                    help="pin the degrees of freedom; default learns it as a "
                         "global latent (observed tails imply nu ~ 4)")
    ap.add_argument("--mode", choices=["holdout", "oof"], default="holdout",
                    help="'oof' additionally refits per fold over the "
                         "non-test stars, so EVERY star gets an "
                         "out-of-sample prediction rather than the 20% in "
                         "the test split. Mirrors the other two runners.")
    ap.add_argument("--early-stopping", default="on", choices=["on", "off"],
                    help="'off' runs the full --iterations budget. Early "
                         "stopping fires around epoch 340-430 of a ~600-epoch "
                         "budget with the LR already decayed to ~13%, so the "
                         "weight posterior may simply not have finished "
                         "contracting -- which would show up as an inflated "
                         "model_uncertainty and over-wide predictions.")
    ap.add_argument("--prior-scale", type=float, default=0.5,
                    help="width of the N(0,s) prior on every network weight. "
                         "The reported model (epistemic) uncertainty came out "
                         "at 0.18 dex against an actual residual scatter of "
                         "0.118 on the RGB benchmark, i.e. over-dispersed; "
                         "try 0.25-0.5.")
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
        stars, err_systematics=args.input_err_systematics == "on",
        err_inflation=args.err_inflation)
    # None: the shared dataset already decided membership, including whether
    # ages above 13.8 Gyr are kept. Dropping or capping them again here would
    # silently override that.
    frame, report = prep.clean_labels(frame, "shared", drop_saturated=None)
    if report.get("n_dropped", 0):
        print(f"warning: clean_labels dropped rows on a shared dataset "
              f"(should be 0): {report}")
    frame = prep.derive_features(frame)

    # clean_labels only guards prep.BASE_FEATURES, so the labels added here
    # to match the spectral methods would otherwise reach SVI unchecked
    needed = [c for f in FEATURES for c in (f, f"{f}_ERR")]
    if args.logg_seismic_from:
        src = pd.read_parquet(args.logg_seismic_from)
        pc = next((c for c in ("pred_logg_seismic", "logg_seismic_pred")
                   if c in src.columns), None)
        ec = next((c for c in ("pred_err_logg_seismic", "logg_seismic_pred_err")
                   if c in src.columns), None)
        if pc is None:
            sys.exit(f"{args.logg_seismic_from}: no predicted logg_seismic")
        key = "row_id" if ("row_id" in src.columns
                           and "row_id" in frame.columns) else "APOGEE_ID"
        src = src.loc[:, [key, pc] + ([ec] if ec else [])]
        if "is_primary" in pd.read_parquet(args.logg_seismic_from).columns:
            ip = pd.read_parquet(args.logg_seismic_from, columns=[key, "is_primary"])
            src = src.merge(ip, on=key).query("is_primary").drop(columns="is_primary")
        src = src.drop_duplicates(key)
        before = frame["LOGG_SEISMIC"].to_numpy(float).copy()
        frame = frame.merge(src, on=key, how="left", suffixes=("", "_src"))
        got = frame[pc].to_numpy(float)
        n_bad = int((~np.isfinite(got)).sum())
        if n_bad:
            print(f"warning: {n_bad} stars had no predicted logg_seismic; "
                  f"falling back to the seismic value for those")
            got = np.where(np.isfinite(got), got, before)
        frame["LOGG_SEISMIC"] = got
        if ec:
            e = frame[ec].to_numpy(float)
            frame["LOGG_SEISMIC_ERR"] = np.where(np.isfinite(e) & (e > 0), e,
                                                 frame["LOGG_SEISMIC_ERR"])
        frame = frame.drop(columns=[c for c in (pc, ec) if c in frame.columns])
        d = got - before
        print(f"LOGG_SEISMIC substituted from {args.logg_seismic_from}: "
              f"median offset {np.median(d[np.isfinite(d)]):+.4f}, "
              f"scatter {1.4826*np.median(np.abs(d[np.isfinite(d)] - np.median(d[np.isfinite(d)]))):.4f} dex")

    missing = [c for c in needed
               if c not in frame.columns
               or not np.isfinite(frame[c].to_numpy(float)).all()]
    if missing:
        sys.exit(f"features absent or non-finite in the shared dataset: "
                 f"{missing} — rebuild the dataset with --rebuild")

    parts = {s: frame[frame["split"] == s].reset_index(drop=True)
             for s in ("train", "val", "test")}
    print({k: len(v) for k, v in parts.items()})

    feat_cols = [f"{f}_NORM" for f in FEATURES]
    err_cols = [f"{f}_ERR_NORM" for f in FEATURES]
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32),
                                  device=device)

    def fit_predict(train_df, val_df, pred_df, verbose=True):
        """Train on train_df (early-stopping on val_df) and predict pred_df.

        Normalisation stats are refit on train_df every time, so an OOF fold
        never sees statistics derived from the stars it predicts.
        """
        stats = prep.fit_norm_stats(train_df, FEATURES)
        tr, va, pr = (prep.apply_norm(d, FEATURES, stats)
                      for d in (train_df, val_df, pred_df))
        if args.sample_weights == "inverse-frequency":
            tr, _, _ = prep.add_sample_weights(tr, prep.N_AGE_BINS,
                                               args.weight_clip)
        else:
            tr = tr.assign(train_weight=1.0)

        def tensors(d):
            return (t(d[feat_cols].values), t(d[err_cols].values),
                    t(d[prep.TARGET].values), t(d[prep.TARGET_ERR].values))

        Xt, Xe, yt, ye = tensors(tr)
        Xv, Xve, yv, yve = tensors(va)
        Xs, Xse, ys, yse = tensors(pr)
        wt = t(tr["train_weight"].values)

        set_seed(args.seed)
        model = BayesianNeuralNetwork(
            input_dim=len(FEATURES), hidden_dim=args.hidden_dim,
            use_skip_connections=True, use_empirical_output_bias=True,
            use_leaky_relu=True, y_mean=float(yt.mean()),
            y_std=float(yt.std()),
            intrinsic_scatter_prior=args.intrinsic_prior[0],
            intrinsic_scatter_prior_logstd=args.intrinsic_prior[1],
            prior_scale=args.prior_scale,
            var_prior_scale=args.var_prior_scale,
            likelihood=args.likelihood,
            student_t_df=args.student_t_df,
            latent_inputs=args.latent_inputs == "on",
        ).to(device)

        guide, losses = train_smooth_bnn(
            model, Xt, Xe, yt, ye,
            num_iterations=200 if args.smoke else args.iterations,
            initial_lr=args.initial_lr, lr_decay_per_epoch=0.995,
            batch_size=args.batch_size, seed=args.seed,
            w_train=wt, minibatch_scale=True,
            X_val=Xv, X_err_val=Xve, y_val=yv, y_err_val=yve,
            early_stopping=(args.early_stopping == "on") and not args.smoke,
            patience=8, eval_every=5, restore_best=True)

        n_samp = 100 if args.smoke else args.num_samples
        samples, mean_pred, model_unc, intrinsic = \
            get_targeted_posterior_samples(model, guide, Xs, Xse, yse,
                                           num_samples=n_samp)
        ys_np, yse_np = ys.cpu().numpy(), yse.cpu().numpy()
        pred_median = np.median(samples, axis=0)
        model_unc_mean = np.mean(model_unc, axis=0)
        intrinsic_mean = float(np.mean(intrinsic))
        # For the Normal this is the predictive sigma. For the Student-t it
        # is the SCALE: the core width. The distribution's own standard
        # deviation is scale * sqrt(nu/(nu-2)), reported alongside so the
        # two are never confused -- a t scale compared against a Normal
        # sigma is not like for like, which is exactly the mistake that made
        # the Normal fit look twice as over-conservative as it was.
        total_unc = np.sqrt(model_unc_mean ** 2 + intrinsic_mean ** 2
                            + yse_np ** 2)
        nu = getattr(model, "student_t_df_fitted_", args.student_t_df)
        if args.likelihood == "student_t" and nu and nu > 2:
            total_std_equiv = total_unc * np.sqrt(nu / (nu - 2.0))
        else:
            total_std_equiv = total_unc
        residual = ys_np - pred_median        # bingo's own sign convention
        out = pd.DataFrame({
            "row_id": pred_df["row_id"].values,
            "APOGEE_ID": pred_df["APOGEE_ID"].values,
            "source": pred_df.get("source"),
            "split": pred_df["split"].values,
            "is_primary": True,
            "observational_uncertainty": yse_np,
            "model_uncertainty": model_unc_mean,
            "intrinsic_scatter": intrinsic_mean,
            "total_predictive_uncertainty": total_unc,
            "total_predictive_std": total_std_equiv,
            "pred_median": pred_median,
            "pred_mean_only": np.asarray(mean_pred).mean(axis=0)
            if np.ndim(mean_pred) > 1 else np.asarray(mean_pred),
            "true_age": ys_np,
            "residual": residual,
            "normalized_residual": residual / total_unc,
        })
        return out, len(losses), intrinsic_mean, nu, model, guide, stats

    (summary, epochs_run, intrinsic_mean, nu,
     model_final, guide_final, stats) = fit_predict(
        parts["train"], parts["val"], parts["test"])

    if args.mode == "oof":
        # Every non-test star also gets a prediction from a model that never
        # saw it: fold k is predicted, fold (k+1) serves as its early-stopping
        # validation set, the remaining folds train. Test rows keep the
        # holdout predictions above.
        nontest = frame[frame["split"] != "test"].reset_index(drop=True)
        folds = sorted({int(f) for f in nontest["fold"] if f >= 0})
        extra = []
        for i, k in enumerate(folds):
            vk = folds[(i + 1) % len(folds)]
            pr = nontest[nontest.fold == k]
            va = nontest[nontest.fold == vk]
            tr = nontest[~nontest.fold.isin([k, vk])]
            if not len(pr) or not len(va) or not len(tr):
                continue
            print(f"\nOOF fold {k}: train {len(tr)}, val {len(va)}, "
                  f"predict {len(pr)}")
            o = fit_predict(tr, va, pr)[0]
            extra.append(o)
        if extra:
            summary = pd.concat([summary] + extra, ignore_index=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "targeted_prediction_summary.csv", index=False)

    if args.apply_to:
        print(f"\napplying the trained model to {args.apply_to}")
        import pyarrow.dataset as _ds
        want = ["raw_teff","raw_e_teff","raw_logg","raw_e_logg","raw_fe_h",
                "raw_e_fe_h","raw_mg_h","raw_e_mg_h","raw_c_h","raw_e_c_h",
                "raw_n_h","raw_e_n_h","raw_al_h","raw_e_al_h","snr"]
        dset = _ds.dataset(args.apply_to)
        idc = next((c for c in ("APOGEE_ID","sdss4_apogee_id")
                    if c in dset.schema.names), None)
        cols = [idc] + [c for c in want if c in dset.schema.names]
        ext = dset.to_table(columns=cols).to_pandas()
        ext = stardata.derive_columns(ext.copy())
        ext["APOGEE_ID"] = ext[idc].astype(str)
        for c in ("is_primary", "split", "fold", "row_id"):
            ext[c] = True if c == "is_primary" else 0
        ext["age_L"] = np.nan; ext["log_age_L"] = np.nan
        ext["e_log_age_L"] = 0.1
        ef = stardata.to_bingo_frame(
            ext, err_systematics=args.input_err_systematics == "on",
            err_inflation=args.err_inflation)
        ef = prep.derive_features(ef)
        ok = np.all([np.isfinite(ef[f].to_numpy(float)) for f in FEATURES], axis=0)
        print(f"  {len(ef)} stars, {int(ok.sum())} with all features finite")
        efn = prep.apply_norm(ef[ok].reset_index(drop=True), FEATURES, stats)
        Xa = t(efn[feat_cols].values); Xae = t(efn[err_cols].values)
        yz = t(np.zeros(len(efn), np.float32))
        sm, mp, mu_, it_ = get_targeted_posterior_samples(
            model_final, guide_final, Xa, Xae, yz,
            num_samples=100 if args.smoke else args.num_samples)
        pm = np.median(sm, axis=0)
        tot = np.sqrt(np.mean(mu_, axis=0)**2 + float(np.mean(it_))**2)
        ao = pd.DataFrame({"star_id": efn["APOGEE_ID"].values,
                           "pred_log_age": pm, "pred_sigma": tot})
        ao["in_training_range"] = stardata.coverage_flag(
            efn.rename(columns={"TEFF":"raw_teff","LOGG":"raw_logg",
                                "FE_H":"raw_fe_h"}),
            stars[stars["split"] == "train"],
            labels=["raw_teff","raw_logg","raw_fe_h"])
        apath = Path(args.apply_out or (outdir / "bnn_applied.parquet"))
        ao.to_parquet(apath)
        n_out = int((~ao.in_training_range).sum())
        print(f"  wrote {len(ao)} predictions to {apath}")
        print(f"  {n_out} ({100*n_out/max(len(ao),1):.1f}%) outside the "
              f"training parameter range")

    import pyro
    pyro.get_param_store().save(str(outdir / "targeted_bnn_params.pth"))

    # test rows only: with --mode oof the summary also holds fold
    # predictions, which must not enter the held-out test metrics
    tst = summary[summary["split"] == "test"]
    residual = tst["residual"].to_numpy(float)
    bias = float(np.median(residual))
    sig = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    run_info = {
        "dataset_dir": str(args.dataset_dir),
        "config": {k: getattr(args, k) for k in
                   ("hidden_dim", "initial_lr", "weight_clip",
                    "sample_weights", "input_err_systematics", "prior_scale",
                    "var_prior_scale", "likelihood", "student_t_df",
                    "latent_inputs", "err_inflation", "logg_seismic_from",
                    "early_stopping", "mode",
                    "batch_size", "iterations", "seed", "smoke")},
        "student_t_df_fitted": (float(nu) if args.likelihood == "student_t"
                                and nu else None),
        "features": FEATURES,
        "n": {k: int(len(v)) for k, v in parts.items()},
        "n_predicted": int(len(summary)),
        "epochs_run": epochs_run,
        "test_bias": bias, "test_scatter": sig,
        "test_rms": float(np.sqrt(np.mean(residual ** 2))),
        "mean_intrinsic_scatter": intrinsic_mean,
    }
    (outdir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"test bias={bias:+.4f} scatter={sig:.4f} -> "
          f"{outdir / 'targeted_prediction_summary.csv'}")


if __name__ == "__main__":
    main()
