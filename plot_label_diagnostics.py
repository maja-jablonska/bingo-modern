#!/usr/bin/env python
"""Standard label-diagnostic plots for the bingo BNN, in the shared style.

Produces predicted-vs-true logAge coloured by the predictive uncertainty,
residual-systematics grids (offsets vs stellar parameters), and a Kiel map
coloured by residual, via the shared ``stardiag`` module (sibling checkout).

Usage:
    python plot_label_diagnostics.py                       # auto-detect run
    python plot_label_diagnostics.py --output-dir BNN_targeted_output_RGB \
        --selection-file ~/scr_mk27/.../rgb_selection_all_missions.parquet
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

for _c in (HERE.parent / "stardiag", Path.home() / "code" / "stardiag",
           Path.home() / "scr_mk27" / "stardiag"):
    if (_c / "stardiag.py").exists():
        sys.path.insert(0, str(_c))
        break
else:
    sys.exit("stardiag module not found (expected a 'stardiag' checkout "
             "next to this repo)")
import stardiag  # noqa: E402

OUTPUT_CANDIDATES = ["BNN_targeted_output_RGB_kepler", "BNN_targeted_output_RGB",
                     "BNN_targeted_output"]
DATASET_CANDIDATES = [
    "~/scr_mk27/bulge-ages-and-orbits/data/merged_with_ages_raw.parquet",
]
DR17_TEST_CSV = HERE / "test_data" / "TestOriginalNorm_dr17_clean.csv"


def load_params_for(summary_csv, dataset=None, selection_file=None):
    """Stellar-parameter table matching the prediction summary."""
    import pandas as pd
    header = pd.read_csv(summary_csv, nrows=0).columns
    if "APOGEE_ID" in header:  # RGB runs: re-join the all-missions catalogue
        import prepare_dataset as prep
        dataset = next((p for p in ([dataset] if dataset else []) +
                        DATASET_CANDIDATES
                        if p and Path(os.path.expanduser(p)).exists()), None)
        if dataset is None:
            print("warning: merged catalogue not found; plotting without "
                  "stellar parameters", file=sys.stderr)
            return None
        cat = prep.load_willett_merged(os.path.expanduser(dataset),
                                       target_state="rgb",
                                       selection_file=selection_file)
        cat = (cat.sort_values("SNR", ascending=False)
                  .drop_duplicates(prep.ID_COL, keep="first"))
        return prep.derive_features(cat)
    # legacy DR17 run: no IDs, positional row alignment with the test set
    return pd.read_csv(DR17_TEST_CSV)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", help="BNN run dir holding "
                    "targeted_prediction_summary.csv (default: auto-detect)")
    ap.add_argument("--dataset", help="merged_with_ages_raw.parquet override")
    ap.add_argument("--selection-file", help="rgb selection parquet "
                    "(is_rgb flags for K2/TESS)")
    ap.add_argument("--plot-dir", help="where to write PNGs "
                    "(default: <output-dir>/diagnostic_plots)")
    args = ap.parse_args()

    run_dir = (Path(args.output_dir) if args.output_dir else
               next((HERE / d for d in OUTPUT_CANDIDATES
                     if (HERE / d / "targeted_prediction_summary.csv").exists()),
                    None))
    if run_dir is None or not (run_dir / "targeted_prediction_summary.csv").exists():
        sys.exit(f"no targeted_prediction_summary.csv found in "
                 f"{args.output_dir or OUTPUT_CANDIDATES}")
    summary_csv = run_dir / "targeted_prediction_summary.csv"
    print(f"predictions: {summary_csv}")

    params = load_params_for(summary_csv, args.dataset, args.selection_file)
    spec = stardiag.load_bingo(summary_csv, params_table=params)

    outdir = Path(args.plot_dir) if args.plot_dir else run_dir / "diagnostic_plots"
    for p in stardiag.make_all(spec, outdir):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
