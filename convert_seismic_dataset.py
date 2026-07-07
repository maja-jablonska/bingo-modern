"""
Convert the AnniesLasso dataset_prep.ipynb outputs to the BNN's expected format.

Input:  rgb_clean.parquet / rc_clean.parquet (label-only copies fetched from Gadi:
        train_data/{rgb,rc}_clean_labels.parquet). These carry the Astra "raw_*"
        label convention and Dnu-scaling asteroseismic ages.

Output: one unsplit CSV per input in the schema prepare_dataset.py / train_bnn.py
        expect -- ID_COL=APOGEE_ID, target logAge/logAgeErr (+ age in Gyr), and
        FOO / FOO_ERR feature pairs. Feed it to train_pipeline.ipynb by setting
        DATASET to the output path (the notebook's lowercase RENAME map is then a
        no-op, and it re-derives `age` from logAge, consistently with ours).

Usage:
    python convert_seismic_dataset.py                          # both states
    python convert_seismic_dataset.py train_data/rgb_clean_labels.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUTS = [
    "./train_data/rgb_clean_labels.parquet",
    "./train_data/rc_clean_labels.parquet",
]

# Source column -> expected name. Value/error pairs follow prepare_dataset.py's
# FOO / FOO_ERR convention; BASE_FEATURES there is LOGG TEFF MG_FE FE_H C_FE N_FE,
# the rest are carried along for experiments with a wider label vector.
COLUMN_MAP = {
    "APOGEE_ID": "APOGEE_ID",
    "raw_teff": "TEFF", "raw_e_teff": "TEFF_ERR",
    "raw_logg": "LOGG", "raw_e_logg": "LOGG_ERR",
    "raw_fe_h": "FE_H", "raw_e_fe_h": "FE_H_ERR",
    "mg_fe": "MG_FE", "e_mg_fe": "MG_FE_ERR",
    "c_fe": "C_FE", "e_c_fe": "C_FE_ERR",
    "n_fe": "N_FE", "e_n_fe": "N_FE_ERR",
    "al_fe": "AL_FE", "e_al_fe": "AL_FE_ERR",
    "ca_fe": "CA_FE", "e_ca_fe": "CA_FE_ERR",
    "o_fe": "O_FE", "e_o_fe": "O_FE_ERR",
    "ce_fe": "CE_FE", "e_ce_fe": "CE_FE_ERR",
    "nd_fe": "ND_FE", "e_nd_fe": "ND_FE_ERR",
    "log_age_Dnu": "logAge", "log_age_Dnu_err": "logAgeErr",
    "age_Dnu": "age",
    # Extra context, not consumed by the pipeline
    "KIC_ID": "KIC_ID",
    "snr_x": "SNR",
    "EvoState_pred": "EVO_STATE",
}

REQUIRED = ["APOGEE_ID", "age", "logAge", "logAgeErr",
            "LOGG", "LOGG_ERR", "TEFF", "TEFF_ERR", "MG_FE", "MG_FE_ERR",
            "FE_H", "FE_H_ERR", "C_FE", "C_FE_ERR", "N_FE", "N_FE_ERR"]


def convert(path):
    path = Path(path)
    df = pd.read_parquet(path)

    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name}: missing source columns {missing}")

    out = df[list(COLUMN_MAP)].rename(columns=COLUMN_MAP).reset_index(drop=True)

    n_nan = out[REQUIRED].isna().any(axis=1).sum()
    out_path = path.with_name(path.name.replace("_labels", "").replace(".parquet", "_bnn.csv"))
    out.to_csv(out_path, index=False)

    print(f"{path.name} -> {out_path}")
    print(f"  {len(out)} stars, {out.shape[1]} columns"
          f" ({n_nan} rows have NaNs in required columns; clean_labels drops them)")
    print(f"  logAge range [{out['logAge'].min():.3f}, {out['logAge'].max():.3f}],"
          f" median logAgeErr {out['logAgeErr'].median():.3f} dex")
    return out_path


if __name__ == "__main__":
    inputs = sys.argv[1:] or DEFAULT_INPUTS
    for p in inputs:
        convert(p)
