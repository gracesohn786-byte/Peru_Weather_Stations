from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
MONTHLYFILE_CSV = DATA_DIR / "Monthly_LascarData_200.csv"
DERIVED_DIR = DATA_DIR / "derived"


ELEVATIONS_M: List[float] = [3850.0, 3955.0, 4122.0, 4355.0, 4560.0, 4760.0]
ELEVATION_TO_STATION_ID: Dict[float, str] = {
    3850.0: "LlanWX",
    3955.0: "LlanUp1",
    4122.0: "LlanUp2",
    4355.0: "LlanUp3",
    4560.0: "LlanUp4",
    4760.0: "LlanPort",
}


MISSING_MARKERS = {-99.99, -99.9, -99.0, -99}


def _to_float(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s


def load_monthlyfile_raw(path: Path = MONTHLYFILE_CSV) -> pd.DataFrame:
    """
    Read Monthly_LascarData_200.csv where:
    - Row 1 is reference text
    - Row 2 is the real header (with duplicates)
    """
    df = pd.read_csv(
        path,
        skiprows=[0],
        na_values=["", " ", "NA", "NaN", "nan", "-99.99", "-99.9", "-99", "-99.0"],
    )

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(list(MISSING_MARKERS), np.nan)

    return df


def parse_month_year(df: pd.DataFrame, col: str = "Month-YR") -> pd.DataFrame:
    out = df.copy()
    out["month_label"] = out[col].astype(str).str.strip()
    dt = pd.to_datetime(out["month_label"], format="%b-%y", errors="coerce")
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out = out.dropna(subset=["year", "month"])
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    return out


def extract_mean_blocks(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract monthly mean blocks by fixed column positions determined from the file:
    - Temperature means: columns 1..6
    - Dew point means: columns 9..14
    - RH means: columns 15..20
    These indices correspond to the header structure (see column inspection).
    """
    temp = df.iloc[:, 1:7].copy()
    dp = df.iloc[:, 9:15].copy()
    rh = df.iloc[:, 15:21].copy()

    temp.columns = [f"temp_{e:g}" for e in ELEVATIONS_M]
    dp.columns = [f"dewpoint_{e:g}" for e in ELEVATIONS_M]
    rh.columns = [f"rh_{e:g}" for e in ELEVATIONS_M]

    return temp, dp, rh


def extract_aux_vars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract non-elevation monthly variables selected for your analysis:
    - wind speed (two columns in file; keep both)
    - solar radiation
    - precipitation
    - summed and average discharge
    """
    cols = df.columns

    def _safe(colname: str) -> Optional[str]:
        return colname if colname in cols else None

    wind0 = _safe("WindSpeed(m/s)")
    wind1 = _safe("WindSpeed(m/s).1")
    solar = _safe("Solar Radiation (W/m2)")
    precip = _safe("Precipitation (mm)")
    summed_q = _safe("Summed Discharge")
    avg_q = _safe("Average Discharge")

    out = pd.DataFrame(index=df.index)

    if wind0:
        out["wind_m_s"] = _to_float(df[wind0])
    if wind1:
        out["wind_m_s_2"] = _to_float(df[wind1])
    if solar:
        out["solar_w_m2"] = _to_float(df[solar])
    if precip:
        out["precip_mm"] = _to_float(df[precip])
    if summed_q:
        out["summed_discharge"] = _to_float(df[summed_q])
    if avg_q:
        out["average_discharge"] = _to_float(df[avg_q])

    return out


def reshape_to_long(
    df_meta: pd.DataFrame,
    temp_block: pd.DataFrame,
    dp_block: pd.DataFrame,
    rh_block: pd.DataFrame,
    aux: pd.DataFrame,
) -> pd.DataFrame:
    base = df_meta[["month_label", "year", "month"]].copy()

    rows: List[pd.DataFrame] = []
    for elevation in ELEVATIONS_M:
        e_key = f"{elevation:g}"
        tmp = pd.DataFrame(
            {
                "month_label": base["month_label"].values,
                "year": base["year"].values,
                "month": base["month"].values,
                "elevation_m": elevation,
                "station_id": ELEVATION_TO_STATION_ID.get(elevation),
                "temp_c_monthly": _to_float(temp_block[f"temp_{e_key}"]),
                "dewpoint_c_monthly": _to_float(dp_block[f"dewpoint_{e_key}"]),
                "rh_pct_monthly": _to_float(rh_block[f"rh_{e_key}"]),
            }
        )
        rows.append(tmp)

    long_df = pd.concat(rows, ignore_index=True)

    aux_rep = aux.copy()
    aux_rep["row_id"] = np.arange(len(aux_rep))
    long_df["row_id"] = np.tile(np.arange(len(aux_rep)), len(ELEVATIONS_M))
    long_df = long_df.merge(aux_rep, on="row_id", how="left").drop(columns=["row_id"])

    return long_df


def apply_qc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["temp_c_monthly", "dewpoint_c_monthly"]:
        out.loc[(out[col] < -50) | (out[col] > 50), col] = np.nan

    out.loc[(out["rh_pct_monthly"] < 0) | (out["rh_pct_monthly"] > 100), "rh_pct_monthly"] = np.nan

    # Dew point should not exceed temperature
    mask = out["dewpoint_c_monthly"] > out["temp_c_monthly"]
    out.loc[mask, "dewpoint_c_monthly"] = np.nan

    for col in ["wind_m_s", "wind_m_s_2", "solar_w_m2", "precip_mm"]:
        if col in out.columns:
            out.loc[out[col] < 0, col] = np.nan

    # Drop rows where all core fields are missing
    core = ["temp_c_monthly", "dewpoint_c_monthly", "rh_pct_monthly"]
    out = out[~out[core].isna().all(axis=1)]

    return out


def write_outputs(df_long: pd.DataFrame, out_dir: Path = DERIVED_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "monthly_from_monthlyfile_clean.csv"
    df_long.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    if not MONTHLYFILE_CSV.exists():
        raise FileNotFoundError(f"Monthly file CSV not found at {MONTHLYFILE_CSV}")

    raw = load_monthlyfile_raw(MONTHLYFILE_CSV)
    meta = parse_month_year(raw, col="Month-YR")
    temp, dp, rh = extract_mean_blocks(meta)
    aux = extract_aux_vars(meta)

    long_df = reshape_to_long(meta, temp, dp, rh, aux)
    long_df = apply_qc(long_df)

    write_outputs(long_df, DERIVED_DIR)


if __name__ == "__main__":
    main()

