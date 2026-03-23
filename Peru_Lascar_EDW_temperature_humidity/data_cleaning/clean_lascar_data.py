from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
HOURLY_CSV = DATA_DIR / "Lascar_T_RH_DP_Langanuco_Hourly_2006_2025.csv"
MONTHLY_CSV = DATA_DIR / "Monthly_LascarData_200.csv"
DERIVED_DIR = DATA_DIR / "derived"


# Mapping of station label → elevation (m) taken from the metadata row
STATION_ELEVATION_M: Dict[str, float] = {
    "LlanWX": 3850.0,
    "LlanUp1": 3955.0,
    "LlanUp2": 4122.0,
    "LlanUp3": 4355.0,
    "LlanUp4": 4560.0,
    "LlanPort": 4760.0,
}


AIR_TEMP_COLS: Dict[str, str] = {
    "LlanWX": "AirTempLlanWX",
    "LlanUp1": "AirTempLlanUp1LAS",
    "LlanUp2": "AirTempLlanUp2SolCorrected",
    "LlanUp3": "AirTempLlanUp3LAS",
    "LlanUp4": "AirTempLlanUp4LASolCorrected",
    "LlanPort": "AirTempLlanPort",
}

DEWPOINT_COLS: Dict[str, str] = {
    "LlanWX": "DewPtLlanWX",
    "LlanUp1": "DewPtLlanUp1LAS",
    "LlanUp2": "DewPtLlanUp2SolCorrected",
    "LlanUp3": "DewPtLlanUp3LAS",
    "LlanUp4": "DewPtLlanUp4LASolCorrected",
    "LlanPort": "DewPtLlanPort",
}

RH_COLS: Dict[str, str] = {
    "LlanWX": "RHLlanWX",
    "LlanUp1": "RHLlanUp1LAS",
    "LlanUp2": "RHLlanUp2SolCorrected",
    "LlanUp3": "RHLlanUp3LAS",
    "LlanUp4": "RHLlanUp4LASolCorrected",
    "LlanPort": "RHLlanPort",
}


@dataclass
class LascarRecord:
    station_id: str
    elevation_m: float
    datetime: pd.Timestamp
    temp_c: float
    dewpoint_c: float
    rh_pct: float


def load_hourly_raw(path: Path = HOURLY_CSV) -> pd.DataFrame:
    """Load the raw hourly CSV, skipping the second metadata header row."""

    df = pd.read_csv(
        path,
        header=0,
        skiprows=[1],
        na_values=["NA", "NaN", "nan", "", " ", -999, -999.0],
    )

    # Standardize the datetime column name and parse
    if "Time" not in df.columns:
        raise ValueError("Expected 'Time' column in hourly Lascar file.")

    df = df.rename(columns={"Time": "datetime_raw"})

    df["datetime"] = pd.to_datetime(
        df["datetime_raw"],
        errors="coerce",
        dayfirst=True,
    )

    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime")

    return df


def to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def apply_qc_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic physical range filters to T, RH, dew point."""

    all_temp_cols = list(AIR_TEMP_COLS.values())
    all_rh_cols = list(RH_COLS.values())
    all_dew_cols = list(DEWPOINT_COLS.values())

    df = to_numeric(df, all_temp_cols + all_rh_cols + all_dew_cols)

    # Temperature plausible range (°C)
    for col in all_temp_cols:
        if col in df.columns:
            df.loc[(df[col] < -50) | (df[col] > 50), col] = np.nan

    # Relative humidity [0, 100] %
    for col in all_rh_cols:
        if col in df.columns:
            df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan

    # Dew point should not exceed temperature at same station/time
    for station, dew_col in DEWPOINT_COLS.items():
        temp_col = AIR_TEMP_COLS.get(station)
        if dew_col in df.columns and temp_col in df.columns:
            mask = df[dew_col] > df[temp_col]
            df.loc[mask, dew_col] = np.nan

    return df


def hourly_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape hourly data into a long station-elevation format."""

    records: List[LascarRecord] = []

    for station_id in STATION_ELEVATION_M:
        temp_col = AIR_TEMP_COLS.get(station_id)
        dew_col = DEWPOINT_COLS.get(station_id)
        rh_col = RH_COLS.get(station_id)

        missing_cols = [
            col for col in (temp_col, dew_col, rh_col) if col not in df.columns
        ]
        if missing_cols:
            continue

        station_df = df[["datetime", temp_col, dew_col, rh_col]].copy()
        station_df = station_df.rename(
            columns={
                temp_col: "temp_c",
                dew_col: "dewpoint_c",
                rh_col: "rh_pct",
            }
        )

        station_df = station_df.dropna(
            subset=["temp_c", "dewpoint_c", "rh_pct"]
        )

        elevation_m = STATION_ELEVATION_M[station_id]

        for row in station_df.itertuples(index=False):
            records.append(
                LascarRecord(
                    station_id=station_id,
                    elevation_m=elevation_m,
                    datetime=row.datetime,
                    temp_c=float(row.temp_c),
                    dewpoint_c=float(row.dewpoint_c),
                    rh_pct=float(row.rh_pct),
                )
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "station_id",
                "elevation_m",
                "datetime",
                "temp_c",
                "dewpoint_c",
                "rh_pct",
            ]
        )

    long_df = pd.DataFrame([r.__dict__ for r in records])
    long_df = long_df.sort_values(["station_id", "datetime"])

    return long_df


def derive_humidity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple derived humidity metrics (saturation and actual vapor pressure)."""

    # Tetens / August-Roche-Magnus approximation
    # es (hPa) = 6.1094 * exp(17.625 * T / (T + 243.04))
    t = df["temp_c"]
    rh = df["rh_pct"]

    es_hpa = 6.1094 * np.exp(17.625 * t / (t + 243.04))
    ea_hpa = es_hpa * (rh / 100.0)

    df = df.copy()
    df["sat_vapor_pressure_hpa"] = es_hpa
    df["vapor_pressure_hpa"] = ea_hpa

    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate long-form hourly records to daily statistics."""

    df = df.set_index("datetime")
    df["date"] = df.index.date

    group_cols = ["station_id", "elevation_m", "date"]

    agg = df.groupby(group_cols).agg(
        temp_c_mean=("temp_c", "mean"),
        temp_c_min=("temp_c", "min"),
        temp_c_max=("temp_c", "max"),
        rh_pct_mean=("rh_pct", "mean"),
        rh_pct_min=("rh_pct", "min"),
        rh_pct_max=("rh_pct", "max"),
        dewpoint_c_mean=("dewpoint_c", "mean"),
        dewpoint_c_min=("dewpoint_c", "min"),
        dewpoint_c_max=("dewpoint_c", "max"),
        n_obs_day=("temp_c", "count"),
    ).reset_index()

    # Filter out days with too few hourly observations
    agg = agg[agg["n_obs_day"] >= 18]

    return agg


def aggregate_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily records to monthly statistics."""

    df = df_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    group_cols = ["station_id", "elevation_m", "year", "month"]

    agg = df.groupby(group_cols).agg(
        temp_c_mean_month=("temp_c_mean", "mean"),
        temp_c_min_month=("temp_c_min", "mean"),
        temp_c_max_month=("temp_c_max", "mean"),
        rh_pct_mean_month=("rh_pct_mean", "mean"),
        dewpoint_c_mean_month=("dewpoint_c_mean", "mean"),
        n_days_month=("date", "nunique"),
    ).reset_index()

    return agg


def save_outputs(
    hourly_long: pd.DataFrame,
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    out_dir: Path = DERIVED_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    hourly_path = out_dir / "hourly_clean.csv"
    daily_path = out_dir / "daily_clean.csv"
    monthly_path = out_dir / "monthly_clean.csv"

    hourly_long.to_csv(hourly_path, index=False)
    daily.to_csv(daily_path, index=False)
    monthly.to_csv(monthly_path, index=False)


def main() -> None:
    if not HOURLY_CSV.exists():
        raise FileNotFoundError(f"Hourly CSV not found at {HOURLY_CSV}")

    hourly_raw = load_hourly_raw(HOURLY_CSV)
    hourly_qc = apply_qc_hourly(hourly_raw)
    hourly_long = hourly_wide_to_long(hourly_qc)

    # Drop any remaining rows missing key variables
    hourly_long = hourly_long.dropna(
        subset=["temp_c", "dewpoint_c", "rh_pct", "elevation_m", "datetime"]
    )

    hourly_long = derive_humidity_metrics(hourly_long)

    daily = aggregate_daily(hourly_long)
    monthly = aggregate_monthly(daily)

    save_outputs(hourly_long, daily, monthly, DERIVED_DIR)


if __name__ == "__main__":
    main()

