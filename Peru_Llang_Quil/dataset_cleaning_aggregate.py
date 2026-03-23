from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class StationConfig:
    name: str
    path: Path
    elevation_m: float
    time_col: str
    column_map: Dict[str, str]


def load_station(config: StationConfig) -> pd.DataFrame:
    """Load a station CSV and standardize the main column names.

    Expected standardized columns (if available in the raw file):
      - time: pandas datetime
      - tair_c: air temperature [°C]
      - rh_pct: relative humidity [%]
      - p_mm: precipitation depth per interval [mm]
      - wind_speed_ms: mean wind speed [m/s]
      - wind_gust_ms: gust wind speed [m/s]
      - wind_dir_deg: wind direction [°]
      - sw_in_wm2: incoming shortwave radiation [W/m²]
    """
    # Use latin1 to safely read non-UTF-8 CSV exports.
    df = pd.read_csv(config.path, encoding="latin1")

    # Normalize column names to avoid issues with BOMs or stray spaces.
    # Handle both true BOM (\ufeff) and the 'ï»¿' sequence that appears
    # when a UTF-8 BOM is decoded as latin1.
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .str.strip()
    )

    if config.time_col not in df.columns:
        raise ValueError(
            f"Time column '{config.time_col}' not found in {config.path}; "
            f"available columns: {list(df.columns)}"
        )

    df[config.time_col] = pd.to_datetime(df[config.time_col])
    df = df.set_index(config.time_col).sort_index()

    # Rename selected physical variables to a common schema.
    rename_map = {raw: std for raw, std in config.column_map.items() if raw in df.columns}
    df = df.rename(columns=rename_map)

    # Coerce expected numeric variables to numeric, turning non-numeric
    # entries into NaN to avoid type errors during aggregation.
    numeric_cols = [
        "tair_c",
        "rh_pct",
        "p_mm",
        "wind_speed_ms",
        "wind_gust_ms",
        "wind_dir_deg",
        "sw_in_wm2",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep time as a column as well as index for convenience.
    df = df.reset_index().rename(columns={config.time_col: "time"})

    return df


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add useful time-based helper columns."""
    out = df.copy()
    out["year"] = out["time"].dt.year
    out["month"] = out["time"].dt.month
    out["day"] = out["time"].dt.day
    out["doy"] = out["time"].dt.dayofyear
    out["hour"] = out["time"].dt.hour
    out["date"] = out["time"].dt.date
    return out


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly data to daily statistics."""
    if "date" not in df.columns:
        df = add_time_columns(df)

    group = df.groupby("date")

    daily = pd.DataFrame(index=group.size().index)
    daily.index.name = "date"

    if "tair_c" in df.columns:
        daily["tmean_c"] = group["tair_c"].mean()
        daily["tmin_c"] = group["tair_c"].min()
        daily["tmax_c"] = group["tair_c"].max()

    if "p_mm" in df.columns:
        daily["p_mm"] = group["p_mm"].sum()

    if "rh_pct" in df.columns:
        daily["rh_mean_pct"] = group["rh_pct"].mean()

    if "wind_speed_ms" in df.columns:
        daily["wind_speed_mean_ms"] = group["wind_speed_ms"].mean()

    if "sw_in_wm2" in df.columns:
        daily["sw_in_mean_wm2"] = group["sw_in_wm2"].mean()

    return daily.reset_index()


def aggregate_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily statistics to monthly."""
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    group = df.groupby(["year", "month"])
    monthly = pd.DataFrame(index=group.size().index)

    if "tmean_c" in df.columns:
        monthly["tmean_c"] = group["tmean_c"].mean()
        monthly["tmin_c"] = group["tmin_c"].mean()
        monthly["tmax_c"] = group["tmax_c"].mean()

    if "p_mm" in df.columns:
        monthly["p_mm"] = group["p_mm"].sum()

    if "rh_mean_pct" in df.columns:
        monthly["rh_mean_pct"] = group["rh_mean_pct"].mean()

    return monthly.reset_index()


def compute_positive_degree_days(
    daily: pd.DataFrame,
    temp_col: str = "tmean_c",
    base_temp_c: float = 0.0,
) -> pd.DataFrame:
    """Compute positive degree-day (PDD) sums by hydrological year."""
    df = daily.copy()
    if temp_col not in df.columns:
        raise ValueError(f"Temperature column '{temp_col}' not found in daily data.")

    df["date"] = pd.to_datetime(df["date"])
    df["hydro_year"] = np.where(df["date"].dt.month >= 10, df["date"].dt.year + 1, df["date"].dt.year)
    df["pdd_degC"] = (df[temp_col] - base_temp_c).clip(lower=0.0)

    pdd = (
        df.groupby("hydro_year")["pdd_degC"]
        .sum()
        .rename("pdd_degC_sum")
        .reset_index()
    )
    return pdd


def compute_freezing_days(daily: pd.DataFrame, temp_col: str = "tmean_c") -> pd.DataFrame:
    """Count days at or below freezing by hydrological year."""
    df = daily.copy()
    if temp_col not in df.columns:
        raise ValueError(f"Temperature column '{temp_col}' not found in daily data.")

    df["date"] = pd.to_datetime(df["date"])
    df["hydro_year"] = np.where(df["date"].dt.month >= 10, df["date"].dt.year + 1, df["date"].dt.year)
    df["is_freezing_day"] = df[temp_col] <= 0.0

    out = (
        df.groupby("hydro_year")["is_freezing_day"]
        .sum()
        .rename("n_freezing_days")
        .reset_index()
    )
    return out


def compute_heavy_precip_days(
    daily: pd.DataFrame,
    threshold_mm: float = 20.0,
) -> pd.DataFrame:
    """Count days with heavy precipitation per hydrological year."""
    if "p_mm" not in daily.columns:
        raise ValueError("Column 'p_mm' not found in daily data.")

    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["hydro_year"] = np.where(df["date"].dt.month >= 10, df["date"].dt.year + 1, df["date"].dt.year)
    df["heavy_rain_day"] = df["p_mm"] >= threshold_mm

    out = (
        df.groupby("hydro_year")["heavy_rain_day"]
        .sum()
        .rename(f"n_days_p>={threshold_mm}mm")
        .reset_index()
    )
    return out


def merge_stations_for_edw(
    daily_a: pd.DataFrame,
    daily_b: pd.DataFrame,
    name_a: str,
    name_b: str,
    elevation_a_m: float,
    elevation_b_m: float,
) -> pd.DataFrame:
    """Merge daily data from two stations and compute elevation-dependent metrics."""
    df_a = daily_a.copy()
    df_b = daily_b.copy()

    df_a["date"] = pd.to_datetime(df_a["date"])
    df_b["date"] = pd.to_datetime(df_b["date"])

    merged = pd.merge(df_a, df_b, on="date", suffixes=(f"_{name_a}", f"_{name_b}"))
    merged["elev_a_m"] = elevation_a_m
    merged["elev_b_m"] = elevation_b_m

    if f"tmean_c_{name_a}" in merged.columns and f"tmean_c_{name_b}" in merged.columns:
        merged["delta_t_c"] = merged[f"tmean_c_{name_a}"] - merged[f"tmean_c_{name_b}"]
        dz = elevation_a_m - elevation_b_m
        if dz != 0:
            merged["lapse_rate_c_per_100m"] = merged["delta_t_c"] / dz * 100.0

    return merged


def describe_station_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact table of variable name, non-missing count, and basic stats."""
    summary_rows = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            summary_rows.append(
                {
                    "variable": col,
                    "non_missing": series.notna().sum(),
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "p25": series.quantile(0.25),
                    "median": series.median(),
                    "p75": series.quantile(0.75),
                    "max": series.max(),
                }
            )
        else:
            summary_rows.append(
                {
                    "variable": col,
                    "non_missing": series.notna().sum(),
                    "mean": np.nan,
                    "std": np.nan,
                    "min": None,
                    "p25": None,
                    "median": None,
                    "p75": None,
                    "max": None,
                }
            )
    return pd.DataFrame(summary_rows)


def example_configs(base_dir: Path) -> Tuple[StationConfig, StationConfig]:
    """Return example StationConfig objects for Llanganuco and Quilcayhuanca.

    Adjust the paths, elevations, time column, and column_map keys to match
    your actual CSV header names before running this script.
    """
    llang = StationConfig(
        name="Llanganuco",
        path=base_dir
        / "Llanganuco-20260220T200537Z-1-001"
        / "Llanganuco"
        / "Llanganuco_WX_hourly_12Jul2004_7Dec2025.csv",
        elevation_m=3850.0,
        time_col="Datetime",
        column_map={
            "AirTemperature (Â°C)": "tair_c",
            "RH(corrected)": "rh_pct",
            "Precip. mm": "p_mm",
            "Wind Speed (m/s)": "wind_speed_ms",
            "Gust Speed (m/s)": "wind_gust_ms",
            "Wind Direction (Â°)": "wind_dir_deg",
            "Solar Radiation (W/m^2)": "sw_in_wm2",
        },
    )

    quil = StationConfig(
        name="Quilcayhuanca",
        path=base_dir
        / "Quilcayhuanca-20260220T200528Z-1-001"
        / "Quilcayhuanca"
        / "Quilcay_CasaDeAgua_4Jul2013_7Dec2025.csv",
        elevation_m=3600.0,
        time_col="Datetime",
        column_map={
            "AirTemperature (Â°C)": "tair_c",
            "RH(corrected)": "rh_pct",
            "Precip. mm": "p_mm",
            "Wind Speed (m/s)": "wind_speed_ms",
            "Gust Speed (m/s)": "wind_gust_ms",
            "Wind Direction (Â°)": "wind_dir_deg",
            "Solar Radiation (W/m^2)": "sw_in_wm2",
        },
    )

    return llang, quil


def main_inspect_and_aggregate(base_dir: Optional[Path] = None) -> None:
    """Inspect variables and build daily/monthly summary tables for both stations."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    llang_cfg, quil_cfg = example_configs(base_dir)

    llang = add_time_columns(load_station(llang_cfg))
    quil = add_time_columns(load_station(quil_cfg))

    print("Llanganuco variable summary:")
    print(describe_station_variables(llang).to_string(index=False))
    print("\nQuilcayhuanca variable summary:")
    print(describe_station_variables(quil).to_string(index=False))

    llang_daily = aggregate_daily(llang)
    quil_daily = aggregate_daily(quil)

    llang_monthly = aggregate_monthly(llang_daily)
    quil_monthly = aggregate_monthly(quil_daily)

    out_dir = base_dir / "derived"
    out_dir.mkdir(exist_ok=True)

    llang_daily.to_csv(out_dir / "llanganuco_daily.csv", index=False)
    quil_daily.to_csv(out_dir / "quilcayhuanca_daily.csv", index=False)
    llang_monthly.to_csv(out_dir / "llanganuco_monthly.csv", index=False)
    quil_monthly.to_csv(out_dir / "quilcayhuanca_monthly.csv", index=False)

    print(f"\nWrote daily and monthly summary CSVs to {out_dir}")


def main_degree_days_and_extremes(base_dir: Optional[Path] = None) -> None:
    """Compute degree-day and heavy-precipitation indices and save as CSV."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    derived_dir = base_dir / "derived"
    llang_daily = pd.read_csv(derived_dir / "llanganuco_daily.csv", parse_dates=["date"])
    quil_daily = pd.read_csv(derived_dir / "quilcayhuanca_daily.csv", parse_dates=["date"])

    llang_pdd = compute_positive_degree_days(llang_daily)
    quil_pdd = compute_positive_degree_days(quil_daily)

    llang_freeze = compute_freezing_days(llang_daily)
    quil_freeze = compute_freezing_days(quil_daily)

    llang_heavy = compute_heavy_precip_days(llang_daily, threshold_mm=20.0)
    quil_heavy = compute_heavy_precip_days(quil_daily, threshold_mm=20.0)

    llang_indices = (
        llang_pdd.merge(llang_freeze, on="hydro_year")
        .merge(llang_heavy, on="hydro_year")
    )
    quil_indices = (
        quil_pdd.merge(quil_freeze, on="hydro_year")
        .merge(quil_heavy, on="hydro_year")
    )

    llang_indices.to_csv(derived_dir / "llanganuco_hydro_indices.csv", index=False)
    quil_indices.to_csv(derived_dir / "quilcayhuanca_hydro_indices.csv", index=False)

    print(f"Wrote hydro-climatic indices to {derived_dir}")


def main_edw_daily(base_dir: Optional[Path] = None) -> None:
    """Build a merged daily dataset with elevation-dependent warming metrics."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    derived_dir = base_dir / "derived"
    llang_daily = pd.read_csv(derived_dir / "llanganuco_daily.csv")
    quil_daily = pd.read_csv(derived_dir / "quilcayhuanca_daily.csv")

    llang_cfg, quil_cfg = example_configs(base_dir)
    edw_daily = merge_stations_for_edw(
        llang_daily,
        quil_daily,
        name_a="llang",
        name_b="quil",
        elevation_a_m=llang_cfg.elevation_m,
        elevation_b_m=quil_cfg.elevation_m,
    )

    edw_daily.to_csv(derived_dir / "edw_daily_llang_vs_quil.csv", index=False)
    print(f"Wrote EDW comparison table to {derived_dir}")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    main_inspect_and_aggregate(base)
    main_degree_days_and_extremes(base)
    main_edw_daily(base)

