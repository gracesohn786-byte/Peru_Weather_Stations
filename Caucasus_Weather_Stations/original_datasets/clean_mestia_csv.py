from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from clean_caucasus_weather import DERIVED_DIR, build_daily_from_station, parse_station_metadata


BASE_DIR = Path(__file__).resolve().parent
MESTIA_CSV = BASE_DIR / "Mestia 1441m Lat43d03_ Long42d45_.csv"


def load_mestia_csv(path: Path) -> pd.DataFrame:
    meta = parse_station_metadata(path)

    df = pd.read_csv(
        path,
        na_values=[-99.99, -99.9, -99.0, -99, "-99.99", "-99.0", "-99"],
    )

    df["date"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"]),
        errors="coerce",
    )

    for col in ["T_mean", "T_max", "T_min", "Prec(mm)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col].isin([-99, -99.0, -99.9, -99.99]), col] = np.nan

    for col in ["T_mean", "T_max", "T_min"]:
        if col in df.columns:
            df.loc[(df[col] < -60) | (df[col] > 50), col] = np.nan

    if "Prec(mm)" in df.columns:
        df.loc[df["Prec(mm)"] < 0, "Prec(mm)"] = np.nan

    out = pd.DataFrame(
        {
            "date": df["date"],
            "tmean_c": df.get("T_mean"),
            "tmax_c": df.get("T_max"),
            "tmin_c": df.get("T_min"),
            "precip_mm": df.get("Prec(mm)"),
            "station_id": meta.station_id,
            "station_name": meta.station_name,
            "elevation_m": meta.elevation_m,
            "latitude": meta.latitude,
            "longitude": meta.longitude,
        }
    )

    out = out.dropna(subset=["date"])
    out = out[~out[["tmean_c", "tmax_c", "tmin_c"]].isna().all(axis=1)]
    return out


def aggregate_mestia_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    df = df_daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    grouped = (
        df.groupby(
            [
                "station_id",
                "station_name",
                "elevation_m",
                "latitude",
                "longitude",
                "year",
                "month",
            ],
            as_index=False,
        )
        .agg(
            tmean_c_month=("tmean_c", "mean"),
            tmax_c_month=("tmax_c", "mean"),
            tmin_c_month=("tmin_c", "mean"),
            dtr_c_month=("dtr_c", "mean"),
            precip_mm_month=("precip_mm", "sum"),
            n_days_month=("date", "nunique"),
        )
    )

    return grouped[grouped["n_days_month"] >= 10]


def main() -> None:
    if not MESTIA_CSV.exists():
        raise FileNotFoundError(f"Mestia CSV not found at {MESTIA_CSV}")

    raw = load_mestia_csv(MESTIA_CSV)
    daily = build_daily_from_station(raw)

    daily_precip = (
        raw.groupby(
            ["station_id", "station_name", "elevation_m", "latitude", "longitude", "date"],
            as_index=False,
        )
        .agg(precip_mm=("precip_mm", "sum"))
    )
    daily = daily.merge(
        daily_precip,
        on=["station_id", "station_name", "elevation_m", "latitude", "longitude", "date"],
        how="left",
    )

    monthly = aggregate_mestia_monthly(daily)

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DERIVED_DIR / "mestia_daily_clean.csv", index=False)
    monthly.to_csv(DERIVED_DIR / "mestia_monthly_clean.csv", index=False)


if __name__ == "__main__":
    main()

