from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DERIVED_DIR = BASE_DIR / "derived"


@dataclass
class StationMetadata:
    station_id: str
    station_name: str
    elevation_m: Optional[float]
    latitude: Optional[float]
    longitude: Optional[float]
    source_file: str


def slugify(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def parse_station_metadata(path: Path) -> StationMetadata:
    stem = path.stem  # e.g. "Mestia 1441m Lat43d03_ Long42d45_"
    parts = stem.split()

    station_name_parts: List[str] = []
    elevation_m: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    for i, token in enumerate(parts):
        if token.endswith("m") and token[:-1].replace(".", "", 1).isdigit():
            try:
                elevation_m = float(token[:-1])
            except ValueError:
                elevation_m = None
            station_name_parts = parts[:i]
            continue

    if not station_name_parts:
        station_name = stem
    else:
        station_name = " ".join(station_name_parts)

    for token in parts:
        if token.lower().startswith("lat"):
            numeric = token[3:].replace("_", "")
            try:
                latitude = float(numeric.replace("d", "."))
            except ValueError:
                latitude = None
        if token.lower().startswith("long"):
            numeric = token[4:].replace("_", "")
            try:
                longitude = float(numeric.replace("d", "."))
            except ValueError:
                longitude = None

    station_id = slugify(station_name)

    return StationMetadata(
        station_id=station_id,
        station_name=station_name,
        elevation_m=elevation_m,
        latitude=latitude,
        longitude=longitude,
        source_file=path.name,
    )


def find_station_files(base_dir: Path) -> List[Path]:
    return sorted(base_dir.glob("*.xlsx"))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = {}
    for col in df.columns:
        col_str = str(col).strip().lower()
        col_str = col_str.replace(" ", "_")
        col_str = col_str.replace("-", "_")
        new_cols[col] = col_str
    df = df.rename(columns=new_cols)
    return df


def _pick_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    for cand in candidates:
        cand_l = cand.lower()
        for col in cols:
            if cand_l in col.lower():
                return col
    return None


def load_station_excel(path: Path, meta: StationMetadata) -> pd.DataFrame:
    df = pd.read_excel(
        path,
        sheet_name=0,
        na_values=[-99.0, -99, "-99.0", "-99"],
    )

    df = _normalize_columns(df)

    date_col = _pick_column(df.columns, ["date", "day", "datum"])
    datetime_col = _pick_column(df.columns, ["datetime", "time"])

    if date_col is not None and datetime_col is None:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    elif datetime_col is not None:
        dt = pd.to_datetime(df[datetime_col], errors="coerce")
        df["date"] = dt.dt.date
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError(f"Could not find a date/datetime column in {path.name}")

    df = df.dropna(subset=["date"])

    tmean_col = _pick_column(df.columns, ["tmean", "t_mean", "tavg", "t_avg", "t_average"])
    tmax_col = _pick_column(df.columns, ["tmax", "t_max"])
    tmin_col = _pick_column(df.columns, ["tmin", "t_min"])

    for col in [tmean_col, tmax_col, tmin_col]:
        if col is not None:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col].isin([-99, -99.0]), col] = np.nan

    for col in [tmean_col, tmax_col, tmin_col]:
        if col is not None:
            df.loc[(df[col] < -60) | (df[col] > 50), col] = np.nan

    out = pd.DataFrame({"date": df["date"]})

    if tmean_col is not None:
        out["tmean_c"] = df[tmean_col]
    else:
        out["tmean_c"] = np.nan

    if tmax_col is not None:
        out["tmax_c"] = df[tmax_col]
    else:
        out["tmax_c"] = np.nan

    if tmin_col is not None:
        out["tmin_c"] = df[tmin_col]
    else:
        out["tmin_c"] = np.nan

    out["station_id"] = meta.station_id
    out["station_name"] = meta.station_name
    out["elevation_m"] = meta.elevation_m
    out["latitude"] = meta.latitude
    out["longitude"] = meta.longitude

    return out


def build_daily_from_station(df_station: pd.DataFrame) -> pd.DataFrame:
    df = df_station.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    key_cols = ["tmean_c", "tmax_c", "tmin_c"]
    df = df[~df[key_cols].isna().all(axis=1)]

    df["dtr_c"] = df["tmax_c"] - df["tmin_c"]
    df["n_obs_day"] = 1

    grouped = (
        df.groupby(
            ["station_id", "station_name", "elevation_m", "latitude", "longitude", "date"],
            as_index=False,
        )
        .agg(
            tmean_c=("tmean_c", "mean"),
            tmax_c=("tmax_c", "max"),
            tmin_c=("tmin_c", "min"),
            dtr_c=("dtr_c", "mean"),
            n_obs_day=("n_obs_day", "sum"),
        )
    )

    return grouped


def combine_daily(station_dailies: List[pd.DataFrame]) -> pd.DataFrame:
    if not station_dailies:
        return pd.DataFrame(
            columns=[
                "station_id",
                "station_name",
                "elevation_m",
                "latitude",
                "longitude",
                "date",
                "tmean_c",
                "tmax_c",
                "tmin_c",
                "dtr_c",
                "n_obs_day",
            ]
        )
    combined = pd.concat(station_dailies, ignore_index=True)
    combined = combined.sort_values(["station_id", "date"])
    return combined


def aggregate_to_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
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
            n_days_month=("date", "nunique"),
        )
    )

    grouped = grouped[grouped["n_days_month"] >= 10]

    return grouped


def write_outputs(
    df_daily: pd.DataFrame,
    df_monthly: pd.DataFrame,
    metadata: List[StationMetadata],
    out_dir: Path = DERIVED_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_path = out_dir / "caucasus_daily_clean.csv"
    monthly_path = out_dir / "caucasus_monthly_clean.csv"
    meta_path = out_dir / "station_metadata.csv"

    df_daily.to_csv(daily_path, index=False)
    df_monthly.to_csv(monthly_path, index=False)

    meta_df = pd.DataFrame([m.__dict__ for m in metadata])
    meta_df.to_csv(meta_path, index=False)


def main() -> None:
    files = find_station_files(BASE_DIR)
    if not files:
        raise FileNotFoundError(f"No .xlsx station files found in {BASE_DIR}")

    all_metadata: List[StationMetadata] = []
    station_dailies: List[pd.DataFrame] = []

    for path in files:
        meta = parse_station_metadata(path)
        all_metadata.append(meta)
        try:
            raw_station = load_station_excel(path, meta)
            daily_station = build_daily_from_station(raw_station)
            station_dailies.append(daily_station)
        except Exception as exc:
            print(f"Warning: skipping {path.name} due to error: {exc}")

    daily = combine_daily(station_dailies)
    monthly = aggregate_to_monthly(daily)

    write_outputs(daily, monthly, all_metadata, DERIVED_DIR)


if __name__ == "__main__":
    main()

