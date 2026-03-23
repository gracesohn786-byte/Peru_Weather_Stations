from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DERIVED_DIR = BASE_DIR / "Caucasus_Weather_Stations" / "derived"
OUT_DIR = BASE_DIR / "docs" / "caucasus_edw_summaries"


STATIONS = {
    "khaishi": {
        "daily": DERIVED_DIR / "khaishi_daily_clean.csv",
        "monthly": DERIVED_DIR / "khaishi_monthly_clean.csv",
        "hydro": DERIVED_DIR / "khaishi_hydro_indices.csv",
        "edw_daily": DERIVED_DIR / "edw_daily_khaishi_vs_zugdidi.csv",
        "high_label": "Khaishi (700 m)",
    },
    "zugdidi": {
        "daily": DERIVED_DIR / "zugdidi_daily_clean.csv",
        "monthly": DERIVED_DIR / "zugdidi_monthly_clean.csv",
        "hydro": DERIVED_DIR / "zugdidi_hydro_indices.csv",
        "edw_daily": DERIVED_DIR / "edw_daily_zugdidi_vs_zugdidi.csv",
        "high_label": "Zugdidi (118 m)",
    },
    "lentekhi": {
        "daily": DERIVED_DIR / "lentekhi_daily_clean.csv",
        "monthly": DERIVED_DIR / "lentekhi_monthly_clean.csv",
        "hydro": DERIVED_DIR / "lentekhi_hydro_indices.csv",
        "edw_daily": DERIVED_DIR / "edw_daily_lentekhi_vs_zugdidi.csv",
        "high_label": "Lentekhi (731 m)",
    },
    "mamisoni_pass": {
        "daily": DERIVED_DIR / "mamisoni_pass_daily_clean.csv",
        "monthly": DERIVED_DIR / "mamisoni_pass_monthly_clean.csv",
        "hydro": DERIVED_DIR / "mamisoni_pass_hydro_indices.csv",
        "edw_daily": DERIVED_DIR / "edw_daily_mamisoni_pass_vs_zugdidi.csv",
        "high_label": "Mamisoni Pass (2854 m)",
    },
    "mestia": {
        "daily": DERIVED_DIR / "mestia_daily_clean.csv",
        "monthly": DERIVED_DIR / "mestia_monthly_clean.csv",
        "hydro": DERIVED_DIR / "mestia_hydro_indices.csv",
        "edw_daily": DERIVED_DIR / "edw_daily_mestia_vs_zugdidi.csv",
        "high_label": "Mestia (1441 m)",
    },
    "shovi": {
        "daily": DERIVED_DIR / "shovi_daily_clean.csv",
        "monthly": DERIVED_DIR / "shovi_monthly_clean.csv",
        "hydro": DERIVED_DIR / "shovi_hydro_indices.csv",
        "edw_daily": DERIVED_DIR / "edw_daily_shovi_vs_zugdidi.csv",
        "high_label": "Shovi (1508 m)",
    },
    "tsageri": {
        "daily": DERIVED_DIR / "tsageri_daily_clean.csv",
        "monthly": DERIVED_DIR / "tsageri_monthly_clean.csv",
        "hydro": DERIVED_DIR / "tsageri_hydro_indices.csv",
        "edw_daily": DERIVED_DIR / "edw_daily_tsageri_vs_zugdidi.csv",
        "high_label": "Tsageri (500 m)",
    },
    "lebarde": {
        "daily": DERIVED_DIR / "lebarde_daily_clean.csv",
        "monthly": DERIVED_DIR / "lebarde_monthly_clean.csv",
        "hydro": DERIVED_DIR / "lebarde_hydro_indices.csv",
        "edw_daily": DERIVED_DIR / "edw_daily_lebarde_vs_zugdidi.csv",
        "high_label": "Lebarde (1491 m)",
    },
}


def _parse_dates_maybe(series: pd.Series) -> pd.Series:
    """Parse date strings robustly across M/D/YYYY and ISO formats."""
    dt = pd.to_datetime(series, errors="coerce")
    # If many fail (likely day/month swapped), try day-first once.
    if dt.notna().mean() < 0.9:
        dt2 = pd.to_datetime(series, errors="coerce", dayfirst=True)
        if dt2.notna().mean() > dt.notna().mean():
            dt = dt2
    return dt


def fit_linear_trend_monthly(df: pd.DataFrame, value_col: str) -> float:
    series = df.dropna(subset=[value_col]).copy()
    if series.empty:
        return np.nan

    # Fractional year as in the notebooks: year + (month - 0.5)/12.
    years = series["year"].to_numpy() + (series["month"].to_numpy() - 0.5) / 12.0
    y = series[value_col].to_numpy()
    slope_year, _ = np.polyfit(years, y, 1)
    return float(slope_year * 10.0)  # per decade


def monthly_edw_summary(edw_daily: pd.DataFrame) -> Tuple[float, float]:
    df = edw_daily.copy()
    df["date"] = _parse_dates_maybe(df["date"])
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    monthly = df.resample("ME").mean()
    mean_delta_t = float(monthly["delta_t_c"].mean())
    mean_lapse_rate = float(monthly["lapse_rate_c_per_100m"].mean())
    return mean_delta_t, mean_lapse_rate


def freezing_days_last5(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["date"] = _parse_dates_maybe(df["date"])
    df = df.dropna(subset=["date"])

    freezing = (df["tmean_c"] <= 0.0)
    per_year = freezing.groupby(df["date"].dt.year).sum()
    per_year.index.name = "Year"
    per_year.name = "Freezing days"
    per_year = per_year.sort_index().tail(5)
    return per_year.reset_index()


def heatwave_days_last5(daily: pd.DataFrame, q: float = 0.9) -> pd.DataFrame:
    df = daily.copy()
    df["date"] = _parse_dates_maybe(df["date"])
    df = df.dropna(subset=["date"])

    threshold = float(df["tmean_c"].quantile(q))
    hot = df["tmean_c"] > threshold
    per_year = hot.groupby(df["date"].dt.year).sum()
    per_year.index.name = "Year"
    per_year.name = "Heatwave days"
    per_year = per_year.sort_index().tail(5)
    out = per_year.reset_index()
    out["threshold_tmean_c(90th)"] = threshold
    return out


def write_markdown(station_key: str, out_path: Path) -> None:
    cfg = STATIONS[station_key]

    daily = pd.read_csv(cfg["daily"])
    monthly = pd.read_csv(cfg["monthly"])
    hydro = pd.read_csv(cfg["hydro"])
    edw_daily = pd.read_csv(cfg["edw_daily"])

    station_name = str(daily["station_name"].dropna().iloc[0]) if "station_name" in daily.columns else cfg["high_label"]

    warming_rate = fit_linear_trend_monthly(monthly, "tmean_c_month")
    mean_delta_t, mean_lapse_rate = monthly_edw_summary(edw_daily)

    freezing_last5 = freezing_days_last5(daily)
    heatwave_last5 = heatwave_days_last5(daily, q=0.9)

    hydro_sorted = hydro.sort_values("hydro_year")
    hydro_tail = hydro_sorted.tail(5).copy()

    # Build markdown sections (mirrors docs/edw_console_summary.md style).
    lines = []
    lines.append(f"### Monthly warming rates (from `{cfg['high_label']}` monthly mean series)")
    lines.append("")
    lines.append("From the monthly trend analysis:")
    lines.append("")
    if np.isnan(warming_rate):
        lines.append(f"- **{station_name}**: N/A")
    else:
        lines.append(f"- **{station_name}**: {warming_rate:.3f} °C/decade")
    lines.append("")
    lines.append("---")

    lines.append(f"### EDW summary ({cfg['high_label']} − Zugdidi 118 m) — monthly means")
    lines.append("")
    lines.append("Using the EDW daily comparison dataset resampled to monthly means:")
    lines.append("")
    lines.append(f"- **Mean ΔT (high − low)**: {mean_delta_t:.3f} °C")
    lines.append(f"- **Mean lapse rate**: {mean_lapse_rate:.3f} °C per 100 m")
    lines.append("")
    lines.append("Positive ΔT and lapse rate indicate the higher station is warmer than expected relative to the lower station over the analyzed period.")
    lines.append("---")

    lines.append("### Freezing days per year (daily data)")
    lines.append("")
    lines.append(f"**{station_name} — freezing days per year (last 5 years in output):**")
    lines.append("")
    lines.append("| Year | Freezing days |")
    lines.append("|------|---------------|")
    for _, row in freezing_last5.iterrows():
        year = int(row["Year"])
        val = int(row["Freezing days"])
        lines.append(f"| {year} | {val} |")
    lines.append("---")

    lines.append("### Heatwave days per year (Tmean > 90th percentile)")
    lines.append("")
    lines.append("Threshold defined as the 90th percentile of daily mean temperature for this station over the full record.")
    lines.append("")
    thresh = float(daily["tmean_c"].quantile(0.9))
    lines.append(f"**{station_name} — heatwave days per year (last 5 years in output):** (threshold = {thresh:.2f} °C)")
    lines.append("")
    lines.append("| Year | Heatwave days |")
    lines.append("|------|---------------|")
    for _, row in heatwave_last5.iterrows():
        year = int(row["Year"])
        val = int(row["Heatwave days"])
        lines.append(f"| {year} | {val} |")
    lines.append("---")

    lines.append(f"### Hydrological indices summary — {station_name}")
    lines.append("")
    lines.append(f"From `{Path(cfg['hydro']).name}` (last 5 hydrological years):")
    lines.append("")
    lines.append("| Hydro year | PDD sum (`pdd_degC_sum`) | Freezing days (`n_freezing_days`) | Heavy-precip days (`n_days_p>=20.0mm`) |")
    lines.append("|-----------:|--------------------------|------------------------------------|----------------------------------------|")
    for _, row in hydro_tail.iterrows():
        hy = int(row["hydro_year"])
        pdd = row["pdd_degC_sum"]
        frz = int(row["n_freezing_days"])
        hvy = int(row["n_days_p>=20.0mm"])
        pdd_str = f"{float(pdd):.1f}" if pd.notna(pdd) else "N/A"
        lines.append(f"| {hy} | {pdd_str} | {frz} | {hvy} |")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for station_key in STATIONS:
        out_path = OUT_DIR / f"caucasus_{station_key}_edw_summary.md"
        write_markdown(station_key, out_path)
        print("wrote:", out_path)


if __name__ == "__main__":
    main()

