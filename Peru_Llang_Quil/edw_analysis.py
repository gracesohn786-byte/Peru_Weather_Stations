from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DERIVED_DIR = BASE_DIR / "derived"


@dataclass
class DerivedPaths:
    llang_daily: Path = DERIVED_DIR / "llanganuco_daily.csv"
    quil_daily: Path = DERIVED_DIR / "quilcayhuanca_daily.csv"
    llang_monthly: Path = DERIVED_DIR / "llanganuco_monthly.csv"
    quil_monthly: Path = DERIVED_DIR / "quilcayhuanca_monthly.csv"
    llang_hydro: Path = DERIVED_DIR / "llanganuco_hydro_indices.csv"
    quil_hydro: Path = DERIVED_DIR / "quilcayhuanca_hydro_indices.csv"
    edw_daily: Path = DERIVED_DIR / "edw_daily_llang_vs_quil.csv"


PATHS = DerivedPaths()


# ---------------------------------------------------------------------
# Loading helpers (Todo: load-derived-data)
# ---------------------------------------------------------------------


def load_daily() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load daily climate datasets for both stations."""
    llang = pd.read_csv(PATHS.llang_daily, parse_dates=["date"])
    quil = pd.read_csv(PATHS.quil_daily, parse_dates=["date"])
    return llang, quil


def load_monthly() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load monthly climate datasets for both stations with datetime."""
    llang = pd.read_csv(PATHS.llang_monthly)
    quil = pd.read_csv(PATHS.quil_monthly)

    for df in (llang, quil):
        df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))

    return llang, quil


def load_hydro_indices() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load hydro-climate index datasets for both stations."""
    llang = pd.read_csv(PATHS.llang_hydro)
    quil = pd.read_csv(PATHS.quil_hydro)
    return llang, quil


def load_edw_daily() -> pd.DataFrame:
    """Load EDW comparison dataset with parsed dates."""
    edw = pd.read_csv(PATHS.edw_daily, parse_dates=["date"])
    return edw


# ---------------------------------------------------------------------
# 3.1 Monthly trends and seasonal structure (Todo: monthly-trends-seasonality)
# ---------------------------------------------------------------------


def fit_linear_trend(df: pd.DataFrame, value_col: str) -> Tuple[float, float]:
    """Fit a simple linear trend to a monthly series; return slope (°C/decade), intercept."""
    series = df.dropna(subset=[value_col]).copy()
    if series.empty:
        return np.nan, np.nan

    # Use fractional year as time variable.
    years = series["year"].values + (series["month"].values - 0.5) / 12.0
    t = years
    y = series[value_col].values
    slope_year, intercept = np.polyfit(t, y, 1)
    slope_decade = slope_year * 10.0
    return slope_decade, intercept


def analyze_monthly_trends_and_seasonality(show_plots: bool = True) -> None:
    """Compute and optionally plot warming trends and seasonal cycles for both stations."""
    llang, quil = load_monthly()

    # Trend analysis
    slope_ll, _ = fit_linear_trend(llang, "tmean_c")
    slope_qu, _ = fit_linear_trend(quil, "tmean_c")

    print("Monthly warming rate (°C/decade):")
    print(f"  Llanganuco:    {slope_ll:.3f} °C/decade" if not np.isnan(slope_ll) else "  Llanganuco:    N/A")
    print(f"  Quilcayhuanca: {slope_qu:.3f} °C/decade" if not np.isnan(slope_qu) else "  Quilcayhuanca: N/A")

    if not show_plots:
        return

    # Time series with rolling means
    plt.figure(figsize=(12, 6))
    plt.plot(llang["date"], llang["tmean_c"], label="Llanganuco (monthly)", alpha=0.4)
    plt.plot(quil["date"], quil["tmean_c"], label="Quilcayhuanca (monthly)", alpha=0.4)
    plt.plot(llang["date"], llang["tmean_c"].rolling(12, min_periods=6).mean(), label="Llanganuco (12-mo mean)", linewidth=2)
    plt.plot(quil["date"], quil["tmean_c"].rolling(12, min_periods=6).mean(), label="Quilcayhuanca (12-mo mean)", linewidth=2)
    plt.title("Long-Term Temperature Trends by Elevation")
    plt.xlabel("Year")
    plt.ylabel("Mean monthly temperature (°C)")
    plt.legend()
    plt.tight_layout()

    # Seasonal cycle (climatology) – Llanganuco
    plt.figure(figsize=(8, 4))
    seasonal_ll = llang.groupby("month")["tmean_c"].mean()
    seasonal_qu = quil.groupby("month")["tmean_c"].mean()
    plt.plot(seasonal_ll.index, seasonal_ll.values, marker="o", label="Llanganuco")
    plt.plot(seasonal_qu.index, seasonal_qu.values, marker="o", label="Quilcayhuanca")
    plt.title("Seasonal Cycle of Temperature")
    plt.xlabel("Month")
    plt.ylabel("Climatological mean T (°C)")
    plt.legend()
    plt.tight_layout()

    # Year–month heatmap (Llanganuco)
    pivot_ll = llang.pivot(index="year", columns="month", values="tmean_c")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_ll, cmap="coolwarm", cbar_kws={"label": "Tmean (°C)"})
    plt.title("Monthly Temperature Heatmap — Llanganuco")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.tight_layout()


# ---------------------------------------------------------------------
# 3.3 EDW signal: ΔT and lapse rate
# ---------------------------------------------------------------------


def analyze_edw(show_plots: bool = True) -> None:
    """Analyze elevation-dependent warming using daily EDW dataset."""
    edw = load_edw_daily()
    edw = edw.set_index("date").sort_index()

    monthly_delta = edw["delta_t_c"].resample("ME").mean()
    monthly_lapse = edw["lapse_rate_c_per_100m"].resample("ME").mean()

    print("EDW summary (ΔT and lapse rate) — monthly means:")
    print(f"  ΔT (high − low) mean: {monthly_delta.mean():.3f} °C")
    print(f"  Lapse rate mean:      {monthly_lapse.mean():.3f} °C/100m")

    if not show_plots:
        return

    # ΔT time series
    plt.figure(figsize=(12, 5))
    monthly_delta.plot(alpha=0.5, label="Monthly ΔT")
    monthly_delta.rolling(12, min_periods=6).mean().plot(linewidth=2, label="12-mo mean ΔT")
    plt.title("Elevation Temperature Difference (ΔT = Thigh − Tlow)")
    plt.ylabel("ΔT (°C)")
    plt.xlabel("Year")
    plt.legend()
    plt.tight_layout()

    # Lapse rate time series
    plt.figure(figsize=(12, 5))
    monthly_lapse.plot(alpha=0.5, label="Monthly lapse rate")
    monthly_lapse.rolling(12, min_periods=6).mean().plot(linewidth=2, label="12-mo mean lapse rate")
    plt.title("Daily-Derived Lapse Rate Over Time")
    plt.ylabel("Lapse rate (°C / 100 m)")
    plt.xlabel("Year")
    plt.legend()
    plt.tight_layout()


# ---------------------------------------------------------------------
# 3.4 Daily variability and extremes
# ---------------------------------------------------------------------


def analyze_daily_extremes(show_plots: bool = True) -> None:
    """Analyze daily variability: freezing days, heatwaves, and simple extremes."""
    llang, quil = load_daily()

    # Freezing days per year
    for name, df in (("Llanganuco", llang), ("Quilcayhuanca", quil)):
        df = df.set_index("date")
        freezing = (df["tmean_c"] <= 0).resample("YE").sum()
        print(f"\nFreezing days per year — {name}:")
        print(freezing.tail())

        if show_plots:
            plt.figure(figsize=(8, 4))
            freezing.index = freezing.index.year
            plt.plot(freezing.index, freezing.values, marker="o")
            plt.title(f"Freezing Days per Year — {name}")
            plt.xlabel("Year")
            plt.ylabel("Number of freezing days")
            plt.tight_layout()

    # Heatwaves: use station-specific 90th percentile of daily mean T
    for name, df in (("Llanganuco", llang), ("Quilcayhuanca", quil)):
        df = df.set_index("date")
        thresh = df["tmean_c"].quantile(0.9)
        hot = df["tmean_c"] > thresh
        heatwave_days_per_year = hot.resample("YE").sum()
        print(f"\nHeatwave days per year (Tmean > 90th pct) — {name}:")
        print(heatwave_days_per_year.tail())

        if show_plots:
            plt.figure(figsize=(8, 4))
            heatwave_days_per_year.index = heatwave_days_per_year.index.year
            plt.bar(heatwave_days_per_year.index, heatwave_days_per_year.values)
            plt.title(f"Heatwave Days per Year — {name}")
            plt.xlabel("Year")
            plt.ylabel("Number of hot days")
            plt.tight_layout()


# ---------------------------------------------------------------------
# 3.5 Hydro-climate indices and melt proxies
# ---------------------------------------------------------------------


def analyze_hydro_indices(show_plots: bool = True) -> None:
    """Analyze PDD, freezing days, and heavy-precipitation indices."""
    llang, quil = load_hydro_indices()

    for name, df in (("Llanganuco", llang), ("Quilcayhuanca", quil)):
        print(f"\nHydrological indices summary — {name}:")
        print(df[["hydro_year", "pdd_degC_sum", "n_freezing_days", "n_days_p>=20.0mm"]].tail())

        if show_plots:
            plt.figure(figsize=(10, 5))
            plt.plot(df["hydro_year"], df["pdd_degC_sum"], marker="o", label="PDD sum")
            plt.title(f"Positive Degree Days by Hydrological Year — {name}")
            plt.xlabel("Hydrological year")
            plt.ylabel("PDD (°C·day)")
            plt.tight_layout()

            plt.figure(figsize=(10, 5))
            plt.plot(df["hydro_year"], df["n_freezing_days"], marker="o", label="Freezing days")
            plt.plot(df["hydro_year"], df["n_days_p>=20.0mm"], marker="s", label="Heavy precip days (>=20 mm)")
            plt.title(f"Freezing and Heavy-Precipitation Days — {name}")
            plt.xlabel("Hydrological year")
            plt.ylabel("Days")
            plt.legend()
            plt.tight_layout()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------


def run_all(show_plots: bool = True) -> None:
    """Run all core analyses sequentially."""
    analyze_monthly_trends_and_seasonality(show_plots=show_plots)
    analyze_edw(show_plots=show_plots)
    analyze_daily_extremes(show_plots=show_plots)
    analyze_hydro_indices(show_plots=show_plots)


if __name__ == "__main__":
    sns.set(style="whitegrid")
    run_all(show_plots=True)

