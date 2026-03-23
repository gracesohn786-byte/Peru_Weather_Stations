from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass(frozen=True)
class Paths:
    hourly: Path
    daily: Path
    monthly: Path
    out_dir: Path
    plots_dir: Path
    docs_md: Path


def _robust_parse_date(series: pd.Series) -> pd.Series:
    """
    Parse dates tolerant to M/D/Y vs D/M/Y.
    Returns NaT for unparseable values.
    """
    dt = pd.to_datetime(series, errors="coerce")
    # Heuristic: if too many values failed, try dayfirst.
    if dt.notna().mean() < 0.9:
        dt2 = pd.to_datetime(series, errors="coerce", dayfirst=True)
        if dt2.notna().mean() > dt.notna().mean():
            dt = dt2
    return dt


def _fractional_year(year: np.ndarray, month: np.ndarray) -> np.ndarray:
    return year + (month - 0.5) / 12.0


def _slope_per_decade(df: pd.DataFrame, year_col: str, month_col: str, value_col: str) -> float:
    series = df.dropna(subset=[value_col]).copy()
    if series.empty:
        return float("nan")

    years = _fractional_year(series[year_col].to_numpy(), series[month_col].to_numpy())
    y = series[value_col].to_numpy(dtype=float)

    # If all y are identical, polyfit is fine but slope will be ~0.
    slope_year, _ = np.polyfit(years, y, 1)
    return float(slope_year * 10.0)


def _windowed_slopes(
    df: pd.DataFrame,
    year_col: str,
    month_col: str,
    value_col: str,
    last_window_years: int = 20,
    prev_window_years: int = 20,
    min_months: int = 24,
) -> tuple[float, float, float]:
    """
    Returns (last_window_slope_decade, prev_window_slope_decade, acceleration = last - prev).
    """
    if df.empty:
        return float("nan"), float("nan"), float("nan")

    max_year = int(df[year_col].max())
    min_year = int(df[year_col].min())

    last_start = max_year - (last_window_years - 1)
    prev_end = last_start - 1
    prev_start = prev_end - (prev_window_years - 1)

    last_df = df[(df[year_col] >= last_start) & (df[year_col] <= max_year)]
    prev_df = df[(df[year_col] >= prev_start) & (df[year_col] <= prev_end)]

    # Ensure enough non-missing monthly points.
    last_n = last_df[value_col].notna().sum()
    prev_n = prev_df[value_col].notna().sum()

    last_slope = float("nan")
    prev_slope = float("nan")
    if last_n >= min_months:
        last_slope = _slope_per_decade(last_df, year_col, month_col, value_col)
    if prev_n >= min_months:
        prev_slope = _slope_per_decade(prev_df, year_col, month_col, value_col)

    accel = float("nan")
    if np.isfinite(last_slope) and np.isfinite(prev_slope):
        accel = last_slope - prev_slope
    return float(last_slope), float(prev_slope), float(accel)


def _residuals_linear_trend(df: pd.DataFrame, year_col: str, month_col: str, value_col: str) -> pd.Series:
    """
    Fit linear model to non-missing values and return residuals (value - fitted).
    """
    series = df.dropna(subset=[value_col]).copy()
    if series.empty:
        return pd.Series(index=df.index, data=np.nan, dtype=float)

    x = _fractional_year(series[year_col].to_numpy(), series[month_col].to_numpy())
    y = series[value_col].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    x_all = _fractional_year(df[year_col].to_numpy(), df[month_col].to_numpy())
    fitted = slope * x_all + intercept
    residual = df[value_col] - fitted
    return residual.astype(float)


def _pearson_safe(a: pd.Series, b: pd.Series) -> float:
    x = a.astype(float)
    y = b.astype(float)
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def main() -> None:
    lascar_dir = Path(__file__).resolve().parent
    derived_dir = lascar_dir / "derived"

    root_dir = lascar_dir.parent
    docs_dir = root_dir / "docs"

    paths = Paths(
        hourly=derived_dir / "hourly_clean.csv",
        daily=derived_dir / "daily_clean.csv",
        monthly=derived_dir / "monthly_clean.csv",
        out_dir=derived_dir,
        plots_dir=derived_dir / "plots" / "peru_lascar_humidity_edw",
        docs_md=docs_dir / "peru_lascar_humidity_edw_summary.md",
    )
    paths.plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading derived CSVs...")
    hourly = pd.read_csv(paths.hourly)
    hourly["datetime"] = _robust_parse_date(hourly["datetime"])
    hourly = hourly.dropna(subset=["datetime"])
    hourly["hour"] = hourly["datetime"].dt.hour
    hourly["year"] = hourly["datetime"].dt.year

    daily = pd.read_csv(paths.daily)
    daily["date"] = _robust_parse_date(daily["date"])
    daily = daily.dropna(subset=["date"])

    monthly = pd.read_csv(paths.monthly)
    monthly["monthly_date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))

    station_table = (
        monthly.groupby("station_id", as_index=False)
        .agg(elevation_m=("elevation_m", "mean"))
        .sort_values("elevation_m")
    )
    if station_table.empty:
        raise RuntimeError("No stations found in monthly_clean.csv")

    high_station = station_table.iloc[-1]["station_id"]
    low_station = station_table.iloc[0]["station_id"]
    high_elev = float(station_table.iloc[-1]["elevation_m"])
    low_elev = float(station_table.iloc[0]["elevation_m"])

    print("Stations discovered:", station_table.to_dict(orient="records"))
    print("EDW high/low selection:")
    print(f"  high: {high_station} (elev={high_elev})")
    print(f"  low:  {low_station} (elev={low_elev})")

    # ------------------------------------------------------------------
    # 1) Station-wise trends + acceleration (monthly)
    # ------------------------------------------------------------------
    vars_monthly = {
        "temp_c": "temp_c_mean_month",
        "rh_pct": "rh_pct_mean_month",
        "dewpoint_c": "dewpoint_c_mean_month",
    }

    rows = []
    corr_rows = []

    for station_id, grp in monthly.groupby("station_id"):
        elev = float(grp["elevation_m"].mean())

        trend_row = {"station_id": station_id, "elevation_m": elev}
        for short_name, col in vars_monthly.items():
            trend_row[f"{short_name}_slope_decade"] = _slope_per_decade(
                grp, "year", "month", col
            )
            # 20-year acceleration (may be N/A if record is too short)
            last20_slope, prev20_slope, accel20 = _windowed_slopes(
                grp,
                "year",
                "month",
                col,
                last_window_years=20,
                prev_window_years=20,
            )
            trend_row[f"{short_name}_slope_last20_decade"] = last20_slope
            trend_row[f"{short_name}_slope_prev20_decade"] = prev20_slope
            trend_row[f"{short_name}_accel20_decade"] = accel20

            # Rolling-decadal acceleration: last 10 years minus previous 10 years
            last10_slope, prev10_slope, accel10 = _windowed_slopes(
                grp,
                "year",
                "month",
                col,
                last_window_years=10,
                prev_window_years=10,
                min_months=10,
            )
            trend_row[f"{short_name}_slope_last10_decade"] = last10_slope
            trend_row[f"{short_name}_slope_prev10_decade"] = prev10_slope
            trend_row[f"{short_name}_accel10_decade"] = accel10

        rows.append(trend_row)

        # Relationship correlations (station-wise across time)
        temp = grp[vars_monthly["temp_c"]]
        rh = grp[vars_monthly["rh_pct"]]
        dew = grp[vars_monthly["dewpoint_c"]]

        corr_rows.append(
            {
                "station_id": station_id,
                "elevation_m": elev,
                "corr_temp_vs_dewpoint": _pearson_safe(temp, dew),
                "corr_temp_vs_rh": _pearson_safe(temp, rh),
                "corr_rh_vs_dewpoint": _pearson_safe(rh, dew),
            }
        )

        # Detrended correlations (residuals after removing linear trends per series)
        temp_res = _residuals_linear_trend(grp, "year", "month", vars_monthly["temp_c"])
        rh_res = _residuals_linear_trend(grp, "year", "month", vars_monthly["rh_pct"])
        dew_res = _residuals_linear_trend(grp, "year", "month", vars_monthly["dewpoint_c"])

        corr_rows[-1].update(
            {
                "corr_detr_temp_vs_dewpoint": _pearson_safe(temp_res, dew_res),
                "corr_detr_temp_vs_rh": _pearson_safe(temp_res, rh_res),
                "corr_detr_rh_vs_dewpoint": _pearson_safe(rh_res, dew_res),
            }
        )

    station_trends = pd.DataFrame(rows)
    station_trends_path = paths.out_dir / "station_monthly_trends_rh_dewpoint.csv"
    station_trends.to_csv(station_trends_path, index=False)

    station_corrs = pd.DataFrame(corr_rows)
    station_corrs_path = paths.out_dir / "station_monthly_correlations_rh_dewpoint.csv"
    station_corrs.to_csv(station_corrs_path, index=False)

    # Cross-station relationships: correlate trend slopes across stations
    def _corr_between_slopes(col_a: str, col_b: str) -> float:
        df = station_trends[["station_id", "elevation_m", col_a, col_b]].copy()
        df = df.dropna(subset=[col_a, col_b])
        if len(df) < 3:
            return float("nan")
        return float(np.corrcoef(df[col_a].astype(float), df[col_b].astype(float))[0, 1])

    cross_stats = {
        "corr_temp_slope_vs_rh_slope": _corr_between_slopes("temp_c_slope_decade", "rh_pct_slope_decade"),
        "corr_temp_slope_vs_dewpoint_slope": _corr_between_slopes(
            "temp_c_slope_decade", "dewpoint_c_slope_decade"
        ),
        "corr_rh_slope_vs_dewpoint_slope": _corr_between_slopes(
            "rh_pct_slope_decade", "dewpoint_c_slope_decade"
        ),
        "corr_temp_accel10_vs_rh_accel10": _corr_between_slopes(
            "temp_c_accel10_decade", "rh_pct_accel10_decade"
        ),
        "corr_temp_accel10_vs_dewpoint_accel10": _corr_between_slopes(
            "temp_c_accel10_decade", "dewpoint_c_accel10_decade"
        ),
    }

    cross_path = paths.out_dir / "cross_station_trend_relationships_rh_dewpoint.csv"
    pd.DataFrame([cross_stats]).to_csv(cross_path, index=False)

    # ------------------------------------------------------------------
    # 2) EDW-style humidity deltas (daily -> monthly)
    # ------------------------------------------------------------------
    daily_high = daily[daily["station_id"] == high_station].copy()
    daily_low = daily[daily["station_id"] == low_station].copy()

    if daily_high.empty or daily_low.empty:
        raise RuntimeError("High/low station daily series empty; check station selection.")

    edw_daily = daily_high.merge(
        daily_low,
        on="date",
        suffixes=("_high", "_low"),
        how="inner",
    )

    edw_daily["delta_rh_pct_mean"] = edw_daily["rh_pct_mean_high"] - edw_daily["rh_pct_mean_low"]
    edw_daily["delta_dewpoint_c_mean"] = edw_daily["dewpoint_c_mean_high"] - edw_daily["dewpoint_c_mean_low"]
    edw_daily["delta_temp_c_mean"] = edw_daily["temp_c_mean_high"] - edw_daily["temp_c_mean_low"]

    edw_daily_path = paths.out_dir / "edw_daily_humidity_deltas_high_minus_low.csv"
    edw_daily[
        ["date", "delta_temp_c_mean", "delta_rh_pct_mean", "delta_dewpoint_c_mean", "n_obs_day_high", "n_obs_day_low"]
    ].to_csv(edw_daily_path, index=False)

    edw_daily_ts = edw_daily.set_index("date").sort_index()
    edw_monthly = pd.DataFrame(
        {
            "delta_temp_c_mean": edw_daily_ts["delta_temp_c_mean"].resample("ME").mean(),
            "delta_rh_pct_mean": edw_daily_ts["delta_rh_pct_mean"].resample("ME").mean(),
            "delta_dewpoint_c_mean": edw_daily_ts["delta_dewpoint_c_mean"].resample("ME").mean(),
        }
    ).reset_index()

    edw_monthly["year"] = edw_monthly["date"].dt.year
    edw_monthly["month"] = edw_monthly["date"].dt.month
    edw_monthly_path = paths.out_dir / "edw_monthly_humidity_deltas_high_minus_low.csv"
    edw_monthly.to_csv(edw_monthly_path, index=False)

    # EDW delta slopes/acceleration over monthly deltas
    edw_trend_row = {
        "edw_high_station": str(high_station),
        "edw_high_elev_m": high_elev,
        "edw_low_station": str(low_station),
        "edw_low_elev_m": low_elev,
    }

    for short_name, col in [
        ("delta_temp_c_mean", "delta_temp_c_mean"),
        ("delta_rh_pct_mean", "delta_rh_pct_mean"),
        ("delta_dewpoint_c_mean", "delta_dewpoint_c_mean"),
    ]:
        slope_full = _slope_per_decade(edw_monthly, "year", "month", col)
        last20_slope, prev20_slope, accel20 = _windowed_slopes(
            edw_monthly,
            "year",
            "month",
            col,
            last_window_years=20,
            prev_window_years=20,
        )
        last10_slope, prev10_slope, accel10 = _windowed_slopes(
            edw_monthly,
            "year",
            "month",
            col,
            last_window_years=10,
            prev_window_years=10,
            min_months=10,
        )
        edw_trend_row[f"{col}_slope_decade"] = slope_full
        edw_trend_row[f"{col}_slope_last20_decade"] = last20_slope
        edw_trend_row[f"{col}_slope_prev20_decade"] = prev20_slope
        edw_trend_row[f"{col}_accel20_decade"] = accel20
        edw_trend_row[f"{col}_slope_last10_decade"] = last10_slope
        edw_trend_row[f"{col}_slope_prev10_decade"] = prev10_slope
        edw_trend_row[f"{col}_accel10_decade"] = accel10

    edw_trends_path = paths.out_dir / "edw_monthly_humidity_delta_trends.csv"
    pd.DataFrame([edw_trend_row]).to_csv(edw_trends_path, index=False)

    # ------------------------------------------------------------------
    # 3) Hourly diurnal cycle + VPD
    # ------------------------------------------------------------------
    hourly = hourly.copy()
    hourly["vpd_hpa"] = hourly["sat_vapor_pressure_hpa"] - hourly["vapor_pressure_hpa"]
    hourly = hourly.replace([np.inf, -np.inf], np.nan)

    # Use first/last 10 years over the full hourly dataset record.
    overall_min_year = int(hourly["year"].min())
    overall_max_year = int(hourly["year"].max())
    early_start, early_end = overall_min_year, overall_min_year + 9
    late_start, late_end = overall_max_year - 9, overall_max_year

    diurnal_out = []
    for station_id in [high_station, low_station]:
        h = hourly[hourly["station_id"] == station_id].copy()
        if h.empty:
            continue

        h["date_only"] = h["datetime"].dt.date
        early = h[(h["year"] >= early_start) & (h["year"] <= early_end)]
        late = h[(h["year"] >= late_start) & (h["year"] <= late_end)]

        def _hourly_climatology(df: pd.DataFrame) -> pd.DataFrame:
            return (
                df.groupby("hour")
                .agg(
                    rh_pct_mean=("rh_pct", "mean"),
                    dewpoint_c_mean=("dewpoint_c", "mean"),
                    vpd_hpa_mean=("vpd_hpa", "mean"),
                )
                .reset_index()
            )

        early_clim = _hourly_climatology(early)
        late_clim = _hourly_climatology(late)
        early_clim = early_clim.set_index("hour")
        late_clim = late_clim.set_index("hour")

        # Diurnal range from daily max-min averaged over early/late windows.
        def _diurnal_range_mean(df: pd.DataFrame) -> dict[str, float]:
            daily_ranges = df.groupby("date_only").agg(
                rh_range=("rh_pct", lambda x: float(np.nanmax(x) - np.nanmin(x))),
                dewpoint_range=("dewpoint_c", lambda x: float(np.nanmax(x) - np.nanmin(x))),
                vpd_range=("vpd_hpa", lambda x: float(np.nanmax(x) - np.nanmin(x))),
            )
            return {
                "rh_diurnal_range_mean": float(daily_ranges["rh_range"].mean()),
                "dewpoint_diurnal_range_mean": float(daily_ranges["dewpoint_range"].mean()),
                "vpd_diurnal_range_mean": float(daily_ranges["vpd_range"].mean()),
            }

        early_range = _diurnal_range_mean(early)
        late_range = _diurnal_range_mean(late)

        elev = float(station_table.set_index("station_id").loc[station_id, "elevation_m"])
        diurnal_out.append(
            {
                "station_id": station_id,
                "elevation_m": elev,
                "early_year_start": early_start,
                "early_year_end": early_end,
                "late_year_start": late_start,
                "late_year_end": late_end,
                **{f"early_{k}": v for k, v in early_range.items()},
                **{f"late_{k}": v for k, v in late_range.items()},
            }
        )

        # Store climatologies for plotting (attach station label + regime)
        early_clim["regime"] = "early"
        late_clim["regime"] = "late"
        early_clim = early_clim.reset_index()
        late_clim = late_clim.reset_index()
        early_clim["station_id"] = station_id
        late_clim["station_id"] = station_id
        diurnal_clim = pd.concat([early_clim, late_clim], ignore_index=True)

    diurnal_clim_path = paths.out_dir / "hourly_diurnal_climatology_early_late.csv"
    # Note: diurnal_clim variable exists only after loop; ensure it's defined.
    # We'll recompute for safety below using stored station lists.
    out_frames = []
    for station_id in [high_station, low_station]:
        h = hourly[hourly["station_id"] == station_id].copy()
        if h.empty:
            continue
        early = h[(h["year"] >= early_start) & (h["year"] <= early_end)]
        late = h[(h["year"] >= late_start) & (h["year"] <= late_end)]

        def _hourly_climatology(df: pd.DataFrame, regime: str) -> pd.DataFrame:
            dfc = (
                df.groupby("hour")
                .agg(
                    rh_pct_mean=("rh_pct", "mean"),
                    dewpoint_c_mean=("dewpoint_c", "mean"),
                    vpd_hpa_mean=("vpd_hpa", "mean"),
                )
                .reset_index()
            )
            dfc["regime"] = regime
            dfc["station_id"] = station_id
            return dfc

        out_frames.append(_hourly_climatology(early, "early"))
        out_frames.append(_hourly_climatology(late, "late"))

    diurnal_clim = pd.concat(out_frames, ignore_index=True)
    diurnal_clim.to_csv(diurnal_clim_path, index=False)

    diurnal_metrics = pd.DataFrame(diurnal_out)
    diurnal_metrics_path = paths.out_dir / "hourly_diurnal_range_metrics_early_late.csv"
    diurnal_metrics.to_csv(diurnal_metrics_path, index=False)

    # Plot diurnal climatology comparisons
    # We'll focus on RH and VPD for interpretability.
    plt.figure(figsize=(12, 4))
    for station_id, marker in [(high_station, "o"), (low_station, "s")]:
        tmp = diurnal_clim[diurnal_clim["station_id"] == station_id]
        for regime, alpha in [("early", 0.6), ("late", 1.0)]:
            t = tmp[tmp["regime"] == regime]
            plt.plot(t["hour"], t["rh_pct_mean"], label=f"{station_id} RH {regime}", alpha=alpha, marker=marker)
    plt.title(f"Diurnal cycle of RH (early {early_start}-{early_end} vs late {late_start}-{late_end})")
    plt.xlabel("Hour of day")
    plt.ylabel("Mean RH (%)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(paths.plots_dir / "diurnal_rh_early_late_high_low.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 4))
    for station_id, marker in [(high_station, "o"), (low_station, "s")]:
        tmp = diurnal_clim[diurnal_clim["station_id"] == station_id]
        for regime, alpha in [("early", 0.6), ("late", 1.0)]:
            t = tmp[tmp["regime"] == regime]
            plt.plot(t["hour"], t["vpd_hpa_mean"], label=f"{station_id} VPD {regime}", alpha=alpha, marker=marker)
    plt.title(f"Diurnal cycle of VPD (hPa) (early {early_start}-{early_end} vs late {late_start}-{late_end})")
    plt.xlabel("Hour of day")
    plt.ylabel("Mean VPD (hPa)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(paths.plots_dir / "diurnal_vpd_early_late_high_low.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # 4) Plots for warming/moistening and EDW deltas
    # ------------------------------------------------------------------
    # Scatter: slopes vs elevation
    scatter_df = station_trends.copy()
    plt.figure(figsize=(12, 9))
    for i, (y_col, title) in enumerate(
        [
            ("temp_c_slope_decade", "Temperature warming rate (°C/decade)"),
            ("rh_pct_slope_decade", "RH trend (percentage points/decade)"),
            ("dewpoint_c_slope_decade", "Dew point trend (°C/decade)"),
        ]
    ):
        ax = plt.subplot(3, 1, i + 1)
        ax.scatter(scatter_df["elevation_m"], scatter_df[y_col], s=40)
        ax.set_title(title)
        ax.set_xlabel("Elevation (m)")
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths.plots_dir / "warming_and_moisture_trends_vs_elevation.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Monthly mean time series for high and low
    monthly_series = monthly[monthly["station_id"].isin([high_station, low_station])].copy()
    monthly_series = monthly_series.sort_values(["station_id", "monthly_date"])
    plt.figure(figsize=(12, 9))
    for i, (col, title) in enumerate(
        [
            ("temp_c_mean_month", "Monthly mean temperature (°C)"),
            ("rh_pct_mean_month", "Monthly mean RH (%)"),
            ("dewpoint_c_mean_month", "Monthly mean dew point (°C)"),
        ]
    ):
        ax = plt.subplot(3, 1, i + 1)
        for station_id in [high_station, low_station]:
            s = monthly_series[monthly_series["station_id"] == station_id]
            ax.plot(s["monthly_date"], s[col], label=f"{station_id} ({s['elevation_m'].mean():.0f} m)", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths.plots_dir / "monthly_means_high_vs_low.png", dpi=200, bbox_inches="tight")
    plt.close()

    # EDW delta plots (RH and dew point)
    edw_monthly_ts = edw_monthly.copy().sort_values("date")
    plt.figure(figsize=(12, 8))
    for i, (col, title) in enumerate(
        [
            ("delta_rh_pct_mean", "EDW delta RH (high - low), monthly mean"),
            ("delta_dewpoint_c_mean", "EDW delta dew point (high - low), monthly mean"),
        ]
    ):
        ax = plt.subplot(2, 1, i + 1)
        ax.plot(edw_monthly_ts["date"], edw_monthly_ts[col], alpha=0.6, label="Monthly mean")
        ax.plot(
            edw_monthly_ts["date"],
            edw_monthly_ts[col].rolling(12, min_periods=6).mean(),
            linewidth=2,
            label="12-mo mean",
        )
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths.plots_dir / "edw_monthly_deltas_rh_dewpoint.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Seasonal heatmap for RH at EDW high station
    rh_high = monthly[monthly["station_id"] == high_station][["year", "month", "rh_pct_mean_month"]].copy()
    pivot = rh_high.pivot(index="year", columns="month", values="rh_pct_mean_month")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap="coolwarm", cbar_kws={"label": "RH (%)"})
    plt.title(f"Seasonal heatmap: RH at EDW high ({high_station})")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.tight_layout()
    plt.savefig(paths.plots_dir / "edw_high_rh_seasonal_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # 5) Markdown report
    # ------------------------------------------------------------------
    # Load EDW trends
    edw_trends = pd.read_csv(edw_trends_path)
    # Cross-station stats
    cross_df = pd.read_csv(cross_path)
    cross_row = cross_df.iloc[0].to_dict()

    # Select top-level numeric extracts
    def _fmt(x: float, nd: int = 3) -> str:
        if pd.isna(x):
            return "N/A"
        return f"{x:.{nd}f}"

    # Produce station trends table (compact)
    keep_cols = [
        "station_id",
        "elevation_m",
        "temp_c_slope_decade",
        "temp_c_accel10_decade",
        "rh_pct_slope_decade",
        "rh_pct_accel10_decade",
        "dewpoint_c_slope_decade",
        "dewpoint_c_accel10_decade",
    ]
    station_tbl = station_trends[keep_cols].sort_values("elevation_m")

    # Compute means of within-station correlations (ignoring NaNs)
    corr_means = station_corrs[[
        "corr_temp_vs_dewpoint",
        "corr_temp_vs_rh",
        "corr_rh_vs_dewpoint",
        "corr_detr_temp_vs_dewpoint",
        "corr_detr_temp_vs_rh",
        "corr_detr_rh_vs_dewpoint",
    ]].mean(numeric_only=True).to_dict()

    report_lines: list[str] = []
    report_lines.append("### Peru Lascar humidity EDW analysis")
    report_lines.append("")
    report_lines.append("**EDW high/low selection (by elevation):**")
    report_lines.append(f"- High: `{high_station}` at {high_elev:.0f} m")
    report_lines.append(f"- Low:  `{low_station}` at {low_elev:.0f} m")
    report_lines.append("")
    report_lines.append("**Monthly variables (from `monthly_clean.csv`):**")
    report_lines.append("- Temperature: `temp_c_mean_month` (°C)")
    report_lines.append("- Relative humidity: `rh_pct_mean_month` (%)")
    report_lines.append("- Dew point: `dewpoint_c_mean_month` (°C)")
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("### Station-wise trends and accelerated change (monthly)")
    report_lines.append("")
    report_lines.append(
        "Slopes are linear fits over the full monthly record (°C/decade for temperature & dew point; percentage-points/decade for RH). "
        "Acceleration is computed as: `(slope_last_10_years - slope_prev_10_years)` (rolling-decadal). "
        "For reference, 20-year acceleration is also computed when the record is long enough."
    )
    report_lines.append("")
    report_lines.append("| Station | Elev (m) | T slope | T accel10 | RH slope | RH accel10 | Dew slope | Dew accel10 |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in station_tbl.iterrows():
        report_lines.append(
            f"| {r['station_id']} | {r['elevation_m']:.0f} | {_fmt(r['temp_c_slope_decade'])} | {_fmt(r['temp_c_accel10_decade'])} | "
            f"{_fmt(r['rh_pct_slope_decade'])} | {_fmt(r['rh_pct_accel10_decade'])} | "
            f"{_fmt(r['dewpoint_c_slope_decade'])} | {_fmt(r['dewpoint_c_accel10_decade'])} |"
        )

    report_lines.append("---")
    report_lines.append("### EDW-style humidity deltas (high - low; monthly means)")
    report_lines.append("")
    report_lines.append("From daily series in `daily_clean.csv`, deltas are computed as high minus low, then resampled to monthly means.")
    report_lines.append("")
    report_lines.append("| Metric | Full-record slope | Last10 slope | Prev10 slope | Accel10 |")
    report_lines.append("|---|---:|---:|---:|---:|")
    for col, label in [
        ("delta_rh_pct_mean", "ΔRH (%p/decade)"),
        ("delta_dewpoint_c_mean", "Δdewpoint (°C/decade)"),
    ]:
        report_lines.append(
            f"| {label} | {_fmt(edw_trends[f'{col}_slope_decade'].iloc[0])} | {_fmt(edw_trends[f'{col}_slope_last10_decade'].iloc[0])} | "
            f"{_fmt(edw_trends[f'{col}_slope_prev10_decade'].iloc[0])} | {_fmt(edw_trends[f'{col}_accel10_decade'].iloc[0])} |"
        )

    report_lines.append("---")
    report_lines.append("### Moisture-temperature relationships (correlations)")
    report_lines.append("")
    report_lines.append("Station-wise Pearson correlations computed across time (monthly means).")
    report_lines.append("")
    report_lines.append(
        f"- Mean corr(T vs dew point): {_fmt(corr_means['corr_temp_vs_dewpoint'], 3)}"
    )
    report_lines.append(
        f"- Mean corr(T vs RH): {_fmt(corr_means['corr_temp_vs_rh'], 3)}"
    )
    report_lines.append(
        f"- Mean corr(RH vs dew point): {_fmt(corr_means['corr_rh_vs_dewpoint'], 3)}"
    )
    report_lines.append("")
    report_lines.append("Detrended correlations computed using residuals after removing each station's linear trend:")
    report_lines.append(
        f"- Mean detrended corr(T vs dew point): {_fmt(corr_means['corr_detr_temp_vs_dewpoint'], 3)}"
    )
    report_lines.append(
        f"- Mean detrended corr(T vs RH): {_fmt(corr_means['corr_detr_temp_vs_rh'], 3)}"
    )
    report_lines.append(
        f"- Mean detrended corr(RH vs dew point): {_fmt(corr_means['corr_detr_rh_vs_dewpoint'], 3)}"
    )
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("### Elevation dependence of accelerated warming (cross-station)")
    report_lines.append("")
    report_lines.append("Correlation between station-level linear slopes across elevations:")
    report_lines.append("")
    report_lines.append("| Pair (across stations) | Pearson r |")
    report_lines.append("|---|---:|")
    for k, v in [
        ("corr_temp_slope_vs_rh_slope", cross_row.get("corr_temp_slope_vs_rh_slope")),
        ("corr_temp_slope_vs_dewpoint_slope", cross_row.get("corr_temp_slope_vs_dewpoint_slope")),
        ("corr_rh_slope_vs_dewpoint_slope", cross_row.get("corr_rh_slope_vs_dewpoint_slope")),
        ("corr_temp_accel10_vs_rh_accel10", cross_row.get("corr_temp_accel10_vs_rh_accel10")),
        ("corr_temp_accel10_vs_dewpoint_accel10", cross_row.get("corr_temp_accel10_vs_dewpoint_accel10")),
    ]:
        report_lines.append(f"| {k} | {_fmt(float(v), 3) if pd.notna(v) else 'N/A'} |")

    report_lines.append("---")
    report_lines.append("### Hourly diurnal regime change (VPD / RH)")
    report_lines.append("")
    report_lines.append(
        f"Diurnal comparisons are computed for early ({overall_min_year}-{overall_min_year + 9}) and late ({overall_max_year - 9}-{overall_max_year}) years "
        f"using `hourly_clean.csv` VPD: `vpd_hpa = sat_vapor_pressure_hpa - vapor_pressure_hpa`."
    )
    report_lines.append("")
    report_lines.append(
        f"Plots saved: `diurnal_rh_early_late_high_low.png`, `diurnal_vpd_early_late_high_low.png`."
    )
    report_lines.append("")

    paths.docs_md.write_text("\n".join(report_lines), encoding="utf-8")
    print("Wrote report:", paths.docs_md)

    # ------------------------------------------------------------------
    # 6) Verification: check non-empty outputs
    # ------------------------------------------------------------------
    output_csvs = [
        station_trends_path,
        station_corrs_path,
        cross_path,
        edw_daily_path,
        edw_monthly_path,
        edw_trends_path,
        diurnal_clim_path,
        diurnal_metrics_path,
    ]
    for p in output_csvs:
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"Expected output CSV missing/empty: {p}")

    output_pngs = [
        "warming_and_moisture_trends_vs_elevation.png",
        "monthly_means_high_vs_low.png",
        "edw_monthly_deltas_rh_dewpoint.png",
        "edw_high_rh_seasonal_heatmap.png",
        "diurnal_rh_early_late_high_low.png",
        "diurnal_vpd_early_late_high_low.png",
    ]
    for name in output_pngs:
        p = paths.plots_dir / name
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"Expected plot missing/empty: {p}")

    print("All outputs verified.")


if __name__ == "__main__":
    main()

