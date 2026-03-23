### Peru Lascar humidity EDW analysis

**EDW high/low selection (by elevation):**
- High: `LlanPort` at 4760 m
- Low:  `LlanWX` at 3850 m

**Monthly variables (from `monthly_clean.csv`):**
- Temperature: `temp_c_mean_month` (°C)
- Relative humidity: `rh_pct_mean_month` (%)
- Dew point: `dewpoint_c_mean_month` (°C)

---
### Station-wise trends and accelerated change (monthly)

Slopes are linear fits over the full monthly record (°C/decade for temperature & dew point; percentage-points/decade for RH). Acceleration is computed as: `(slope_last_10_years - slope_prev_10_years)` (rolling-decadal). For reference, 20-year acceleration is also computed when the record is long enough.

| Station | Elev (m) | T slope | T accel10 | RH slope | RH accel10 | Dew slope | Dew accel10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LlanWX | 3850 | -0.231 | 0.409 | 2.912 | 2.512 | 0.385 | 0.897 |
| LlanUp1 | 3955 | -0.117 | 0.073 | 0.431 | 4.170 | -0.111 | 0.993 |
| LlanUp2 | 4122 | 0.200 | 0.087 | 0.911 | -0.580 | 0.258 | -0.149 |
| LlanUp3 | 4355 | 0.437 | -0.459 | 1.858 | -2.970 | 0.689 | -1.231 |
| LlanUp4 | 4560 | 0.274 | 0.698 | 2.625 | -4.087 | 0.673 | -0.216 |
| LlanPort | 4760 | 2.158 | N/A | -0.690 | N/A | 2.050 | N/A |
---
### EDW-style humidity deltas (high - low; monthly means)

From daily series in `daily_clean.csv`, deltas are computed as high minus low, then resampled to monthly means.

| Metric | Full-record slope | Last10 slope | Prev10 slope | Accel10 |
|---|---:|---:|---:|---:|
| ΔRH (%p/decade) | -4.432 | -4.432 | N/A | N/A |
| Δdewpoint (°C/decade) | 1.517 | 1.517 | N/A | N/A |
---
### Moisture-temperature relationships (correlations)

Station-wise Pearson correlations computed across time (monthly means).

- Mean corr(T vs dew point): 0.140
- Mean corr(T vs RH): -0.147
- Mean corr(RH vs dew point): 0.950

Detrended correlations computed using residuals after removing each station's linear trend:
- Mean detrended corr(T vs dew point): 0.105
- Mean detrended corr(T vs RH): -0.165
- Mean detrended corr(RH vs dew point): 0.956

---
### Elevation dependence of accelerated warming (cross-station)

Correlation between station-level linear slopes across elevations:

| Pair (across stations) | Pearson r |
|---|---:|
| corr_temp_slope_vs_rh_slope | -0.690 |
| corr_temp_slope_vs_dewpoint_slope | 0.956 |
| corr_rh_slope_vs_dewpoint_slope | -0.458 |
| corr_temp_accel10_vs_rh_accel10 | -0.009 |
| corr_temp_accel10_vs_dewpoint_accel10 | 0.500 |
---
### Hourly diurnal regime change (VPD / RH)

Diurnal comparisons are computed for early (2006-2015) and late (2016-2025) years using `hourly_clean.csv` VPD: `vpd_hpa = sat_vapor_pressure_hpa - vapor_pressure_hpa`.

Plots saved: `diurnal_rh_early_late_high_low.png`, `diurnal_vpd_early_late_high_low.png`.
