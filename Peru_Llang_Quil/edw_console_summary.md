### Monthly warming rates (from `edw_analysis.py`)

From the monthly trend analysis:

- **Llanganuco**: 0.404 °C/decade  
- **Quilcayhuanca**: 0.559 °C/decade  

These are linear warming rates estimated from the monthly mean temperature time series.

---

### EDW summary (ΔT and lapse rate — monthly means)

Using the EDW daily comparison dataset resampled to monthly means:

- **Mean ΔT (high − low)**: 0.263 °C  
- **Mean lapse rate**: 0.105 °C per 100 m  

Positive ΔT and lapse rate indicate that, on average, the higher station is slightly warmer than expected relative to the lower station over the analyzed period.

---

### Freezing days per year (daily data)

**Llanganuco — freezing days per year (last 5 years in output):**

| Year | Freezing days |
|------|---------------|
| 2021 | 0             |
| 2022 | 0             |
| 2023 | 0             |
| 2024 | 0             |
| 2025 | 0             |

**Quilcayhuanca — freezing days per year (last 5 years in output):**

| Year | Freezing days |
|------|---------------|
| 2021 | 0             |
| 2022 | 0             |
| 2023 | 0             |
| 2024 | 0             |
| 2025 | 0             |

At both elevations, recent years show no days with daily mean temperature ≤ 0 °C.

---

### Heatwave days per year (Tmean > 90th percentile)

Thresholds were defined separately for each station as the 90th percentile of daily mean temperature, then the number of days exceeding that threshold was counted per year.

**Llanganuco — heatwave days per year (last 5 years in output):**

| Year | Heatwave days |
|------|---------------|
| 2021 | 0             |
| 2022 | 10            |
| 2023 | 67            |
| 2024 | 107           |
| 2025 | 25            |

**Quilcayhuanca — heatwave days per year (last 5 years in output):**

| Year | Heatwave days |
|------|---------------|
| 2021 | 22            |
| 2022 | 13            |
| 2023 | 19            |
| 2024 | 105           |
| 2025 | 29            |

These counts suggest notable interannual variability and an apparent increase in hot days in some recent years, especially 2023–2024.

---

### Hydrological indices summary — Llanganuco

From `llanganuco_hydro_indices.csv` (last 5 hydrological years shown in the console):

| Hydro year | PDD sum (`pdd_degC_sum`) | Freezing days (`n_freezing_days`) | Heavy-precip days (`n_days_p>=20.0mm`) |
|-----------:|--------------------------|------------------------------------|----------------------------------------|
| 2022      | …                        | …                                  | 0                                      |
| 2023      | …                        | …                                  | 0                                      |
| 2024      | …                        | …                                  | 1                                      |
| 2025      | …                        | …                                  | 4                                      |
| 2026      | …                        | …                                  | 0                                      |

(*PDD and freezing-day values were not explicitly printed in the console snippet; fill them in from the CSV if needed for a report.*)

---

### Hydrological indices summary — Quilcayhuanca

From `quilcayhuanca_hydro_indices.csv` (last 5 hydrological years shown in the console):

| Hydro year | PDD sum (`pdd_degC_sum`) | Freezing days (`n_freezing_days`) | Heavy-precip days (`n_days_p>=20.0mm`) |
|-----------:|--------------------------|------------------------------------|----------------------------------------|
| 2022      | …                        | …                                  | 3                                      |
| 2023      | …                        | …                                  | 8                                      |
| 2024      | …                        | …                                  | 0                                      |
| 2025      | …                        | …                                  | 8                                      |
| 2026      | …                        | …                                  | 1                                      |

Again, PDD and freezing-day values can be read directly from the CSV for more detailed quantitative reporting; the console excerpt mainly highlights the heavy-precipitation counts.

