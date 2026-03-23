
Below is a **structured data analyst work plan** specifically tailored to:

✅ Your role: **data analysis + time-series modeling**
✅ Your datasets: Peru weather + glacier context
✅ Your deliverables this week
✅ Your final goal: **time-series model of glacier coverage linked to EDW**

This reads like a **professional research workflow** you can directly follow and also reuse in your methods memo.

---

# 🎯 Your Role This Week (Data Analyst Perspective)

You are responsible for building the **quantitative backbone** of the project:

> Detect climate trends → relate them to elevation → support glacier change modeling.

Your outputs must show:

1. Evidence of warming patterns
2. Seasonal climate structure
3. Data readiness for glacier modeling
4. Initial glacier time-series framework

---

# 🧭 MASTER WORKFLOW OVERVIEW

## Phase 1 — Research Framing (Day 1)

## Phase 2 — Data Inventory & Acquisition (Day 1–2)

## Phase 3 — Data Cleaning & Preprocessing (Day 2–3)

## Phase 4 — Weather Time-Series Analysis (Day 3–5)

## Phase 5 — Glacier Coverage Time-Series Modeling (Day 5–6)

## Phase 6 — Deliverables Preparation (Day 6–7)

---

# 1️⃣ Define Research Questions (EDW-Focused)

Your analysis should answer **testable quantitative questions**.

### Core Questions

### EDW Detection

* Does temperature increase vary by elevation zone?
* Are higher elevation stations warming faster?

### Climate Dynamics

* How do seasonal temperature cycles behave in the Andes?
* Is warming stronger during dry vs wet seasons?

### Glacier Implications

* Do temperature trends align with expected glacier retreat periods?
* Can temperature anomalies predict glacier coverage change?

---

### Example Analyst-Level Question (Use in proposal)

> “How do long-term temperature trends across elevation gradients influence glacier coverage variability in the Cordillera Blanca?”

---

# 2️⃣ Identify Study Regions (Your Analytical Framing)

You are working primarily with:

## 🇵🇪 Cordillera Blanca (Peru)

Key characteristics:

* Tropical glaciers
* Strong elevation gradients
* Sensitive to EDW

Stations in your datasets likely include:

* Lascar
* Llanganuco
* Quilcay watershed

Create a **station metadata table**:

| Station | Elevation | Latitude | Years Available | Variables |
| ------- | --------- | -------- | --------------- | --------- |

(This becomes part of your deliverable.)

---

# 3️⃣ Data Inventory & Metadata Table (DELIVERABLE #1)

Create a spreadsheet documenting:

| Dataset           | Type             | Temporal Resolution | Variables | Years      | Source | Use         |
| ----------------- | ---------------- | ------------------- | --------- | ---------- | ------ | ----------- |
| Lascar Monthly    | Weather          | Monthly             | Temp, RH  | 2006–2025  | Sensor | Trend       |
| Llanganuco Hourly | Weather          | Hourly              | Temp, RH  | 2004–2025  | Sensor | Seasonality |
| Quilcay           | Weather/Hydro    | Hourly              | Temp      | 2013–2025  | Sensor | Hydrology   |
| GLIMS             | Glacier polygons | Irregular           | Area      | Multi-year | GLIMS  | Glacier TS  |

---

# 4️⃣ Data Cleaning & Preprocessing

## Tools

* Python (primary)
* Excel (quick inspection)
* R (optional validation)

---

## Step A — Standardize Time

Convert all datasets to:

```
YYYY-MM-DD HH:MM:SS
```

Python:

```python
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')
```

---

## Step B — Handle Missing Data

Check:

* gaps
* sensor outages
* impossible values

Rules:

* Temp < -40°C → remove
* RH > 100% → remove

Fill gaps:

```python
df.interpolate(method='time')
```

---

## Step C — Create Derived Variables (IMPORTANT)

Create:

### Temperature anomaly

```python
df['anom'] = df['temp'] - df['temp'].mean()
```

### Monthly averages

```python
monthly = df.resample('M').mean()
```

### Seasonal labels

```python
df['month'] = df.index.month
```

---

# 5️⃣ Weather Dataset Exploration (Your Core Analysis)

## A. Preliminary Visualizations (DELIVERABLE #2)

### 1. Long-Term Temperature Trend

(Line plot)

Shows warming signal.

---

### 2. Seasonal Cycle

Average temperature by month.

Reveals:

* wet/dry season differences.

---

### 3. Temperature vs Humidity

Scatter plot.

Tests drying hypothesis.

---

### 4. Heatmap (VERY STRONG FIGURE)

Year vs Month colored by temperature.

Use:

* Tableau (best visualization)

---

# 6️⃣ Time-Series Analysis (Key Requirement)

## A. Decomposition

Separate trend + seasonality.

Python:

```python
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(monthly['temp'], model='additive')
result.plot()
```

Outputs:

* Trend
* Seasonal cycle
* Residual variability

---

## B. Trend Significance (Research-Level)

Use Mann–Kendall test:

```python
import pymannkendall as mk
mk.original_test(monthly['temp'])
```

You will report:

* trend direction
* significance

---

## C. Warming Rate

Regression:

```python
from sklearn.linear_model import LinearRegression
```

Result:

> °C per decade

(This is EDW evidence.)

---

# 7️⃣ Glacier Coverage Time-Series Model (YOUR MAIN MODEL)

Even before satellite processing, build framework.

---

## Step 1 — Acquire Glacier Data

Use:

* GLIMS glacier outlines
* Multi-year area estimates

Create dataset:

| Year | Glacier Area (km²) |

---

## Step 2 — Build Glacier Time Series

Plot:

```
Year vs Glacier Area
```

Expected:
Downward trend.

---

## Step 3 — Link Climate to Glacier Change

Merge datasets:

```python
merged = glacier.merge(temp_annual,on='year')
```

---

## Step 4 — Model

### Simple Climate–Glacier Model

[
GlacierArea = β_0 + β_1 Temperature + β_2 Precipitation
]

Python:

```python
LinearRegression().fit(X,y)
```

---

### Advanced Option (Strong ISR Contribution)

Lag model:

[
Area_t = Temp_{t-1}
]

Glaciers respond with delay.

---

# 8️⃣ Identify Seasonal Cycles Related to EDW

Test:

* Is warming strongest during dry season?
* Are nighttime temps increasing faster?

Compute:

```python
df.groupby('month')['temp'].mean()
```

Plot seasonal anomalies.

---

# 9️⃣ Tableau Dashboard (Recommended)

Create:

### Dashboard Components

1. Temperature trend line
2. Seasonal heatmap
3. Station comparison
4. Anomaly plot

This becomes presentation-ready.

---

# 🔟 Deliverable Guide

---

## ✅ Deliverable 1 — Project Proposal Outline

You contribute:

* Data sources
* Analytical framework
* Modeling approach

Include:

* Time-series methods
* EDW detection strategy

---

## ✅ Deliverable 2 — Data Inventory & Metadata

Create table documenting:

* dataset
* resolution
* variables
* coverage

---

## ✅ Deliverable 3 — Preliminary Time-Series Plots

Must include:

* temperature trend
* seasonal decomposition
* anomaly plot
* station comparison

---

## ✅ Deliverable 4 — Short Methods Memo

Structure:

### Data

Weather sensors + GLIMS

### Processing

Cleaning, interpolation, aggregation

### Analysis

* time-series decomposition
* trend tests
* regression modeling

### Goal

Link climate trends to glacier coverage.

---

# ⭐ Suggested Weekly Timeline

| Day | Task                           |
| --- | ------------------------------ |
| Mon | Research questions + inventory |
| Tue | Cleaning datasets              |
| Wed | Exploratory plots              |
| Thu | Time-series decomposition      |
| Fri | Glacier model prototype        |
| Sat | Tableau dashboard              |
| Sun | Methods memo                   |

---

# 🚀 What Will Impress Your ISR Team Most

Focus on producing:

✅ A clear warming trend figure
✅ Seasonal decomposition plot
✅ Initial glacier–temperature relationship model
✅ Clean metadata documentation

That shows you’ve moved from **data → evidence → modeling**.

---

