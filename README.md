# SkyNest - Airline Flight Delay Analysis and Prediction
### BDA Lab Prototype

A professional aviation-themed Streamlit web application for analysing and predicting Indian airline flight delays.

---

## Dataset Disclosure

> **All 100,000 records in SkyNest_India_Combined_Flight_Dataset.csv are SYNTHETIC / SIMULATED (data_type = SYNTHETIC_SIMULATED_NOT_REAL_FLIGHT_RECORD). No real airline operations data is used. This application is for academic and demonstration purposes only.**

| Property | Value |
|---|---|
| Rows | 100,000 |
| Columns | 49 |
| Airlines | 8 (IndiGo, Air India, SpiceJet, Air India Express, Vistara, Alliance Air, Akasa Air, Star Air) |
| Airports | 76 origin + 76 destination IATA codes |
| Date range | 2024-01-01 to 2026-09-26 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Data type | SYNTHETIC_SIMULATED_NOT_REAL_FLIGHT_RECORD |

---

## Setup and Run

### 1. Install dependencies

`
python -m pip install -r requirements.txt
`

### 2. Run the application

`
python -m streamlit run app.py
`

Open your browser at http://localhost:8501

---

## Application Pages

### Dashboard
- KPI cards: total flights, delayed flights (>=15 min), on-time, delay rate, avg delay
- Airline-wise delay rate (horizontal bar chart)
- Flight delay severity distribution (donut chart)
- Monthly delay trend 2024-2026 (dual-axis line chart)
- Day-of-week and season delay rate (bar charts)
- Weather condition impact (grouped bar chart)
- Scheduled departure hour vs delay rate (bar chart)

### Detailed Analysis
- Interactive filters: airline, origin, destination, year
- Airline summary table with CSV export
- Top 15 busiest origin airports
- Top 15 destinations by delay rate
- Top 20 routes bubble chart (volume vs delay rate)
- Data quality summary (missing values, duplicates, unique IDs)
- Route summary CSV export

### Predict Delay
- Random Forest classifier (120 trees, max_depth=12, class_weight=balanced)
- Trained on 30,000-row random sample for performance
- Features: airline, origin IATA, destination IATA, day of week, season, weather condition, airport traffic level, month, scheduled duration, distance, temperature, visibility, wind speed, precipitation, public holiday flag, departure hour
- Target: delay_target (1 = departure_delay_min >= 15 min)
- No target leakage: departure_delay_min, delay_category, and delay_target are excluded from predictors
- Outputs: predicted class, probability gauge, classification report, feature importance chart
- Evaluation metrics computed on held-out 20% test set

### Weather
- Live forecast from Open-Meteo API (free, no API key required)
- Forecasts available for today and up to 16 days ahead ONLY
- Hourly temperature, precipitation, wind speed, visibility, weather description
- Airport selector uses coordinates from the dataset (all 76 airports have valid coordinates)
- Historical delay context from the synthetic dataset shown alongside the live forecast
- Live weather is NOT fed into the prediction model

---

## Files

| File | Purpose |
|---|---|
| pp.py | Main Streamlit application |
| weather_api.py | Open-Meteo API integration module |
| 
equirements.txt | Python package dependencies |
| SkyNest_India_Combined_Flight_Dataset.csv | Synthetic flight dataset (do not overwrite) |

---

## Known Limitations

1. The dataset is entirely synthetic - delay statistics do not reflect actual Indian airline performance.
2. The ML model F1 score on the delayed class is modest (~19-25%) because the dataset is class-imbalanced (19.6% delayed) and the synthetic weather/delay relationship, while present, is not strongly deterministic.
3. Open-Meteo weather forecasts are only available for future dates (max 16 days ahead). Historical weather is unavailable via the free forecast endpoint.
4. Departure hour analysis from scheduled_departure (HH:MM string) is computed at runtime and is not pre-aggregated.

---

## Requirements

`
streamlit>=1.35.0
pandas>=2.0.0
plotly>=5.18.0
requests>=2.31.0
scikit-learn>=1.3.0
numpy>=1.25.0
`
