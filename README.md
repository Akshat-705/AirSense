# AirSense: Air Quality Intelligence

AirSense is a Flask-based machine-learning dashboard that estimates AQI from pollutant measurements and presents historical air-quality trends for Indian cities.

## What it does

- Select a city and explore its AQI history.
- Estimate the latest AQI from PM2.5, PM10, NO2, CO, and SO2.
- Compare Random Forest, Gradient Boosting, and Linear Regression models.
- Display held-out test-set metrics (MAE, RMSE, and R²).
- Show model feature influence and pollutant charts.
- Provide a health outlook based on the estimated AQI.

## Machine-learning workflow

1. Load and numerically normalize pollutant columns.
2. Fill missing pollutant values with training-set medians.
3. Split the data into reproducible training and test sets.
4. Train three regression models and select the model with the lowest test MAE.
5. Refit the selected model on the available data for dashboard predictions.

The current evaluation produced these baseline results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 23.28 | 42.99 | 0.899 |
| Gradient Boosting | 24.40 | 44.16 | 0.894 |
| Linear Regression | 31.40 | 59.71 | 0.805 |

These metrics are estimates for this dataset and split, not a guarantee of real-world forecasting performance. The model currently estimates AQI from same-period pollutant readings; a future enhancement can add lagged readings for true next-day forecasting.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install flask pandas scikit-learn
python App.py
```

Open `http://127.0.0.1:5000` in a browser.

## Optional live monitoring

The dashboard can fetch current pollutant readings from OpenWeather. Create an OpenWeather API key, then set it in the terminal before starting Flask:

```powershell
$env:OPENWEATHER_API_KEY = "your-key-here"
python App.py
```

The browser refreshes live readings every 10 minutes. The key is read from the environment and is never placed in the HTML or committed to the project. OpenWeather's provider AQI uses a 1–5 scale; AirSense also displays its own model-estimated AQI using the project's trained model, so the two values should not be treated as identical scales.
