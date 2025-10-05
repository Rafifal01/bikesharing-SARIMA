import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib, json

# =======================
# Load data & model
# =======================
@st.cache_resource
def load_models():
    gb_model = joblib.load("best_hourly_model.pkl")   # Gradient Boosting (trained)
    data = pd.read_csv("data_bike_sharing.csv")       # dataset
    return gb_model, data

gb_model, data = load_models()

# =======================
# SARIMA Fitting Helper
# =======================
def train_sarima(train_series, order=(1,1,1), seasonal=(1,1,1,7)):
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal, enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    return fitted

# =======================
# Hybrid Prediction
# =======================
def hybrid_predict(gb_pred, sarima_pred):
    # Dynamic weight based on RMSE
    gb_error = np.std(gb_pred - sarima_pred) + 1e-6
    sarima_error = np.std(sarima_pred - gb_pred) + 1e-6
    w_gb = 1 / gb_error
    w_sarima = 1 / sarima_error
    w_sum = w_gb + w_sarima
    w_gb /= w_sum
    w_sarima /= w_sum
    return (w_gb * gb_pred) + (w_sarima * sarima_pred), w_gb, w_sarima

# =======================
# Streamlit UI
# =======================
st.set_page_config(
    page_title="Bike Rental Forecast",
    page_icon="🚴",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🚴 Bike Rental Forecasting App")
st.markdown("Hybrid Model: **Gradient Boosting + SARIMA**")

st.markdown("""
<style>
.big-font {font-size:22px !important;}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">Prediksi jumlah peminjaman sepeda berdasarkan kondisi cuaca dan kalender.</p>', unsafe_allow_html=True)

granularity = st.selectbox("Select Forecast Granularity:", ["Daily", "Weekly", "Monthly"])

# Pilih periode prediksi
n_periods = st.slider("How many periods to forecast?", 7, 60, 14)

with st.sidebar:
    st.header("🔧 Input Kondisi")
    
    st.subheader("🔧 Input Kondisi Cuaca & Kalender")

    season_map = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
    weathersit_map = {"Clear": 1, "Mist": 2, "Light Rain": 3, "Heavy Rain": 4}

    season = st.selectbox("Season", list(season_map.keys()))
    weathersit = st.selectbox("Weather", list(weathersit_map.keys()))
    temp = st.slider("Temperature (normalized 0–1)", 0.0, 1.0, 0.3, 0.01)
    hum = st.slider("Humidity (0–1)", 0.0, 1.0, 0.5, 0.01)
    month = st.slider("Month", 1, 12, 6, 1)
    holiday = st.checkbox("Holiday?", value=False)

# Susun input ke dataframe sesuai feature order model
feature_cols = gb_model.feature_names_in_
user_input = {
    "season": season_map[season],
    "holiday": int(holiday),
    "weathersit": weathersit_map[weathersit],
    "temp": temp,
    "atemp": temp,  # gunakan temp yang sama
    "hum": hum,
    "month": month,
    "lag_1": 0,
    "lag_24": 0,
    "roll_mean_3": 0,
    "roll_mean_24": 0,
    "hour": 12,  # default jam siang
    "dayofweek": 1,  # default Senin
    "is_weekend": 0,
    "is_rush": 0
}

for col in feature_cols:
    if col not in user_input:
        user_input[col] = 0

X_input = pd.DataFrame([user_input])[feature_cols]

if st.button("Run Forecast"):
    # ============ Data Preparation ============ 
    if granularity == "Daily":
        df = data.copy()
        df['dteday'] = pd.to_datetime(df['dteday'])
        df = df.drop_duplicates(subset="dteday")
        df = df.set_index("dteday").asfreq("D")
    elif granularity == "Weekly":
        df = data.copy()
        df['dteday'] = pd.to_datetime(df['dteday'])
        df = df.set_index("dteday").resample("W").sum()
    else:
        df = data.copy()
        df['dteday'] = pd.to_datetime(df['dteday'])
        df = df.set_index("dteday").resample("M").sum()

    train_series = df['cnt']

    # ============ SARIMA ============
    sarima_model = train_sarima(train_series)
    sarima_forecast = sarima_model.forecast(steps=n_periods)

    # ============ GB MULTISTEP ============
    gb_pred = []
    last_vals = train_series.copy()
    for i in range(n_periods):
        # Update lag & rolling mean dari data terakhir
        user_input["lag_1"] = last_vals.iloc[-1]
        user_input["lag_24"] = last_vals.iloc[-min(24, len(last_vals))]
        user_input["roll_mean_3"] = last_vals.iloc[-min(3, len(last_vals)):].mean()
        user_input["roll_mean_24"] = last_vals.iloc[-min(24, len(last_vals)):].mean()
        X_input = pd.DataFrame([user_input])[feature_cols]
        pred = gb_model.predict(X_input)[0]
        gb_pred.append(pred)
        last_vals = pd.concat([last_vals, pd.Series([pred])], ignore_index=True)

    gb_pred = np.array(gb_pred)

    # ============ Hybrid ============
    hybrid_pred, w_gb, w_sarima = hybrid_predict(gb_pred, sarima_forecast.values)

    # ============ Plot ============ 
    if granularity == "Daily":
        start_date = df.index[-1] + pd.Timedelta(days=1)
        freq = "D"
    elif granularity == "Weekly":
        start_date = df.index[-1] + pd.DateOffset(weeks=1)
        freq = "W"
    else:  # Monthly
        start_date = df.index[-1] + pd.DateOffset(months=1)
        freq = "M"

    future_idx = pd.date_range(
        start=start_date,
        periods=n_periods,
        freq=freq
    )

    forecast_df = pd.DataFrame({
        "Hybrid Forecast": np.round(hybrid_pred).astype(int)
    }, index=future_idx)

    st.subheader(f"Forecast Results ({granularity})")
    st.write(forecast_df)

    st.subheader("Forecast Visualization")
    plt.figure(figsize=(10,5))
    plt.plot(forecast_df.index, forecast_df["Hybrid Forecast"], color="green", label="Hybrid Forecast")
    plt.legend()
    st.pyplot(plt)

    st.metric("Prediksi Total", int(forecast_df["Hybrid Forecast"].sum()))
