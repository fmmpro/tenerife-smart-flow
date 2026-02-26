import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

from datetime import timedelta
from collections import deque
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ===============================
# CONFIG
# ===============================

DEFAULT_TENO_URL = (
    "https://datos.tenerife.es/ckan/dataset/abaae767-dca5-4ed7-a405-5410f50e7e81/"
    "resource/634d4743-065c-49f6-9f93-495645e92db5/download/"
    "afluencia-de-acceso-en-vehiculos-a-punta-de-teno-por-horas-del-ano-2026..json"
)

DATA_URL = os.getenv("TENO_DATA_URL", DEFAULT_TENO_URL)

st.set_page_config(page_title="Tenerife Smart Flow", layout="wide")
st.title("Tenerife Smart Flow")
st.caption("Demo en tiempo real: descarga → limpieza → entrenamiento → forecast")


# ===============================
# DESCARGA Y LIMPIEZA
# ===============================

@st.cache_data(ttl=3600)
def download_json(url: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600)
def clean_to_df(raw: dict):
    rows = []
    for item in raw.get("fecha_y_hora", []):
        ts = item.get("fecha_hora")
        a = item.get("afluencia", [])
        if isinstance(a, list) and len(a) > 0:
            veh = a[0].get("vehiculos", 0)
        else:
            veh = 0
        rows.append((ts, veh))

    df = pd.DataFrame(rows, columns=["timestamp", "vehiculos"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    df["vehiculos"] = df["vehiculos"].astype(float)
    return df


# ===============================
# FEATURE ENGINEERING
# ===============================

def make_hourly_full(df):
    df = df.set_index("timestamp")
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="H")
    df = df.reindex(full_index)
    df["vehiculos"] = df["vehiculos"].fillna(0)
    df = df.rename_axis("timestamp").reset_index()
    return df


def add_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    df["lag1"] = df["vehiculos"].shift(1)
    df["lag24"] = df["vehiculos"].shift(24)
    df["lag168"] = df["vehiculos"].shift(168)

    df["roll24_mean"] = df["vehiculos"].shift(1).rolling(24).mean()
    df["roll168_mean"] = df["vehiculos"].shift(1).rolling(168).mean()

    df = df.dropna().reset_index(drop=True)
    return df


# ===============================
# ENTRENAMIENTO
# ===============================

@st.cache_resource
def train_model(train_df):
    X_cols = [
        "hour",
        "weekday",
        "is_weekend",
        "lag1",
        "lag24",
        "lag168",
        "roll24_mean",
        "roll168_mean",
    ]

    X = train_df[X_cols]
    y = train_df["vehiculos"]

    split = int(len(train_df) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))

    return model, X_cols, mae, rmse, split, X_test, y_test, preds


# ===============================
# FORECAST RÁPIDO
# ===============================

def forecast_fast(hourly_full, model, hours):
    base = hourly_full.copy()
    last_ts = base["timestamp"].max()

    history = deque(base["vehiculos"].tolist(), maxlen=5000)

    def mean_last(k):
        if len(history) < k:
            return 0.0
        return float(np.mean(list(history)[-k:]))

    preds = []

    for i in range(1, hours + 1):
        ts = last_ts + timedelta(hours=i)

        hour = ts.hour
        weekday = ts.weekday()
        is_weekend = 1 if weekday >= 5 else 0

        lag1 = history[-1] if len(history) >= 1 else 0
        lag24 = history[-24] if len(history) >= 24 else 0
        lag168 = history[-168] if len(history) >= 168 else 0

        roll24_mean = mean_last(24)
        roll168_mean = mean_last(168)

        X_row = pd.DataFrame([{
            "hour": hour,
            "weekday": weekday,
            "is_weekend": is_weekend,
            "lag1": lag1,
            "lag24": lag24,
            "lag168": lag168,
            "roll24_mean": roll24_mean,
            "roll168_mean": roll168_mean,
        }])

        y_pred = float(model.predict(X_row)[0])
        y_pred = max(0, y_pred)

        preds.append((ts, y_pred))
        history.append(y_pred)

    return pd.DataFrame(preds, columns=["timestamp", "vehiculos_pred"])


# ===============================
# PIPELINE COMPLETO
# ===============================

with st.spinner("Descargando datos..."):
    raw = download_json(DATA_URL)

df_raw = clean_to_df(raw)
hourly_full = make_hourly_full(df_raw)
train_df = add_features(hourly_full)

with st.spinner("Entrenando modelo XGBoost..."):
    model, X_cols, mae, rmse, split, X_test, y_test, preds = train_model(train_df)

st.success("Modelo entrenado correctamente")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.3f}")
c2.metric("RMSE", f"{rmse:.3f}")

# ===============================
# GRÁFICO TEST
# ===============================

st.subheader("Real vs Predicción (test)")

fig, ax = plt.subplots(figsize=(10, 4))
ts_test = train_df["timestamp"].iloc[split:]
ax.plot(ts_test, y_test.values, label="Real")
ax.plot(ts_test, preds, label="Predicción", linestyle="--")
ax.legend()
st.pyplot(fig)


# ===============================
# FORECAST
# ===============================

hours = st.selectbox("Horizonte forecast", [24, 72], index=1)

with st.spinner("Generando forecast..."):
    fc = forecast_fast(hourly_full, model, hours)

st.subheader(f"Forecast {hours} horas")
st.dataframe(fc, use_container_width=True)

fig2, ax2 = plt.subplots(figsize=(10, 4))
recent = hourly_full.tail(24 * 7)
ax2.plot(recent["timestamp"], recent["vehiculos"], label="Real")
ax2.axvline(hourly_full["timestamp"].max(), linestyle=":")
ax2.plot(fc["timestamp"], fc["vehiculos_pred"], linestyle="--", label="Forecast")
ax2.legend()
st.pyplot(fig2)
