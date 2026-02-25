import os
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------
# Config
# -----------------------------
DEFAULT_TENO_URL = (
    "https://datos.tenerife.es/ckan/dataset/abaae767-dca5-4ed7-a405-5410f50e7e81/"
    "resource/634d4743-065c-49f6-9f93-495645e92db5/download/"
    "afluencia-de-acceso-en-vehiculos-a-punta-de-teno-por-horas-del-ano-2026..json"
)
DATA_URL = os.getenv("TENO_DATA_URL", DEFAULT_TENO_URL)

st.set_page_config(page_title="Tenerife Smart Flow", layout="wide")
st.title("Tenerife Smart Flow")
st.caption("Demo (tiempo real): descarga → limpieza → entrenamiento → forecast (Punta de Teno)")


# -----------------------------
# Helpers (cacheados)
# -----------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)  # 1h
def download_json(url: str) -> dict:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def clean_to_df(raw: dict) -> pd.DataFrame:
    """
    Estructura esperada:
      {"fecha_y_hora":[{"fecha_hora":"...","afluencia":[{"vehiculos":x,"personas":y}]}]}
    """
    rows = []
    for item in raw.get("fecha_y_hora", []):
        ts = item.get("fecha_hora")
        a = item.get("afluencia", [])
        if isinstance(a, list) and len(a) > 0 and isinstance(a[0], dict):
            veh = a[0].get("vehiculos", 0)
            per = a[0].get("personas", 0)
        else:
            veh, per = 0, 0
        rows.append((ts, "teno", veh, per))

    df = pd.DataFrame(rows, columns=["timestamp", "zone", "vehiculos", "personas"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Cast numérico
    df["vehiculos"] = pd.to_numeric(df["vehiculos"], errors="coerce").fillna(0).astype(int)
    df["personas"] = pd.to_numeric(df["personas"], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data(ttl=60 * 60, show_spinner=False)
def make_hourly_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex horario continuo para lags reales.
    Rellena huecos con 0 (en esta primera versión).
    """
    d = df[["timestamp", "vehiculos"]].copy()
    d = d.set_index("timestamp").sort_index()

    full_index = pd.date_range(d.index.min(), d.index.max(), freq="H")
    d = d.reindex(full_index)

    d["vehiculos"] = d["vehiculos"].fillna(0)
    d = d.rename_axis("timestamp").reset_index()
    d["zone"] = "teno"
    return d


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["hour"] = d["timestamp"].dt.hour.astype(int)
    d["weekday"] = d["timestamp"].dt.weekday.astype(int)
    d["is_weekend"] = (d["weekday"] >= 5).astype(int)

    # Lags
    d["lag1"] = d["vehiculos"].shift(1)
    d["lag24"] = d["vehiculos"].shift(24)
    d["lag168"] = d["vehiculos"].shift(168)

    # Rolling (usar shift para evitar leakage)
    d["roll24_mean"] = d["vehiculos"].shift(1).rolling(24).mean()
    d["roll168_mean"] = d["vehiculos"].shift(1).rolling(168).mean()

    # Relleno
    d = d.dropna().reset_index(drop=True)
    return d


@st.cache_resource(show_spinner=False)
def train_model(train_df: pd.DataFrame):
    X_cols = ["hour", "weekday", "is_weekend", "lag1", "lag24", "lag168", "roll24_mean", "roll168_mean"]
    X = train_df[X_cols]
    y = train_df["vehiculos"]

    # Split temporal (80/20)
    split = int(len(train_df) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

    return model, X_cols, (mae, rmse), (X_test, y_test, preds), split


def forecast_autoregressive(hourly_full: pd.DataFrame, model, X_cols: list, hours: int) -> pd.DataFrame:
    """
    Forecast iterativo: añade predicciones al final y recalcula features.
    """
    base = hourly_full[["timestamp", "vehiculos"]].copy().sort_values("timestamp")
    last_ts = base["timestamp"].max()

    # Extend timeline
    future_index = pd.date_range(last_ts + timedelta(hours=1), last_ts + timedelta(hours=hours), freq="H")
    future = pd.DataFrame({"timestamp": future_index, "vehiculos": np.nan})
    work = pd.concat([base, future], ignore_index=True)

    # Relleno inicial de nans con 0 para poder construir lags al principio del forecast
    # (durante el forecast, iremos escribiendo predicciones reales)
    for i in range(len(work)):
        if pd.isna(work.loc[i, "vehiculos"]):
            # Construir features sobre una ventana que incluya lo ya predicho
            temp = work.copy()
            temp["zone"] = "teno"
            temp = add_features(temp.rename(columns={"vehiculos": "vehiculos"}))
            # temp ya no tiene filas NaN, así que buscamos la fila correspondiente al timestamp actual
            ts = work.loc[i, "timestamp"]
            row = temp[temp["timestamp"] == ts]
            if row.empty:
                # si todavía no hay suficientes lags (al inicio), poner 0
                y_pred = 0.0
            else:
                X_row = row[X_cols].astype(float)
                y_pred = float(model.predict(X_row)[0])
            work.loc[i, "vehiculos"] = max(0.0, y_pred)

    out = work[work["timestamp"].isin(future_index)].copy()
    out = out.rename(columns={"vehiculos": "vehiculos_pred"})
    return out


def plot_real_vs_pred(train_df, split, y_test, preds):
    fig, ax = plt.subplots(figsize=(12, 4))
    ts_test = train_df["timestamp"].iloc[split:].reset_index(drop=True)
    ax.plot(ts_test, y_test.values, label="Real")
    ax.plot(ts_test, preds, label="Predicción", linestyle="--")
    ax.set_title("Vehículos — Real vs Predicción (test)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Vehículos")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_forecast(hourly_full: pd.DataFrame, forecast_df: pd.DataFrame, days_back: int = 7):
    fig, ax = plt.subplots(figsize=(12, 4))
    cutoff = hourly_full["timestamp"].max() - timedelta(days=days_back)
    recent = hourly_full[hourly_full["timestamp"] >= cutoff]

    ax.plot(recent["timestamp"], recent["vehiculos"], label="Real (últimos días)")
    ax.axvline(hourly_full["timestamp"].max(), linestyle=":", label="Inicio forecast")
    ax.plot(forecast_df["timestamp"], forecast_df["vehiculos_pred"], linestyle="--", label="Forecast")

    ax.set_title(f"Forecast vehículos — {len(forecast_df)}h")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Vehículos")
    ax.legend()
    fig.tight_layout()
    return fig


# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.subheader("Configuración")
    st.write("Fuente de datos:")
    st.code(DATA_URL)
    hours = st.selectbox("Horizonte forecast", [24, 72], index=1)
    st.caption("Tip: puedes forzar recálculo con el botón.")
    recompute = st.button("🔁 Recalcular ahora (limpiar caché)")
    if recompute:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# -----------------------------
# Run pipeline
# -----------------------------
with st.spinner("Descargando datos..."):
    raw = download_json(DATA_URL)

with st.spinner("Limpiando y construyendo dataset..."):
    df_raw = clean_to_df(raw)
    hourly_full = make_hourly_full(df_raw)
    train_df = add_features(hourly_full)

with st.spinner("Entrenando modelo XGBoost..."):
    model, X_cols, (mae, rmse), (X_test, y_test, preds), split = train_model(train_df)

c1, c2, c3 = st.columns(3)
c1.metric("Filas (raw)", f"{len(df_raw):,}".replace(",", "."))
c2.metric("Filas (hourly full)", f"{len(hourly_full):,}".replace(",", "."))
c3.metric("Filas (train)", f"{len(train_df):,}".replace(",", "."))

m1, m2 = st.columns(2)
m1.metric("MAE (test)", f"{mae:.3f}")
m2.metric("RMSE (test)", f"{rmse:.3f}")

colA, colB = st.columns(2)

with colA:
    st.subheader("Real vs Predicción")
    fig1 = plot_real_vs_pred(train_df, split, y_test, preds)
    st.pyplot(fig1, clear_figure=True)

with colB:
    st.subheader(f"Forecast {hours}h + descarga JSON")
    with st.spinner("Generando forecast..."):
        fc = forecast_autoregressive(hourly_full, model, X_cols, hours=hours)

    fig2 = plot_forecast(hourly_full, fc, days_back=7)
    st.pyplot(fig2, clear_figure=True)

    st.dataframe(fc, use_container_width=True, height=360)

    payload = fc.copy()
    payload["timestamp"] = payload["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    json_bytes = payload.to_json(orient="records", force_ascii=False).encode("utf-8")
    st.download_button("⬇️ Descargar forecast JSON", data=json_bytes, file_name=f"forecast_vehiculos_{hours}h.json", mime="application/json")
