import glob
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

OUT_DIR = Path("docs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "hour", "weekday", "is_weekend",
    "lag1", "lag24", "lag168",
    "roll24_mean", "roll168_mean"
]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    df["hour"] = df["timestamp"].dt.hour.astype("int64")
    df["weekday"] = df["timestamp"].dt.weekday.astype("int64")
    df["is_weekend"] = (df["weekday"] >= 5).astype("int64")

    df["vehiculos"] = pd.to_numeric(df["vehiculos"], errors="coerce")

    df["lag1"] = df["vehiculos"].shift(1)
    df["lag24"] = df["vehiculos"].shift(24)
    df["lag168"] = df["vehiculos"].shift(168)

    df["roll24_mean"] = df["vehiculos"].rolling(24).mean()
    df["roll168_mean"] = df["vehiculos"].rolling(168).mean()
    return df

def train_model(df_feat: pd.DataFrame) -> XGBRegressor:
    df_feat = df_feat.dropna().copy()
    df_feat[FEATURES] = df_feat[FEATURES].apply(pd.to_numeric, errors="coerce").astype("float64")
    df_feat = df_feat.dropna(subset=FEATURES)

    cut = int(len(df_feat) * 0.8)
    train = df_feat.iloc[:cut]

    X_train = train[FEATURES]
    y_train = train["vehiculos"].astype("float64")

    model = XGBRegressor(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def main(hours: int = 72):
    parquet_file = sorted(glob.glob("pipeline/data/processed/*_hourly_full.parquet"))[-1]
    df = pd.read_parquet(parquet_file).sort_values("timestamp").copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["vehiculos"] = pd.to_numeric(df["vehiculos"], errors="coerce").fillna(0).astype("float64")

    # Entrenar
    df_feat = make_features(df)
    model = train_model(df_feat)

    # Forecast autoregresivo
    hist = df[["timestamp", "vehiculos"]].copy()
    last_ts = hist["timestamp"].max()

    future_rows = []
    for h in range(1, hours + 1):
        ts = last_ts + pd.Timedelta(hours=h)

        tmp = pd.concat(
            [hist, pd.DataFrame([{"timestamp": ts, "vehiculos": 0.0}])],
            ignore_index=True
        )

        tmp_feat = make_features(tmp).dropna()
        row = tmp_feat.iloc[-1]

        X_row = row[FEATURES].to_frame().T
        X_row = X_row.apply(pd.to_numeric, errors="coerce").astype("float64")

        y_pred = float(model.predict(X_row)[0])

        future_rows.append({"timestamp": ts.isoformat(), "vehiculos_pred": y_pred})
        hist = pd.concat(
            [hist, pd.DataFrame([{"timestamp": ts, "vehiculos": y_pred}])],
            ignore_index=True
        )

    # Guardar JSON (útil para API/memoria)
    out_json = Path("docs") / f"forecast_vehiculos_{hours}h.json"
    out_json.write_text(json.dumps(future_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # Plot: últimos 7 días reales + forecast
    df_plot = df.sort_values("timestamp").copy()
    tail_start = df_plot["timestamp"].max() - pd.Timedelta(days=7)
    df_tail = df_plot[df_plot["timestamp"] >= tail_start][["timestamp", "vehiculos"]].copy()

    df_future = pd.DataFrame(future_rows)
    df_future["timestamp"] = pd.to_datetime(df_future["timestamp"])
    df_future["vehiculos"] = pd.to_numeric(df_future["vehiculos_pred"], errors="coerce")

    plt.figure(figsize=(14,5))
    plt.plot(df_tail["timestamp"], df_tail["vehiculos"], label="Real (últimos 7 días)", linewidth=2)
    plt.plot(df_future["timestamp"], df_future["vehiculos"], label=f"Forecast {hours}h", linestyle="--")
    plt.axvline(df_plot["timestamp"].max(), linestyle=":", linewidth=2, label="Inicio forecast")
    plt.legend()
    plt.title(f"Tenerife Smart Flow — Forecast vehículos {hours}h")
    plt.xlabel("Fecha")
    plt.ylabel("Vehículos")
    plt.tight_layout()

    out_png = OUT_DIR / f"forecast_vehiculos_{hours}h.png"
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"[OK] JSON forecast: {out_json}")
    print(f"[OK] PNG forecast: {out_png}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=int, default=72)
    args = p.parse_args()
    main(hours=args.hours)
