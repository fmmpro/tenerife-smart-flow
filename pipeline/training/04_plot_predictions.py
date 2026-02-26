import glob
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from pathlib import Path

# Carpeta donde guardaremos las figuras (para memoria del concurso)
OUT_DIR = Path("docs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Cargar el parquet horario completo
    parquet_file = sorted(
        glob.glob("pipeline/data/processed/*_hourly_full.parquet")
    )[-1]

    df = pd.read_parquet(parquet_file).sort_values("timestamp")

    # Features de calendario
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Lags reales (horas)
    df["lag1"] = df["vehiculos"].shift(1)
    df["lag24"] = df["vehiculos"].shift(24)
    df["lag168"] = df["vehiculos"].shift(168)

    # Rolling reales
    df["roll24_mean"] = df["vehiculos"].rolling(24).mean()
    df["roll168_mean"] = df["vehiculos"].rolling(168).mean()

    df = df.dropna().copy()

    # Split temporal (80/20)
    cut = int(len(df) * 0.8)
    train = df.iloc[:cut]
    test = df.iloc[cut:].copy()

    X_cols = [
        "hour", "weekday", "is_weekend",
        "lag1", "lag24", "lag168",
        "roll24_mean", "roll168_mean"
    ]

    X_train = train[X_cols]
    y_train = train["vehiculos"]
    X_test = test[X_cols]
    y_test = test["vehiculos"]

    model = XGBRegressor(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    model.fit(X_train, y_train)
    test["pred"] = model.predict(X_test)

    # Plot
    plt.figure(figsize=(14, 5))
    plt.plot(test["timestamp"], y_test.values, label="Real", linewidth=2)
    plt.plot(test["timestamp"], test["pred"].values, label="Predicción", linestyle="--")
    plt.legend()
    plt.title("Tenerife Smart Flow — Vehículos (real vs predicción)")
    plt.xlabel("Fecha")
    plt.ylabel("Número de vehículos")
    plt.tight_layout()

    out_path = OUT_DIR / "vehiculos_real_vs_pred.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[OK] Gráfico guardado en {out_path}")

if __name__ == "__main__":
    main()
