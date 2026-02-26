import glob
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

    df = df.dropna().copy()
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce").astype("float64")
    return df.dropna(subset=FEATURES).copy()

def main():
    # Cargar el parquet horario completo
    parquet_file = sorted(glob.glob("pipeline/data/processed/*_hourly_full.parquet"))[-1]
    df = pd.read_parquet(parquet_file).sort_values("timestamp")

    df_feat = make_features(df)

    # Entrenar (como en el modelo horario)
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

    booster = model.get_booster()

    # Importancia por gain (más informativa que "weight")
    score = booster.get_score(importance_type="gain")

    # Asegurar que todas las features aparecen
    gain = {f: float(score.get(f, 0.0)) for f in FEATURES}
    imp = pd.DataFrame({"feature": list(gain.keys()), "gain": list(gain.values())})
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)

    # Guardar tabla para memoria técnica
    out_csv = Path("docs") / "feature_importance_gain.csv"
    imp.to_csv(out_csv, index=False)

    # Plot
    plt.figure(figsize=(10,5))
    plt.barh(imp["feature"][::-1], imp["gain"][::-1])
    plt.title("Importancia de variables (XGBoost - gain)")
    plt.xlabel("Gain")
    plt.ylabel("Feature")
    plt.tight_layout()

    out_png = OUT_DIR / "feature_importance_gain.png"
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"[OK] CSV: {out_csv}")
    print(f"[OK] PNG: {out_png}")
    print("\nTop 5 features:")
    print(imp.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
