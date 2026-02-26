import glob
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    f = sorted(glob.glob("pipeline/data/processed/*_hourly_full.parquet"))[-1]
    df = pd.read_parquet(f).sort_values("timestamp")

    # Features calendario
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Lags reales (ahora sí son horas)
    df["lag1"] = df["vehiculos"].shift(1)
    df["lag24"] = df["vehiculos"].shift(24)
    df["lag168"] = df["vehiculos"].shift(168)  # 7 días

    # Rolling reales
    df["roll24_mean"] = df["vehiculos"].rolling(24).mean()
    df["roll168_mean"] = df["vehiculos"].rolling(168).mean()

    df = df.dropna().copy()

    # Split temporal
    cut = int(len(df) * 0.8)
    train, test = df.iloc[:cut], df.iloc[cut:]

    X_cols = ["hour", "weekday", "is_weekend", "lag1", "lag24", "lag168", "roll24_mean", "roll168_mean"]
    X_train, y_train = train[X_cols], train["vehiculos"]
    X_test, y_test = test[X_cols], test["vehiculos"]

    model = XGBRegressor(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print("[OK] MODELO HORARIO (reindex + lags reales)")
    print("FILE:", f)
    print(f"MAE={mae:.3f} | RMSE={rmse:.3f}")
    print("test rows:", len(test))

if __name__ == "__main__":
    main()
