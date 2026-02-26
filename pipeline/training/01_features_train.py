import glob
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

def load_latest_parquet():
    f = sorted(glob.glob("pipeline/data/processed/*.parquet"))[-1]
    return pd.read_parquet(f), f

def make_features(df):
    df = df.sort_values("timestamp").copy()

    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month

    # Lags basados en registros (baseline)
    df["lag1"] = df["vehiculos"].shift(1)
    df["lag2"] = df["vehiculos"].shift(2)
    df["lag7"] = df["vehiculos"].shift(7)

    df["roll_mean_7"] = df["vehiculos"].rolling(7).mean()

    return df.dropna().copy()

def temporal_split(df, test_ratio=0.2):
    n = len(df)
    cut = int(n * (1 - test_ratio))
    return df.iloc[:cut], df.iloc[cut:]

def main():
    df, f = load_latest_parquet()
    print("[INFO] File:", f)
    print("[INFO] Rows:", len(df))
    print("[INFO] Range:", df["timestamp"].min(), "->", df["timestamp"].max())

    df_feat = make_features(df)
    train, test = temporal_split(df_feat)

    X_cols = ["hour", "weekday", "day", "month", "lag1", "lag2", "lag7", "roll_mean_7"]
    X_train, y_train = train[X_cols], train["vehiculos"]
    X_test, y_test = test[X_cols], test["vehiculos"]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print(f"[OK] BASELINE MODELO")
    print(f"MAE={mae:.3f} | RMSE={rmse:.3f}")
    print("Test rows:", len(test))

if __name__ == "__main__":
    main()
