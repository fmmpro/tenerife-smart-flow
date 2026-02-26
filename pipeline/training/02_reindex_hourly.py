import glob
import pandas as pd

def main():
    f = sorted(glob.glob("pipeline/data/processed/*.parquet"))[-1]
    df = pd.read_parquet(f)

    df = df.sort_values("timestamp").set_index("timestamp")

    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="H"
    )

    # Rellenamos horas sin registro con 0 vehículos/personas
    df = df.reindex(full_index)
    df["vehiculos"] = df["vehiculos"].fillna(0).astype("int64")
    df["personas"] = df["personas"].fillna(0).astype("int64")

    df["zone"] = "teno"
    df = df.reset_index().rename(columns={"index": "timestamp"})

    out = f.replace(".parquet", "_hourly_full.parquet")
    df.to_parquet(out, index=False)

    print("[OK] Serie horaria completa creada")
    print("FILE:", out)
    print("rows:", len(df))
    print("range:", df["timestamp"].min(), "->", df["timestamp"].max())

if __name__ == "__main__":
    main()
