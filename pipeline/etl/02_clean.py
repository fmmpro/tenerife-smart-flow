import json
from pathlib import Path
from datetime import datetime
import pandas as pd

RAW_DIR = Path("pipeline/data/raw")
OUT_DIR = Path("pipeline/data/processed")
ZONE = "teno"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("afluencia_teno_*.json"))
    if not files:
        raise FileNotFoundError("No hay JSON en pipeline/data/raw. Ejecuta 01_download.py primero.")

    latest = files[-1]
    print(f"[INFO] Usando raw: {latest}")

    obj = json.loads(latest.read_text(encoding="utf-8"))
    data = obj.get("fecha_y_hora")

    if not isinstance(data, list):
        raise ValueError("Formato inesperado: 'fecha_y_hora' no es una lista.")

    rows = []
    for item in data:
        ts = item.get("fecha_hora")
        af = item.get("afluencia", [])

        # afluencia es lista de dicts: [{'vehiculos': X, 'personas': Y}, ...]
        vehiculos = 0
        personas = 0
        if isinstance(af, list):
            for d in af:
                if isinstance(d, dict):
                    vehiculos += int(d.get("vehiculos", 0) or 0)
                    personas += int(d.get("personas", 0) or 0)

        rows.append({
            "timestamp": ts,
            "zone": ZONE,
            "vehiculos": vehiculos,
            "personas": personas
        })

    df = pd.DataFrame(rows)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    # asegurar tipos
    df["vehiculos"] = pd.to_numeric(df["vehiculos"], errors="coerce").fillna(0).astype("int64")
    df["personas"] = pd.to_numeric(df["personas"], errors="coerce").fillna(0).astype("int64")

    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Tras limpiar, dataframe vacío. Revisa el contenido del JSON.")

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"afluencia_{ZONE}_hourly_{stamp}.parquet"
    df.to_parquet(out_path, index=False)

    print(f"[OK] Procesado -> {out_path}")
    print(f"[OK] Filas: {len(df)} | Desde: {df['timestamp'].min()} | Hasta: {df['timestamp'].max()}")
    print("[OK] Stats vehiculos: min={}, max={}, mean={:.2f}".format(
        df["vehiculos"].min(), df["vehiculos"].max(), df["vehiculos"].mean()
    ))
    print("[OK] Stats personas: min={}, max={}, mean={:.2f}".format(
        df["personas"].min(), df["personas"].max(), df["personas"].mean()
    ))

if __name__ == "__main__":
    main()
