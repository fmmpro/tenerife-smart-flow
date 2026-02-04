import os
import requests
from datetime import datetime
from pathlib import Path

DATA_URL = os.getenv("TENO_DATA_URL", "")
OUT_DIR = Path("pipeline/data/raw")

def main():
    if not DATA_URL:
        raise ValueError("Falta TENO_DATA_URL (exporta la URL antes de ejecutar).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"afluencia_teno_{ts}.json"

    r = requests.get(DATA_URL, timeout=90)
    r.raise_for_status()

    out_path.write_bytes(r.content)

    print(f"[OK] Descargado: {out_path}")
    print(f"[OK] Status: {r.status_code} | Bytes: {out_path.stat().st_size} | Content-Type: {r.headers.get('content-type')}")

if __name__ == "__main__":
    main()
