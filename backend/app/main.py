from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
from datetime import datetime, timezone

# ===============================
# APP CONFIG
# ===============================

app = FastAPI(
    title="Tenerife Smart Flow API",
    version="1.1.0",
    description="API para servir el último forecast generado de afluencia vehicular en Punta de Teno."
)

# Permitir acceso desde Flutter / web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción puedes restringir dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorios
ROOT = Path(__file__).resolve().parents[2]  # .../tenerife-smart-flow
DOCS_DIR = ROOT / "docs"


# ===============================
# HEALTH CHECK
# ===============================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "tenerife-smart-flow-backend",
        "env": "dev"
    }


# ===============================
# FORECAST ENDPOINT
# ===============================

@app.get("/forecast")
def forecast(hours: int = Query(72, ge=1, le=168)):
    """
    Devuelve el último forecast generado por el pipeline:
      docs/forecast_vehiculos_24h.json
      docs/forecast_vehiculos_72h.json
    """

    # Validar horizonte permitido
    if hours not in (24, 72):
        raise HTTPException(
            status_code=400,
            detail="hours debe ser 24 o 72"
        )

    fpath = DOCS_DIR / f"forecast_vehiculos_{hours}h.json"

    if not fpath.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No existe {fpath}. Genera primero: "
                   f"python pipeline/training/05_forecast_future.py --hours {hours}"
        )

    try:
        data = json.loads(fpath.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error leyendo JSON: {e}"
        )

    # ===============================
    # NORMALIZACIÓN: evitar negativos
    # ===============================
    for row in data:
        if "vehiculos_pred" in row and row["vehiculos_pred"] is not None:
            row["vehiculos_pred"] = max(0.0, float(row["vehiculos_pred"]))

    return {
        "zone": "teno",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hours": hours,
        "predictions": data,
        "source_file": str(fpath.relative_to(ROOT)),
    }
