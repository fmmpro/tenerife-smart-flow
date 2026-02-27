# backend/app/main.py
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

APP_NAME = "tenerife-smart-flow-api"
SERVICE_HEALTH_NAME = "tenerife-smart-flow-backend"
ENV = os.getenv("ENV", "dev")
ZONE = os.getenv("TSF_ZONE", "teno")

# Repo root = .../backend/app/main.py -> parents[2] => repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"

app = FastAPI(title=APP_NAME)

# CORS (por si luego consumimos desde Streamlit / FlutterFlow / WebView)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # para demo; si quieres lo cerramos luego
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp_non_negative(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evita valores negativos en predicciones (no tiene sentido vehículos < 0)."""
    out: List[Dict[str, Any]] = []
    for row in preds:
        r = dict(row)
        try:
            v = float(r.get("vehiculos_pred", 0.0))
        except Exception:
            v = 0.0
        r["vehiculos_pred"] = max(0.0, v)
        out.append(r)
    return out


def forecast_file_path(hours: int) -> Path:
    return DOCS_DIR / f"forecast_vehiculos_{hours}h.json"


def load_forecast_payload(path: Path) -> Union[Dict[str, Any], List[Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": APP_NAME,
        "status": "ok",
        "env": ENV,
        "zone": ZONE,
        "endpoints": {
            "health": "/health",
            "forecast_24h": "/forecast?hours=24",
            "forecast_72h": "/forecast?hours=72",
        },
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": SERVICE_HEALTH_NAME, "env": ENV}


@app.get("/favicon.ico")
def favicon() -> Response:
    # Evita 404s en logs del navegador
    return Response(status_code=204)


@app.get("/forecast")
def forecast(
    hours: int = Query(24, ge=1, le=168, description="Horizonte en horas (24 o 72 recomendado)"),
) -> Dict[str, Any]:
    if hours not in (24, 72):
        raise HTTPException(status_code=400, detail="hours must be 24 or 72")

    path = forecast_file_path(hours)
    print(f"[forecast] reading file: {path}")  # útil para logs en Render

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Forecast file not found: {path}")

    payload = load_forecast_payload(path)

    # ✅ Soporta ambos formatos: dict (nuevo) o list (antiguo)
    if isinstance(payload, list):
        preds = payload
        # si la lista ya viene como [{"timestamp":..., "vehiculos_pred":...}, ...]
        # respondemos en formato “normalizado”
        preds_norm = clamp_non_negative([p for p in preds if isinstance(p, dict)])
        return {
            "zone": ZONE,
            "generated_at": utc_now_iso(),
            "hours": hours,
            "predictions": preds_norm,
            "source_file": str(path.relative_to(REPO_ROOT)) if path.is_absolute() else str(path),
            "note": "payload was a list; wrapped into dict response",
        }

    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Invalid forecast JSON format (not dict/list)")

    preds = payload.get("predictions", [])
    if isinstance(preds, list):
        preds = clamp_non_negative([p for p in preds if isinstance(p, dict)])
    else:
        preds = []

    # Normalizamos algunas claves mínimas si faltan
    payload.setdefault("zone", ZONE)
    payload.setdefault("hours", hours)
    payload.setdefault("generated_at", utc_now_iso())
    payload["predictions"] = preds
    payload["source_file"] = (
        str(path.relative_to(REPO_ROOT)) if path.is_absolute() else str(path)
    )
    return payload
