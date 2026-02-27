# backend/app/main.py

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query


APP_NAME = "tenerife-smart-flow-backend"
ENV = os.getenv("ENV", "dev")

# Rutas (en Render el repo se clona en /opt/render/project/src)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORECAST_24H = REPO_ROOT / "docs" / "forecast_vehiculos_24h.json"
DEFAULT_FORECAST_72H = REPO_ROOT / "docs" / "forecast_vehiculos_72h.json"

app = FastAPI(title="Tenerife Smart Flow API", version="0.1.0")


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "tenerife-smart-flow-api",
        "status": "ok",
        "env": ENV,
        "endpoints": {
            "health": "/health",
            "forecast_24h": "/forecast?hours=24",
            "forecast_72h": "/forecast?hours=72",
        },
        "zone": "teno",
    }


@app.get("/favicon.ico")
def favicon() -> Dict[str, str]:
    # Evita 404 en navegadores
    return {"ok": "true"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": APP_NAME, "env": ENV}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Forecast file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_predictions(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Asegura:
    - vehiculos_pred >= 0
    - vehiculos_pred_int redondeado para UI móvil
    """
    out: List[Dict[str, Any]] = []
    for item in preds:
        ts = item.get("timestamp")
        pred = item.get("vehiculos_pred", 0.0)
        try:
            pred_f = float(pred)
        except Exception:
            pred_f = 0.0

        pred_f = max(0.0, pred_f)
        pred_int = int(round(pred_f))

        out.append(
            {
                "timestamp": ts,
                "vehiculos_pred": pred_f,
                "vehiculos_pred_int": pred_int,
            }
        )
    return out


@app.get("/forecast")
def forecast(hours: int = Query(24, ge=1, le=168)) -> Dict[str, Any]:
    """
    Devuelve forecast precalculado (JSON) para 24h o 72h.
    (Modo demo/estable para concurso y app móvil.)
    """
    if hours <= 24:
        path = DEFAULT_FORECAST_24H
    elif hours <= 72:
        path = DEFAULT_FORECAST_72H
    else:
        # Si piden más (hasta 168), por ahora devolvemos 72h con un aviso
        path = DEFAULT_FORECAST_72H

    try:
        payload = _read_json(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    preds = payload.get("predictions", [])
    preds_norm = _normalize_predictions(preds)

    return {
        "zone": payload.get("zone", "teno"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hours": hours,
        "predictions": preds_norm,
        "source_file": str(path.relative_to(REPO_ROOT)) if path.exists() else str(path),
        "note": "hours>72 returns 72h for now" if hours > 72 else None,
    }
