from fastapi import FastAPI
import os

app = FastAPI(
    title="Tenerife Smart Flow API",
    version="0.1.0"
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "tenerife-smart-flow-backend",
        "env": os.getenv("APP_ENV", "dev")
    }
