![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-ready-brightgreen)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)


# Tenerife Smart Flow

Sistema predictivo de gestión inteligente de aforos y movilidad insular usando IA y datos abiertos del Cabildo de Tenerife.

## Estructura
- pipeline/: ETL, notebooks y entrenamiento de modelos
- backend/: API FastAPI
- mobile/: App Flutter
- docs/: memoria técnica y diagramas

## Licencia
EUPL 1.2

## Estado del proyecto
- ✅ ETL + Parquet
- ✅ Entrenamiento + interpretabilidad
- ✅ Forecast + gráficos
- 🔜 API /predict (producción)
- 🔜 App híbrida (Flutter)

## Arquitectura (alto nivel)

```mermaid
flowchart LR
  A[Datos Abiertos Cabildo<br/>datos.tenerife.es] --> B[ETL<br/>01_download / 02_clean]
  B --> C[Parquet<br/>processed/*_hourly_full.parquet]
  C --> D[Training<br/>XGBoost + features]
  D --> E[Forecast<br/>24-72h + JSON]
  D --> F[FastAPI Backend<br/>/health /predict]
  F --> G[App Flutter<br/>Android/iOS/Web]
  G --> H[Usuarios / Turistas / Gestores]
## Demo en vivo (Streamlit)
https://tenerife-smart-flow.streamlit.app/
