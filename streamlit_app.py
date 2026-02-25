import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tenerife Smart Flow", layout="wide")

st.title("Tenerife Smart Flow")
st.caption("Demo: predicción de afluencia vehicular (Punta de Teno) con datos abiertos del Cabildo de Tenerife.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Visualizaciones")
    fig_dir = Path("docs/figures")
    imgs = [
        ("Real vs Predicción", fig_dir / "vehiculos_real_vs_pred.png"),
        ("Importancia de variables", fig_dir / "feature_importance_gain.png"),
        ("Forecast 72h", fig_dir / "forecast_vehiculos_72h.png"),
    ]
    for title, p in imgs:
        st.markdown(f"**{title}**")
        if p.exists():
            st.image(str(p), use_container_width=True)
        else:
            st.warning(f"No encontrado: {p}")

with col2:
    st.subheader("Forecast (tabla + descarga)")
    hours = st.selectbox("Horizonte", [24, 72], index=1)
    f = Path("docs") / f"forecast_vehiculos_{hours}h.json"
    if f.exists():
        data = json.loads(f.read_text(encoding="utf-8"))
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        st.dataframe(df, use_container_width=True, height=520)
        st.download_button(
            "Descargar forecast JSON",
            data=f.read_bytes(),
            file_name=f.name,
            mime="application/json",
        )
    else:
        st.error(
            f"No existe {f}. Genera primero el forecast con:\n"
            f"python pipeline/training/05_forecast_future.py --hours {hours}"
        )
