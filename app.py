"""
app.py
======

Aplicación Streamlit para el procesamiento, modelado y visualización de
datos de **Sísmica de Refracción (TRS)** y **MASW**.

    Tab 1  Integración 1D/2D y módulos elásticos
    Tab 2  Topografía y predicción espacial de Vs
    Tab 3  Visualización 3D y mapas base

Ejecución:

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

st.set_page_config(
    page_title="TRS + MASW — Integración, modelado y visualización",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# La importación va después de `set_page_config` (que debe ser el primer
# comando de Streamlit) para poder mostrar un diagnóstico legible cuando el
# entorno tiene binarios incompatibles en vez de un traceback crudo.
try:
    from tabs import tab1_integracion, tab2_topografia, tab3_visualizacion  # noqa: E402
except ImportError as _exc:  # entorno con binarios incompatibles
    _mensaje = str(_exc)
    _abi = any(
        s in _mensaje.lower()
        for s in (
            "numpy.core.multiarray failed to import",
            "numpy.dtype size changed",
            "binary incompatibility",
            "_array_api not found",
        )
    )
    st.error(f"No se pudieron cargar las dependencias: {_mensaje}")
    if _abi:
        st.markdown(
            "**Conflicto de binarios con NumPy.** Algún paquete compilado "
            "(matplotlib, scipy, scikit-learn…) fue construido contra otra versión "
            "mayor de NumPy que la instalada. Ocurre al instalar con `pip` sobre un "
            "entorno conda que ya traía esos paquetes.\n\n"
            "Ejecute `python check_entorno.py` para el diagnóstico completo, o "
            "cree un entorno limpio:\n"
            "```\n"
            "conda create -n trs python=3.11 -y\n"
            "conda activate trs\n"
            "pip install -r requirements.txt\n"
            "```"
        )
    st.stop()

ESTILO = """
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
  [data-testid="stMetricValue"] { font-size: 1.35rem; }
  [data-testid="stMetricLabel"] { font-size: 0.78rem; color: #6b7280; }
  h3 { margin-top: 0.2rem; }
</style>
"""
st.markdown(ESTILO, unsafe_allow_html=True)


def _inicializar_estado() -> None:
    for clave, valor in {
        "lineas": {},
        "asignadas": {},
        "perfiles_2d": {},
        "importados_3d": {},
    }.items():
        st.session_state.setdefault(clave, valor)


def _barra_lateral() -> None:
    with st.sidebar:
        st.markdown("### Estado de la sesión")
        st.metric("Líneas procesadas (Tab 1)", len(st.session_state.get("lineas", {})))
        st.metric("Líneas georreferenciadas", len(st.session_state.get("asignadas", {})))
        st.metric("Perfiles 2D generados", len(st.session_state.get("perfiles_2d", {})))

        if st.session_state.get("lineas"):
            st.caption("Registradas: " + ", ".join(st.session_state["lineas"].keys()))

        st.divider()
        if st.button("Reiniciar sesión", use_container_width=True):
            for clave in ["lineas", "asignadas", "perfiles_2d", "importados_3d",
                          "entrenamiento", "topografia"]:
                st.session_state.pop(clave, None)
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption(
            "**Fuentes**  \n"
            "· Módulos elásticos y clasificación: `Interpretacion_MASW_TRS.ipynb`  \n"
            "· Kriging 3D y fusión con topografía: `crear_linea_sintetica.ipynb`  \n"
            "· Cortes 2D: `plot_2d_puentes.ipynb`  \n"
            "· Dashboard 3D: `plot_3d_mejorado.py` / `Graficos_3D_Vp_Vs.ipynb`"
        )


def main() -> None:
    _inicializar_estado()

    st.title("Sísmica de Refracción + MASW")
    st.caption(
        "Integración de modelos de velocidad, parámetros elásticos, predicción "
        "espacial de Vs y visualización 3D."
    )

    _barra_lateral()

    t1, t2, t3 = st.tabs(
        [
            "1 · Integración y módulos elásticos",
            "2 · Topografía y predicción de Vs",
            "3 · Visualización 3D",
        ]
    )
    with t1:
        tab1_integracion.render()
    with t2:
        tab2_topografia.render()
    with t3:
        tab3_visualizacion.render()


if __name__ == "__main__":
    main()
