"""
core/servicios.py
=================

Capa de servicios cacheados para Streamlit.

Todas las operaciones costosas del núcleo se exponen aquí envueltas en
``@st.cache_data``.  Las funciones reciben **bytes** o DataFrames (tipos que
Streamlit sabe hashear) y nunca objetos de interfaz, de modo que el núcleo
(:mod:`core.fisica`, :mod:`core.ml`, :mod:`core.prediccion`…) permanece
independiente de Streamlit y se puede importar desde un notebook o un
script por lotes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import espacial, fisica, io_utils, ml, plots_2d, plots_3d, prediccion

TTL = 3600  # 1 h


# ---------------------------------------------------------------------------
# Ingesta
# ---------------------------------------------------------------------------
@st.cache_data(ttl=TTL, show_spinner=False)
def cargar_trs_2d(datos: bytes, unidad_vp: str = "auto") -> pd.DataFrame:
    return io_utils.cargar_trs_2d(datos, unidad_vp=unidad_vp)


@st.cache_data(ttl=TTL, show_spinner=False)
def cargar_masw_1d(datos: bytes) -> pd.DataFrame:
    return io_utils.cargar_masw_1d(datos)


@st.cache_data(ttl=TTL, show_spinner=False)
def cargar_topografia(datos: bytes) -> pd.DataFrame:
    return io_utils.cargar_topografia(datos)


@st.cache_data(ttl=TTL, show_spinner=False)
def cargar_modelo_2d(datos: bytes) -> pd.DataFrame:
    return io_utils.cargar_modelo_2d(datos)


# ---------------------------------------------------------------------------
# Tab 1 — integración física
# ---------------------------------------------------------------------------
@st.cache_data(ttl=TTL, show_spinner=False)
def integrar_trs_masw(
    df_trs: pd.DataFrame,
    df_masw: pd.DataFrame,
    potencia_idw: float,
    radio: float | None,
    n_nodos: int,
    prof_max: float | None,
    modo_correccion: str,
    densidad_fija: float | None,
    extrapolar: bool,
    suavizado: float | None = None,
):
    return fisica.integrar_trs_masw(
        df_trs,
        df_masw,
        potencia_idw=potencia_idw,
        radio=radio,
        n_nodos=n_nodos,
        prof_max=prof_max,
        modo_correccion=modo_correccion,
        densidad_fija=densidad_fija,
        extrapolar=extrapolar,
        suavizado=suavizado,
    )


@st.cache_data(ttl=TTL, show_spinner=False)
def clasificar(
    df: pd.DataFrame,
    max_clases: int,
    varianza_pca: float,
    replicar_notebook: bool,
):
    return ml.clasificacion_geofisica_optimizada(
        df,
        max_clases=max_clases,
        varianza_pca=varianza_pca,
        replicar_notebook=replicar_notebook,
    )


# ---------------------------------------------------------------------------
# Tab 2 — espacial y predicción
# ---------------------------------------------------------------------------
@st.cache_data(ttl=TTL, show_spinner=False)
def combinar_trs_con_topografia(
    df_trs: pd.DataFrame,
    topo_linea: pd.DataFrame,
    nombre_linea: str,
    modo: str,
    tolerancia: float,
):
    return espacial.combinar_trs_con_topografia(
        df_trs, topo_linea, nombre_linea, modo=modo, tolerancia=tolerancia
    )


@st.cache_data(ttl=TTL, show_spinner=False)
def proyectar_vs(
    df_entrenamiento: pd.DataFrame,
    df_malla: pd.DataFrame,
    model_type: str,
    spatial_cv_strategy: str,
    n_blocks: int,
    n_trials: int,
    apply_residual_kriging: bool,
    variogram_model: str,
    clip_predictions: bool,
    clip_margin_pct: float,
    random_state: int,
    auto_drop_degenerate: bool = True,
    leverage_max: float = 5.0,
    use_engineered_features: bool = True,
):
    """
    Ejecuta el pipeline completo de proyección de Vs.

    Se reciben parámetros primitivos en vez de un ``PipelineConfig`` para que
    la clave de caché sea estable y legible.
    """
    config = prediccion.PipelineConfig(
        model_type=model_type,
        spatial_cv_strategy=spatial_cv_strategy,
        n_blocks=n_blocks,
        n_trials=n_trials,
        apply_residual_kriging=apply_residual_kriging,
        variogram_model=variogram_model,
        clip_predictions=clip_predictions,
        clip_margin_pct=clip_margin_pct,
        random_state=random_state,
        auto_drop_degenerate=auto_drop_degenerate,
        leverage_max=leverage_max,
        use_engineered_features=use_engineered_features,
    )
    return prediccion.run_full_pipeline(df_entrenamiento, df_malla, config)


@st.cache_data(ttl=TTL, show_spinner=False)
def aplicar_a_nueva_malla(
    _resultado: dict,
    firma: str,
    df_malla: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aplica un pipeline ya entrenado a otra malla 2D.

    ``_resultado`` lleva guion bajo para que Streamlit no intente hashear el
    estimador de scikit-learn; ``firma`` es la clave que sí participa en el
    hash de la caché.
    """
    return prediccion.aplicar_a_nueva_malla(_resultado, df_malla)


@st.cache_data(ttl=TTL, show_spinner=False)
def sintetizar_linea_kriging(
    df_source: pd.DataFrame,
    topo_objetivo: pd.DataFrame,
    variable: str,
    modelo_variograma: str,
    anisotropy_scaling_z: float,
    rango_variograma: float,
    prof_max: float,
    dz: float,
):
    return prediccion.sintetizar_linea_kriging(
        df_source,
        topo_objetivo,
        variable=variable,
        modelo_variograma=modelo_variograma,
        anisotropy_scaling_z=anisotropy_scaling_z,
        rango_variograma=rango_variograma,
        prof_max=prof_max,
        dz=dz,
    )


# ---------------------------------------------------------------------------
# Tab 3 — gráficos
# ---------------------------------------------------------------------------
@st.cache_data(ttl=TTL, show_spinner=False)
def corte_2d_png(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    col_z: str,
    metodo: str,
    vmin: float | None,
    vmax: float | None,
    profundidad_max: float,
    cmap: str,
    contornos: tuple,
    resolucion: int,
    titulo: str,
) -> bytes:
    _fig, png = plots_2d.exportar_slide_2d_recortado(
        df,
        col_x=col_x,
        col_y=col_y,
        col_z=col_z,
        metodo=metodo,
        vmin=vmin,
        vmax=vmax,
        profundidad_max=profundidad_max,
        cmap=cmap,
        contornos=list(contornos) if contornos else None,
        resolucion=resolucion,
        titulo=titulo,
    )
    import matplotlib.pyplot as plt

    plt.close("all")
    return png


@st.cache_data(ttl=TTL, show_spinner=False)
def extension_geotiff(datos: bytes) -> dict:
    return plots_3d.leer_extension_geotiff(datos)


@st.cache_data(ttl=TTL, show_spinner=False)
def dataframe_a_csv(df: pd.DataFrame, index: bool = False) -> bytes:
    return df.to_csv(index=index).encode("utf-8")
