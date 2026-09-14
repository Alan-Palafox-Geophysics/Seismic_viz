"""
tabs/tab1_integracion.py
========================

**Tab 1 — Integración 1D/2D y módulos elásticos.**

Flujo: carga del modelo 2D de Vp (TRS) y del 1D de Vs (MASW) → corrección
topográfica y colapso a un perfil 1D de Vp por promedio ponderado inverso a
la distancia al centro del tendido → interpolación a las profundidades del
MASW → módulos elásticos → clasificación no supervisada (PCA 90 % + K-Means)
→ dos paneles con eje Y compartido y descarga de resultados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import servicios
from core.fisica import ETIQUETAS_MODULOS, MODULOS_ELASTICOS
from core.plots_1d import figura_perfiles_1d, figura_silhouette

# Límites por defecto del panel derecho, por módulo.
LIMITES_POR_DEFECTO = {"Coef_Poisson": (0.0, 5.0)}


def _limites(modulo: str, serie: pd.Series) -> tuple[float, float]:
    if modulo in LIMITES_POR_DEFECTO:
        return LIMITES_POR_DEFECTO[modulo]
    v = serie.replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return 0.0, 1.0
    margen = 0.05 * (v.max() - v.min() or 1.0)
    return float(v.min() - margen), float(v.max() + margen)


def render() -> None:
    st.subheader("Integración 1D/2D y módulos elásticos")
    st.caption(
        "Modelo 2D de Vp (sísmica de refracción) + modelo 1D de Vs (MASW) → "
        "perfil integrado y parámetros elásticos."
    )

    # ------------------------------------------------------------------ carga
    col_a, col_b, col_c = st.columns([1.1, 1.1, 1])
    with col_a:
        archivo_trs = st.file_uploader(
            "Modelo 2D de Vp — TRS  ·  columnas X, Elevación, Vp",
            type=["dat", "txt", "csv", "xyz"],
            key="t1_trs",
        )
    with col_b:
        archivo_masw = st.file_uploader(
            "Modelo 1D de Vs — MASW  ·  columnas Profundidad, Vs",
            type=["txt", "csv", "dat"],
            key="t1_masw",
        )
    with col_c:
        nombre_linea = st.text_input(
            "Nombre de la línea",
            value=st.session_state.get("t1_nombre_sugerido", "Linea_1"),
            key="t1_nombre",
            help="Se usa como identificador en los Tabs 2 y 3.",
        )

    if archivo_trs is None or archivo_masw is None:
        st.info(
            "Cargue ambos archivos para comenzar. El TRS admite Vp en km/s o m/s "
            "(se detecta y normaliza automáticamente a m/s)."
        )
        return

    # ------------------------------------------------------- parámetros físicos
    with st.expander("Parámetros de procesamiento", expanded=False):
        p1, p2, p3 = st.columns(3)
        with p1:
            modo_correccion = st.radio(
                "Corrección topográfica",
                ["por_columna", "global"],
                index=0,
                key="t1_modo_topo",
                help=(
                    "‘por_columna’ referencia cada traza vertical a la topografía "
                    "en su propia abscisa (z=0 en el terreno). ‘global’ resta la "
                    "cota máxima de todo el tendido."
                ),
            )
            unidad_vp = st.selectbox(
                "Unidad de Vp en el archivo TRS",
                ["auto", "km/s", "m/s"],
                index=0,
                key="t1_unidad",
            )
        with p2:
            potencia_idw = st.slider(
                "Potencia del IDW (p)",
                0.0,
                6.0,
                2.0,
                0.5,
                key="t1_potencia",
                help=(
                    "Peso w = 1/|x − x_centro|^p. Con p=0 todas las trazas pesan "
                    "igual; con p alto el resultado tiende a la columna central "
                    "(comportamiento del notebook original)."
                ),
            )
            n_nodos = st.slider("Nodos del perfil 1D", 50, 600, 200, 10, key="t1_nodos")
            usar_suav = st.checkbox(
                "Suavizado manual",
                key="t1_usar_suav",
                help=(
                    "Distancia s en w = 1/(d+s)^p. Sin ella se usa la mediana del "
                    "espaciamiento entre columnas, lo que evita que una traza "
                    "justo en el centro absorba todo el peso."
                ),
            )
            suavizado = (
                st.number_input("s (m)", 0.01, 500.0, 2.5, 0.25, key="t1_suav")
                if usar_suav
                else None
            )
        with p3:
            usar_radio = st.checkbox("Limitar radio de influencia", key="t1_usar_radio")
            radio = (
                st.number_input("Radio (m)", 1.0, 1e4, 30.0, 1.0, key="t1_radio")
                if usar_radio
                else None
            )
            usar_dens = st.checkbox("Densidad constante", key="t1_usar_dens")
            densidad_fija = (
                st.number_input("ρ (t/m³)", 0.5, 5.0, 2.0, 0.05, key="t1_dens")
                if usar_dens
                else None
            )
            extrapolar = st.checkbox(
                "Extrapolar Vp fuera del rango del TRS",
                value=True,
                key="t1_extrap",
                help="Igual que el notebook (interp1d con fill_value='extrapolate').",
            )

    # ----------------------------------------------------------- procesamiento
    try:
        df_trs = servicios.cargar_trs_2d(archivo_trs.getvalue(), unidad_vp)
        df_masw = servicios.cargar_masw_1d(archivo_masw.getvalue())
    except Exception as exc:
        st.error(f"Error al leer los archivos: {exc}")
        return

    prof_max_disponible = None
    try:
        df_res, perfil_vp_1d, info = servicios.integrar_trs_masw(
            df_trs,
            df_masw,
            potencia_idw,
            radio,
            n_nodos,
            prof_max_disponible,
            modo_correccion,
            densidad_fija,
            extrapolar,
            suavizado,
        )
    except Exception as exc:
        st.error(f"Error en el procesamiento: {exc}")
        return

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Columnas del TRS", info["n_columnas_usadas"])
    m2.metric("Centro del tendido", f"{info['x_centro']:.1f} m")
    m3.metric("Profundidad del perfil", f"{info['prof_max']:.1f} m")
    m4.metric("Nodos MASW", len(df_res))
    m5.metric(
        "Peso de la traza central",
        f"{100 * info['frac_peso_columna_central']:.1f} %",
        help="Fracción del peso total que aporta la columna más cercana al centro.",
    )

    n_extrapolados = int(df_res["Fuera_de_rango"].sum())
    if n_extrapolados:
        st.warning(
            f"{n_extrapolados} de {len(df_res)} profundidades del MASW quedan fuera "
            f"del rango del TRS ({info['prof_max']:.1f} m) y su Vp está extrapolada."
        )

    # ----------------------------------------------------------- clasificación
    with st.expander("Clasificación geofísica (PCA + K-Means)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            activar_clases = st.checkbox(
                "Ejecutar clasificación", value=True, key="t1_clasificar"
            )
        with c2:
            max_clases = st.slider("Máximo de clases a evaluar", 2, 10, 8, key="t1_maxk")
        with c3:
            varianza_pca = st.slider(
                "Varianza explicada a retener", 0.50, 0.99, 0.90, 0.01, key="t1_pca"
            )
        replicar = st.checkbox(
            "Replicar el k+1 del notebook original",
            value=False,
            key="t1_bug",
            help=(
                "El notebook guardaba 'mejor_k = k + 1', por lo que entrenaba con "
                "un k distinto al que maximizó la silueta. Actívelo sólo para "
                "reproducir resultados históricos."
            ),
        )

    diag = None
    if activar_clases:
        try:
            df_res, diag = servicios.clasificar(
                df_res, max_clases, varianza_pca, replicar
            )
        except Exception as exc:
            st.warning(f"No se pudo clasificar: {exc}")

    # ------------------------------------------------------------------ gráfico
    st.divider()
    g1, g2, g3, g4 = st.columns([1.4, 1, 1, 1.1])
    with g1:
        modulos = [m for m in MODULOS_ELASTICOS if m in df_res.columns]
        modulo = st.selectbox(
            "Módulo elástico (panel derecho)",
            modulos,
            index=modulos.index("Coef_Poisson") if "Coef_Poisson" in modulos else 0,
            format_func=lambda m: ETIQUETAS_MODULOS.get(m, (m, ""))[0],
            key="t1_modulo",
        )
    lim_def = _limites(modulo, df_res[modulo])
    with g2:
        x_min = st.number_input("X mín.", value=float(lim_def[0]), key=f"t1_xmin_{modulo}")
    with g3:
        x_max = st.number_input("X máx.", value=float(lim_def[1]), key=f"t1_xmax_{modulo}")
    with g4:
        sentido = st.radio(
            "Eje Y (profundidad)",
            ["Invertido (0 arriba)", "Normal (0 abajo)"],
            index=0,
            horizontal=False,
            key="t1_invertir",
        )

    fig = figura_perfiles_1d(
        df_res,
        modulo=modulo,
        invertir_y=sentido.startswith("Invertido"),
        x_min_modulo=x_min,
        x_max_modulo=x_max,
        perfil_vp_1d=perfil_vp_1d,
        mostrar_clases=activar_clases and "Clase_Geofisica" in df_res.columns,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------ diagnósticos
    if diag is not None:
        d1, d2 = st.columns([1, 1.4])
        with d1:
            st.markdown(
                f"**PCA:** {diag['n_variables']} variables → "
                f"{diag['n_componentes']} componentes "
                f"({100 * diag['varianza_acumulada']:.1f} % de varianza).  \n"
                f"**k óptimo:** {diag['k_optimo']} "
                f"(silueta {diag['mejor_silhouette']:.4f})"
            )
            st.plotly_chart(
                figura_silhouette(diag["curva_silhouette"], diag["k_optimo"]),
                use_container_width=True,
            )
        with d2:
            st.caption("Perfil medio de cada clase geofísica")
            st.dataframe(diag["resumen_clases"], use_container_width=True, height=220)

    with st.expander("Tabla de resultados", expanded=False):
        st.dataframe(df_res, use_container_width=True, height=320)

    # -------------------------------------------------------------- descargas
    st.divider()
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Descargar resultados (CSV)",
            servicios.dataframe_a_csv(df_res),
            file_name=f"{nombre_linea}_TRS+MASW_ResInterp.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Descargar perfil 1D de Vp (CSV)",
            servicios.dataframe_a_csv(perfil_vp_1d),
            file_name=f"{nombre_linea}_Vp_1D_IDW.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d3:
        if st.button("Registrar línea para los Tabs 2 y 3", use_container_width=True):
            st.session_state.setdefault("lineas", {})
            st.session_state["lineas"][nombre_linea] = {
                "nombre": nombre_linea,
                "df_trs": df_trs,
                "df_masw": df_masw,
                "df_resultado": df_res,
                "perfil_vp_1d": perfil_vp_1d,
                "info_idw": info,
            }
            st.success(
                f"Línea «{nombre_linea}» registrada. "
                "Ya puede asignarle topografía en el Tab 2."
            )
