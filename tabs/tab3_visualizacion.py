"""
tabs/tab3_visualizacion.py
==========================

**Tab 3 — Visualización 3D y mapas base.**

* Toma los perfiles 2D generados en el Tab 2 (o CSV importados) y los
  renderiza como cortinas 3D que siguen estrictamente la topografía, con
  escala ``rainbow`` y líneas de contorno parametrizables.
* Imagen base: GeoTIFF (extrae su extensión automáticamente) o JPG/PNG con
  esquinas inferior-izquierda y superior-derecha declaradas.
* Exportación: HTML interactivo y cortes estáticos 2D con la estética de
  ``plot_2d_puentes.ipynb``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import servicios
from core.plots_2d import CMAPS_DISPONIBLES
from core.plots_3d import (
    agregar_base_satelital_3d_color,
    figura_a_html,
    plot_3d_variable,
)

CMAPS_3D = ["rainbow", "jet", "turbo", "viridis", "portland", "spectral"]


def _fuentes_disponibles() -> dict:
    """Reúne los perfiles del Tab 2 y los CSV importados en esta pestaña."""
    fuentes = {}
    for nombre, entrada in st.session_state.get("perfiles_2d", {}).items():
        fuentes[nombre] = entrada["df"]
    for nombre, df in st.session_state.get("importados_3d", {}).items():
        fuentes[nombre] = df
    return fuentes


def _importador() -> None:
    archivos = st.file_uploader(
        "Importar CSV de perfiles 2D (salida del Tab 2 o `modelos_2d_vp_vs_*.csv`)",
        type=["csv", "txt"],
        accept_multiple_files=True,
        key="t3_csv",
    )
    if not archivos:
        return
    st.session_state.setdefault("importados_3d", {})
    for archivo in archivos:
        try:
            df = servicios.cargar_modelo_2d(archivo.getvalue())
            st.session_state["importados_3d"][archivo.name] = df
        except Exception as exc:
            st.error(f"{archivo.name}: {exc}")


def _mapa_base(fig, z_default: float):
    with st.expander("Imagen base georreferenciada", expanded=False):
        archivo = st.file_uploader(
            "GeoTIFF, JPG o PNG",
            type=["tif", "tiff", "jpg", "jpeg", "png"],
            key="t3_img",
        )
        if archivo is None:
            return fig

        es_geotiff = archivo.name.lower().endswith((".tif", ".tiff"))
        ext = None
        if es_geotiff:
            try:
                ext = servicios.extension_geotiff(archivo.getvalue())
                st.success(
                    f"Georreferencia extraída con {ext['motor']} · CRS: {ext['crs']} · "
                    f"X [{ext['x_min']:.2f}, {ext['x_max']:.2f}] · "
                    f"Y [{ext['y_min']:.2f}, {ext['y_max']:.2f}]"
                )
            except Exception as exc:
                st.warning(f"{exc}  Declare las esquinas manualmente.")

        if ext is None:
            st.caption("Esquinas delimitadoras en coordenadas UTM")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                x_min = st.number_input("X inferior-izquierda", value=0.0, format="%.2f",
                                        key="t3_xmin")
            with c2:
                y_min = st.number_input("Y inferior-izquierda", value=0.0, format="%.2f",
                                        key="t3_ymin")
            with c3:
                x_max = st.number_input("X superior-derecha", value=100.0, format="%.2f",
                                        key="t3_xmax")
            with c4:
                y_max = st.number_input("Y superior-derecha", value=100.0, format="%.2f",
                                        key="t3_ymax")
            ext = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

        c5, c6 = st.columns(2)
        with c5:
            z_base = st.number_input(
                "Cota del plano base (m)", value=float(z_default), format="%.2f",
                key="t3_zbase",
            )
        with c6:
            opacidad = st.slider("Opacidad", 0.1, 1.0, 0.85, 0.05, key="t3_opac")

        try:
            fig = agregar_base_satelital_3d_color(
                fig,
                archivo.getvalue(),
                ext["x_min"],
                ext["x_max"],
                ext["y_min"],
                ext["y_max"],
                z_base,
                opacidad=opacidad,
            )
        except Exception as exc:
            st.error(f"No se pudo agregar la imagen base: {exc}")
    return fig


def render() -> None:
    st.subheader("Visualización 3D y mapas base")
    st.caption(
        "Cortinas 3D que siguen la topografía real, con contornos de isovelocidad "
        "y contexto satelital."
    )

    _importador()
    fuentes = _fuentes_disponibles()
    if not fuentes:
        st.info(
            "Genere perfiles en el Tab 2 o importe un CSV con columnas "
            "`X, Y, Elevacion` (o `Z`) y la variable a graficar."
        )
        return

    seleccion = st.multiselect(
        "Perfiles a renderizar",
        list(fuentes.keys()),
        default=list(fuentes.keys())[:3],
        key="t3_sel",
    )
    if not seleccion:
        return

    df_todos = pd.concat(
        [fuentes[n].assign(_Perfil=n) for n in seleccion], ignore_index=True
    )

    excluidas = {
        "X",
        "Y",
        "Z",
        "Elevacion",
        "Xo",
        "Z_superficie",
        "Profundidad",
        "dist_centroide",
        "dist_al_masw_mas_cercano",
        "Vp_x_Elevacion",
    }
    numericas = [
        c
        for c in df_todos.columns
        if pd.api.types.is_numeric_dtype(df_todos[c])
        and not pd.api.types.is_bool_dtype(df_todos[c])
        and c not in excluidas
    ]
    if not numericas:
        st.error("No se encontró ninguna variable numérica para colorear.")
        return

    # ------------------------------------------------------------- controles
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        var = st.selectbox(
            "Variable",
            numericas,
            index=numericas.index("Vs") if "Vs" in numericas else 0,
            key="t3_var",
        )
    with c2:
        candidatas_z = [c for c in ["Elevacion", "Z"] if c in df_todos.columns]
        z_col = st.selectbox("Columna de elevación", candidatas_z, key="t3_zcol")
    with c3:
        modo = st.radio("Representación", ["surface", "scatter"], index=0, key="t3_modo")
    with c4:
        cmap = st.selectbox("Escala de color", CMAPS_3D, index=0, key="t3_cmap")
    with c5:
        grid_res = st.slider("Resolución de la cortina", 20, 200, 60, 10, key="t3_res")

    v1, v2, v3 = st.columns([1, 1, 2])
    serie = df_todos[var].replace([np.inf, -np.inf], np.nan).dropna()
    with v1:
        vmin = st.number_input("Mín. de color", value=float(serie.min()), key="t3_vmin")
    with v2:
        vmax = st.number_input("Máx. de color", value=float(serie.max()), key="t3_vmax")
    with v3:
        texto_contornos = st.text_input(
            "Contornos (valores separados por coma)",
            value="300, 720",
            key="t3_cont",
            help="Isolíneas dibujadas sobre la cortina 3D y sobre los cortes 2D.",
        )
    contornos = []
    for token in texto_contornos.split(","):
        token = token.strip()
        if token:
            try:
                contornos.append(float(token))
            except ValueError:
                pass

    # ---------------------------------------------------------------- figura
    fig = None
    for nombre in seleccion:
        df_i = fuentes[nombre].copy()
        df_i["_Perfil"] = nombre
        fig = plot_3d_variable(
            df_i,
            var_color=var,
            x_col="X",
            y_col="Y",
            z_col=z_col,
            line_col="_Perfil",
            fig=fig,
            mode=modo,
            cmap=cmap,
            grid_res=grid_res,
            vmin=vmin,
            vmax=vmax,
            contornos=contornos or None,
            name_prefix="",
            titulo=f"Modelo 3D de {var}",
            subtitulo=" · ".join(seleccion),
        )

    z_default = float(df_todos[z_col].min())
    fig = _mapa_base(fig, z_default)

    st.plotly_chart(fig, use_container_width=True, height=760)

    st.download_button(
        "⬇️ Descargar visualización 3D interactiva (HTML)",
        figura_a_html(fig),
        file_name=f"modelo_3d_{var}.html",
        mime="text/html",
    )

    # ------------------------------------------------------------ cortes 2D
    st.divider()
    st.markdown("##### Cortes estáticos 2D")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        perfil_2d = st.selectbox("Perfil", seleccion, key="t3_p2d")
    with b2:
        cmap_2d = st.selectbox(
            "Escala", CMAPS_DISPONIBLES, index=0, key="t3_cmap2d"
        )
    with b3:
        prof_corte = st.number_input(
            "Profundidad del recorte (m)", 1.0, 300.0, 30.0, 1.0, key="t3_prof2d"
        )
    with b4:
        metodo = st.selectbox(
            "Interpolación", ["linear", "cubic", "nearest"], key="t3_met2d"
        )

    df_2d = fuentes[perfil_2d]
    col_x = "Xo" if "Xo" in df_2d.columns else "X"
    if z_col not in df_2d.columns:
        st.warning(f"El perfil «{perfil_2d}» no tiene la columna «{z_col}».")
        return

    try:
        png = servicios.corte_2d_png(
            df_2d,
            col_x,
            z_col,
            var,
            metodo,
            float(vmin),
            float(vmax),
            float(prof_corte),
            cmap_2d,
            tuple(contornos),
            200,
            f"{perfil_2d} — perfil continuo 2D (recortado al relieve)",
        )
    except Exception as exc:
        st.error(f"No se pudo generar el corte 2D: {exc}")
        return

    st.image(png, use_container_width=True)
    st.download_button(
        "⬇️ Descargar corte 2D (PNG, 300 dpi)",
        png,
        file_name=f"Seccion_{perfil_2d.replace(' · ', '_').replace(' ', '_')}_{var}.png",
        mime="image/png",
    )
