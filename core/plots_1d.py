"""
core/plots_1d.py
================

Figura interactiva del Tab 1: dos subplots que comparten el eje Y
(profundidad).

* Izquierda — perfiles 1D de Vp (TRS colapsado por IDW) y Vs (MASW), con el
  eje X en velocidad (m/s).
* Derecha — el módulo elástico seleccionado, con límites de X configurables
  (por defecto Coeficiente de Poisson de 0 a 5).

El sentido del eje Y se controla desde fuera con ``invertir_y``: la
inversión se aplica al eje compartido, de modo que ambos paneles quedan
siempre alineados.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .fisica import ETIQUETAS_MODULOS

# Paleta consistente para los dos paneles.
COLOR_VP = "#D1495B"
COLOR_VS = "#00798C"
COLOR_MODULO = "#3E4C59"
COLOR_CLASE = "#EDAE49"


def figura_perfiles_1d(
    df_resultado: pd.DataFrame,
    modulo: str = "Coef_Poisson",
    invertir_y: bool = True,
    x_min_modulo: float | None = 0.0,
    x_max_modulo: float | None = 5.0,
    perfil_vp_1d: pd.DataFrame | None = None,
    mostrar_clases: bool = False,
    altura: int = 720,
) -> go.Figure:
    """
    Construye la figura de dos paneles del Tab 1.

    Parameters
    ----------
    df_resultado : pd.DataFrame
        Salida de :func:`core.fisica.integrar_trs_masw` (opcionalmente ya
        clasificada, con la columna ``Clase_Geofisica``).
    modulo : str
        Columna a graficar en el panel derecho.
    invertir_y : bool
        True ⇒ la profundidad crece hacia abajo (convención geofísica).
    x_min_modulo, x_max_modulo : float | None
        Límites del eje X derecho.  ``None`` en cualquiera de los dos deja
        ese extremo en automático.
    perfil_vp_1d : pd.DataFrame | None
        Perfil denso de Vp antes de interpolar al MASW.  Si se pasa, se
        dibuja como línea tenue de referencia.
    mostrar_clases : bool
        Sombrea el panel derecho por ``Clase_Geofisica`` si existe.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    titulo_mod, unidad_mod = ETIQUETAS_MODULOS.get(modulo, (modulo, ""))

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=(
            "Perfiles de velocidad (Vp / Vs)",
            titulo_mod,
        ),
    )

    prof = df_resultado["Profundidad"]

    # --- Panel izquierdo: velocidades ------------------------------------
    if perfil_vp_1d is not None and not perfil_vp_1d.empty:
        fig.add_trace(
            go.Scatter(
                x=perfil_vp_1d["Vp"],
                y=perfil_vp_1d["Profundidad"],
                mode="lines",
                name="Vp TRS (perfil denso IDW)",
                line=dict(color=COLOR_VP, width=1, dash="dot"),
                opacity=0.55,
                hovertemplate="Vp: %{x:.1f} m/s<br>z: %{y:.2f} m<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=df_resultado["Vp_TRS_interpolado"],
            y=prof,
            mode="lines+markers",
            name="Vp (TRS interpolada)",
            line=dict(color=COLOR_VP, width=2.4, shape="hv"),
            marker=dict(size=5),
            hovertemplate="Vp: %{x:.1f} m/s<br>z: %{y:.2f} m<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_resultado["Vs_MASW_original"],
            y=prof,
            mode="lines+markers",
            name="Vs (MASW)",
            line=dict(color=COLOR_VS, width=2.4, shape="hv"),
            marker=dict(size=5),
            hovertemplate="Vs: %{x:.1f} m/s<br>z: %{y:.2f} m<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # --- Panel derecho: módulo elástico ----------------------------------
    if modulo in df_resultado.columns:
        fig.add_trace(
            go.Scatter(
                x=df_resultado[modulo],
                y=prof,
                mode="lines+markers",
                name=titulo_mod,
                line=dict(color=COLOR_MODULO, width=2.4, shape="hv"),
                marker=dict(size=5),
                hovertemplate=(
                    f"{titulo_mod}: %{{x:.4f}}<br>z: %{{y:.2f}} m<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    if mostrar_clases and "Clase_Geofisica" in df_resultado.columns:
        clases = df_resultado["Clase_Geofisica"].dropna().unique()
        for c in sorted(clases):
            sub = df_resultado.loc[df_resultado["Clase_Geofisica"] == c]
            fig.add_trace(
                go.Scatter(
                    x=sub[modulo],
                    y=sub["Profundidad"],
                    mode="markers",
                    name=f"Clase {int(c)}",
                    marker=dict(size=11, symbol="circle-open", line=dict(width=2)),
                    hovertemplate=(
                        f"Clase {int(c)}<br>%{{x:.4f}}<br>z: %{{y:.2f}} m<extra></extra>"
                    ),
                ),
                row=1,
                col=2,
            )

    # --- Ejes -------------------------------------------------------------
    fig.update_xaxes(
        title_text="Velocidad (m/s)",
        row=1,
        col=1,
        gridcolor="#e6e8eb",
        zeroline=False,
    )
    rango_x2 = None
    if x_min_modulo is not None and x_max_modulo is not None:
        rango_x2 = [float(x_min_modulo), float(x_max_modulo)]
    fig.update_xaxes(
        title_text=f"{titulo_mod} [{unidad_mod}]" if unidad_mod else titulo_mod,
        row=1,
        col=2,
        range=rango_x2,
        gridcolor="#e6e8eb",
        zeroline=False,
    )

    # El eje Y es compartido: basta invertir el del primer subplot.
    fig.update_yaxes(
        title_text="Profundidad (m)",
        autorange="reversed" if invertir_y else True,
        row=1,
        col=1,
        gridcolor="#e6e8eb",
    )
    fig.update_yaxes(row=1, col=2, gridcolor="#e6e8eb")

    fig.update_layout(
        height=altura,
        hovermode="y unified",
        paper_bgcolor="#f4f5f7",
        plot_bgcolor="#ffffff",
        font=dict(family="Arial, sans-serif", color="#2b2b33", size=13),
        margin=dict(l=60, r=20, t=70, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#d7d9dd",
            borderwidth=1,
        ),
    )
    return fig


def figura_silhouette(curva: pd.DataFrame, k_optimo: int) -> go.Figure:
    """Curva de Silhouette Score vs. número de clases."""
    fig = go.Figure()
    if curva is not None and not curva.empty:
        fig.add_trace(
            go.Scatter(
                x=curva["k"],
                y=curva["silhouette"],
                mode="lines+markers",
                line=dict(color=COLOR_VS, width=2.4),
                marker=dict(size=8),
                name="Silhouette",
            )
        )
        fig.add_vline(
            x=k_optimo,
            line=dict(color=COLOR_CLASE, width=2, dash="dash"),
            annotation_text=f"k = {k_optimo}",
            annotation_position="top",
        )
    fig.update_layout(
        height=280,
        xaxis_title="Número de clases (k)",
        yaxis_title="Silhouette Score",
        paper_bgcolor="#f4f5f7",
        plot_bgcolor="#ffffff",
        margin=dict(l=50, r=20, t=30, b=40),
        showlegend=False,
    )
    return fig
