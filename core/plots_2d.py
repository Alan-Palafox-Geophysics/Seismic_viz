"""
core/plots_2d.py
================

Cortes estáticos 2D fieles a ``plot_2d_puentes.ipynb``:

* Interpolación ``scipy.griddata`` sobre una malla regular distancia ×
  elevación.
* Recorte al relieve real (``groupby(col_x)[col_y].max()``) por arriba y a
  una profundidad máxima por abajo.
* Relleno continuo con 256 niveles y escala de color configurable — se
  conserva la rampa personalizada
  *morado → azul → cian → verde → amarillo → naranja → rojo → rojo oscuro*
  de ``crear_colormap_personalizado``, además de ``jet`` y ``rainbow``.
* Líneas de contorno parametrizables (p. ej. Vs = 720 m/s como límite
  geotécnico), etiquetadas y calculadas sobre la misma malla interpolada.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata, interp1d

CMAPS_DISPONIBLES = [
    "espectro_personalizado",
    "jet",
    "rainbow",
    "turbo",
    "viridis",
    "nipy_spectral",
    "Spectral_r",
]


def crear_colormap_personalizado(nombre: str = "espectro_personalizado"):
    """
    Rampa continua, de menor a mayor valor:

        morado → azul → cian → verde → amarillo → naranja → rojo → rojo oscuro
    """
    colores = [
        (0.000, "#6A00A8"),  # morado
        (0.125, "#0000FF"),  # azul
        (0.250, "#00FFFF"),  # cian
        (0.375, "#00FF00"),  # verde claro
        (0.500, "#00AA00"),  # verde
        (0.660, "#FFFF00"),  # amarillo
        (0.755, "#FF8000"),  # naranja
        (0.900, "#FF0000"),  # rojo
        (1.000, "#8B0000"),  # rojo oscuro
    ]
    posiciones = [c[0] for c in colores]
    hex_colores = [c[1] for c in colores]
    return LinearSegmentedColormap.from_list(
        nombre, list(zip(posiciones, hex_colores)), N=256
    )


CMAP_ESPECTRO = crear_colormap_personalizado()


def _resolver_cmap(cmap):
    if cmap is None or cmap == "espectro_personalizado":
        return crear_colormap_personalizado()
    if isinstance(cmap, str):
        return plt.get_cmap(cmap)
    return cmap


def colormap_a_plotly(cmap=None, n: int = 64) -> list:
    """
    Traduce un colormap de matplotlib a una ``colorscale`` de Plotly.

    Devuelve la lista ``[[posición, 'rgb(r,g,b)'], …]`` que entienden
    ``go.Surface`` y ``go.Scatter3d``.  Es lo que permite usar en el 3D
    exactamente la misma rampa personalizada que en los cortes 2D, en vez
    de una escala con nombre distinta.
    """
    cm = _resolver_cmap(cmap)
    escala = []
    for i in range(n):
        pos = i / (n - 1)
        r, g, b = (int(round(255 * c)) for c in cm(pos)[:3])
        escala.append([pos, f"rgb({r},{g},{b})"])
    return escala


def exportar_slide_2d_recortado(
    df: pd.DataFrame,
    col_x: str = "Xo",
    col_y: str = "Elevacion",
    col_z: str = "Vs",
    metodo: str = "linear",
    vmin: float | None = None,
    vmax: float | None = None,
    profundidad_max: float = 30.0,
    cmap=None,
    contornos: list[float] | None = None,
    color_contorno: str = "black",
    grosor_contorno: float = 1.75,
    etiquetar_contornos: bool = True,
    resolucion: int = 200,
    titulo: str = "Perfil Continuo 2D (Recortado al Relieve)",
    etiqueta_color: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    dpi: int = 300,
    nombre_archivo: str | None = None,
):
    """
    Genera el perfil 2D continuo recortado al relieve.

    Devuelve ``(fig, png_bytes)``.  Si se indica ``nombre_archivo`` también
    lo guarda en disco con ``dpi`` y ``bbox_inches='tight'``, igual que el
    notebook.
    """
    x = df[col_x].to_numpy(float)
    y = df[col_y].to_numpy(float)
    z = df[col_z].to_numpy(float)

    resolucion = int(max(resolucion, 20))
    xi = np.linspace(x.min(), x.max(), resolucion)
    yi = np.linspace(y.min(), y.max(), resolucion)
    xi, yi = np.meshgrid(xi, yi)

    # Perfil topográfico real
    superficie = df.groupby(col_x)[col_y].max().reset_index()
    f_superficie = interp1d(
        superficie[col_x], superficie[col_y], bounds_error=False, fill_value=np.nan
    )
    limite_topografico = f_superficie(xi[0, :])

    metodo = metodo if metodo in {"linear", "cubic", "nearest"} else "linear"
    zi = griddata((x, y), z, (xi, yi), method=metodo)

    # Máscara topográfica (recorte superior e inferior)
    for i in range(resolucion):
        elev_max = limite_topografico[i]
        if np.isnan(elev_max):
            zi[:, i] = np.nan
            continue
        elev_min = elev_max - profundidad_max
        zi[yi[:, i] > elev_max, i] = np.nan
        zi[yi[:, i] < elev_min, i] = np.nan

    # Escala de color
    if vmin is not None and vmax is not None:
        niveles = np.linspace(vmin, vmax, 256)
    elif vmin is not None:
        niveles = np.linspace(vmin, np.nanmax(zi), 256)
    elif vmax is not None:
        niveles = np.linspace(np.nanmin(zi), vmax, 256)
    else:
        niveles = 256

    fig, ax = plt.subplots(figsize=figsize)
    slide = ax.contourf(xi, yi, zi, levels=niveles, cmap=_resolver_cmap(cmap), extend="both")

    if contornos:
        lineas = ax.contour(
            xi,
            yi,
            zi,
            levels=sorted(float(c) for c in contornos),
            colors=color_contorno,
            linewidths=grosor_contorno,
        )
        if etiquetar_contornos:
            ax.clabel(lineas, inline=True, fontsize=8, fmt="%g")

    ax.plot(superficie[col_x], superficie[col_y], color="black", linewidth=1.75)
    ax.plot(
        superficie[col_x],
        superficie[col_y] - profundidad_max,
        color="black",
        linewidth=1.75,
    )

    fig.colorbar(slide, ax=ax, label=etiqueta_color or f"Velocidad ({col_z}) [m/s]")
    ax.set_xlabel(f"Distancia ({col_x}) [m]")
    ax.set_ylabel(f"{col_y} [m]")
    ax.set_title(titulo)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    buffer.seek(0)

    if nombre_archivo:
        fig.savefig(nombre_archivo, dpi=dpi, bbox_inches="tight")

    return fig, buffer.getvalue()


def figura_topografia(topo_linea: pd.DataFrame, titulo: str = "Perfil topográfico"):
    """Gráfico rápido de la topografía de una línea (Xo vs Z)."""
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(topo_linea["Xo"], topo_linea["Z"], color="#2b2b33", linewidth=2)
    ax.fill_between(
        topo_linea["Xo"], topo_linea["Z"], topo_linea["Z"].min() - 1, color="#e6e8eb"
    )
    ax.set_xlabel("Xo [m]")
    ax.set_ylabel("Cota Z [m]")
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    return fig, buffer.getvalue()
