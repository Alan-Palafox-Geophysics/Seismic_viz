"""
core/plots_3d.py
================

Visualización 3D interactiva (estilo dashboard), evolución directa de
``plot_3d_mejorado.py``.

Novedades respecto al módulo original
-------------------------------------
* ``mode='surface'`` (curtain plot) devuelve también la parametrización
  distancia–elevación de cada línea, lo que permite **superponer líneas de
  contorno parametrizables** directamente sobre la cortina 3D
  (:func:`agregar_contornos_3d`), en los mismos valores que se usan en los
  cortes 2D.
* Soporte de imagen base **GeoTIFF** con extracción automática de sus
  coordenadas (:func:`leer_extension_geotiff`), además de JPG/PNG con
  esquinas declaradas por el usuario.

El criterio de color es el mismo que el de los cortes 2D (incluida la rampa
personalizada, traducida a una ``colorscale`` de Plotly), la barra de color
va horizontal abajo a la derecha, y el único control de la cabecera es la
vista de cámara: mostrar u ocultar líneas se hace con un click en la leyenda.
"""

from __future__ import annotations

import io
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from scipy.interpolate import griddata

from .plots_2d import CMAPS_DISPONIBLES, colormap_a_plotly

# Escalas disponibles para el 3D.  Son las mismas que las de los cortes 2D,
# de modo que una sección impresa y su cortina 3D puedan compartir rampa.
CMAPS_3D = list(CMAPS_DISPONIBLES)

# Nombres que sólo existen en matplotlib: hay que traducirlos a una
# colorscale explícita porque Plotly no los conoce.
_CMAPS_MATPLOTLIB = {
    "espectro_personalizado",
    "nipy_spectral",
    "Spectral_r",
    "gist_rainbow",
}


def resolver_colorscale(cmap):
    """Devuelve algo que Plotly entienda: un nombre suyo o una escala explícita."""
    if cmap is None:
        return colormap_a_plotly(None)
    if isinstance(cmap, str) and cmap in _CMAPS_MATPLOTLIB:
        return colormap_a_plotly(cmap)
    return cmap

# ---------------------------------------------------------------------------
# Paleta / estilo "dashboard" (sólo layout, NO toca colorscales de datos)
# ---------------------------------------------------------------------------
_DASHBOARD_LAYOUT = dict(
    paper_bgcolor="#f4f5f7",
    plot_bgcolor="#ffffff",
    font=dict(family="Arial, sans-serif", color="#2b2b33", size=13),
    margin=dict(l=10, r=10, t=90, b=10),
    legend=dict(
        title=dict(text="Líneas (click para mostrar/ocultar)"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#d7d9dd",
        borderwidth=1,
        x=1.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
    ),
)

_SCENE_STYLE = dict(
    xaxis=dict(backgroundcolor="#eef0f3", gridcolor="#d7d9dd", showspikes=False),
    yaxis=dict(backgroundcolor="#eef0f3", gridcolor="#d7d9dd", showspikes=False),
    zaxis=dict(backgroundcolor="#eef0f3", gridcolor="#d7d9dd", showspikes=False),
    aspectmode="data",
)

_CAMARAS = {
    "Isométrica": dict(eye=dict(x=1.4, y=1.4, z=1.2)),
    "Superior (planta)": dict(eye=dict(x=0.0, y=0.0, z=2.4), up=dict(x=0, y=1, z=0)),
    "Frontal (N-S)": dict(eye=dict(x=0.0, y=2.4, z=0.2)),
    "Lateral (E-O)": dict(eye=dict(x=2.4, y=0.0, z=0.2)),
}

NOMBRE_BASE = "Base Satelital Color"


# ---------------------------------------------------------------------------
# Rango de color consistente entre llamadas
# ---------------------------------------------------------------------------
def _rango_color(df, var_color, fig, vmin, vmax):
    if vmin is not None and vmax is not None:
        return vmin, vmax

    data_min, data_max = float(df[var_color].min()), float(df[var_color].max())

    if fig is not None and len(fig.data) > 0:
        prev_min, prev_max = [], []
        for tr in fig.data:
            if tr.name == NOMBRE_BASE:
                continue
            marker = getattr(tr, "marker", None)
            if marker is not None and getattr(marker, "cmin", None) is not None:
                prev_min.append(marker.cmin)
                prev_max.append(marker.cmax)
            if getattr(tr, "cmin", None) is not None:
                prev_min.append(tr.cmin)
                prev_max.append(tr.cmax)
        if prev_min:
            data_min = min(data_min, min(prev_min))
            data_max = max(data_max, max(prev_max))

    return (vmin if vmin is not None else data_min, vmax if vmax is not None else data_max)


def _colorbar_horizontal(titulo: str) -> dict:
    """
    Barra de color horizontal anclada abajo a la derecha del lienzo.

    Se saca del costado para no competir con la leyenda y para que la escena
    3D use todo el ancho disponible.
    """
    return dict(
        title=dict(text=titulo, side="top"),
        orientation="h",
        x=0.98,
        xanchor="right",
        y=0.02,
        yanchor="bottom",
        len=0.38,
        thickness=14,
        outlinewidth=0,
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="#d7d9dd",
        borderwidth=1,
        tickfont=dict(size=11),
    )


# ---------------------------------------------------------------------------
# Curtain plot
# ---------------------------------------------------------------------------
def construir_curtain(
    df_line: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    var_color: str,
    grid_res: int = 60,
    profundidad_max: float | None = None,
) -> dict | None:
    """
    Interpola ``var_color`` en una malla **que sigue el terreno** y la
    proyecta en 3D sobre el trazado real de la línea.

    La malla no es un rectángulo ``z.min()…z.max()``: para cada posición a
    lo largo de la línea, la vertical arranca en la topografía de esa
    abscisa y baja ``profundidad_max`` metros.  Así la cortina queda
    recortada por el relieve con el mismo criterio que
    :func:`core.plots_2d.exportar_slide_2d_recortado` — sin el escalón que
    producía rellenar el rectángulo por vecino más cercano, que pintaba
    material por encima del terreno.

    Parameters
    ----------
    profundidad_max : float | None
        Espesor de la cortina bajo la topografía.  Por defecto, el espesor
        real máximo del modelo, de modo que no se recorta nada.

    Returns
    -------
    dict | None
        ``{'X','Y','Z','V','D','d','x','y','superficie','base','Z_cont'}``.
        ``Z`` lleva ``NaN`` donde la malla cae por debajo del dato (Plotly
        abre un hueco ahí); ``Z_cont`` es la versión finita que usan los
        contornos.  ``None`` si hay menos de 4 puntos.

    Notas de costo
    --------------
    Una interpolación 2D (``scipy.griddata``) por línea.  Para el tamaño
    típico de una línea de prospección (cientos a pocos miles de nodos) es
    prácticamente instantánea.
    """
    df_line = df_line.sort_values(by=[x_col, y_col])
    x = df_line[x_col].to_numpy(float)
    y = df_line[y_col].to_numpy(float)
    z = df_line[z_col].to_numpy(float)
    v = df_line[var_color].to_numpy(float)

    if len(df_line) < 4:
        return None

    # Distancia acumulada a lo largo del trazado (parámetro horizontal)
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])

    # Techo y piso del dato en cada estación, igual que el groupby(col_x).max()
    # que usa el corte 2D para extraer el perfil topográfico.
    estaciones = pd.DataFrame({"d": d, "z": z}).groupby("d", as_index=False)["z"].agg(
        ["max", "min"]
    )
    estaciones.columns = ["d", "techo", "piso"]
    if len(estaciones) < 2:
        return None

    d_grid = np.linspace(d.min(), d.max(), grid_res)
    superficie = np.interp(d_grid, estaciones["d"], estaciones["techo"])
    piso_dato = np.interp(d_grid, estaciones["d"], estaciones["piso"])

    if profundidad_max is None:
        profundidad_max = float(np.max(estaciones["techo"] - estaciones["piso"]))
    profundidad_max = max(float(profundidad_max), 1e-6)

    # Malla que sigue el terreno: profundidad normalizada 0…1 bajo la superficie
    t = np.linspace(0.0, 1.0, grid_res)
    D = np.tile(d_grid, (grid_res, 1))
    S = np.tile(superficie, (grid_res, 1))
    Z_cont = S - t[:, None] * profundidad_max

    V = griddata((d, z), v, (D, Z_cont), method="linear")
    V_nn = griddata((d, z), v, (D, Z_cont), method="nearest")
    V = np.where(np.isnan(V), V_nn, V)

    # Por debajo del dato no se inventa nada: se abre hueco, como el blanco
    # que deja el corte 2D bajo el modelo.
    fuera = Z_cont < np.tile(piso_dato, (grid_res, 1)) - 1e-9
    V = np.where(fuera, np.nan, V)
    Z = np.where(fuera, np.nan, Z_cont)

    # Mapear distancia -> (X, Y) reales siguiendo el trazado de la línea
    X = np.interp(D, d, x)
    Y = np.interp(D, d, y)

    return {
        "X": X,
        "Y": Y,
        "Z": Z,
        "Z_cont": Z_cont,
        "V": V,
        "D": D,
        "d": d,
        "x": x,
        "y": y,
        "superficie": superficie,
        "base": superficie - profundidad_max,
        "d_grid": d_grid,
        "profundidad_max": profundidad_max,
    }


def _segmentos_contorno(D, Z, V, nivel: float):
    """Extrae las polilíneas del contorno ``V = nivel`` en el plano (D, Z)."""
    try:
        from contourpy import contour_generator

        gen = contour_generator(D, Z, V)
        lineas = gen.lines(float(nivel))
        segmentos = []
        for ln in lineas:
            arr = np.asarray(ln, dtype=float)
            if arr.ndim == 2 and len(arr) >= 2:
                segmentos.append(arr)
        return segmentos
    except Exception:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure()
        cs = plt.contour(D, Z, V, levels=[float(nivel)])
        segmentos = [np.asarray(s) for s in cs.allsegs[0] if len(s) >= 2]
        plt.close(fig)
        return segmentos


def agregar_contornos_3d(
    fig: go.Figure,
    curtain: dict,
    niveles: Iterable[float],
    nombre_linea: str,
    color: str = "black",
    grosor: float = 3.0,
    mostrar_leyenda: bool = True,
) -> go.Figure:
    """
    Superpone líneas de isovalor sobre una cortina 3D.

    Los contornos se calculan en el plano paramétrico (distancia, elevación)
    de la cortina y luego se proyectan al trazado real (X, Y), de modo que
    la isolínea sigue estrictamente la geometría de la línea y su
    topografía.
    """
    # Los contornos se calculan sobre la malla finita (``Z_cont``); los huecos
    # ya vienen marcados como NaN en ``V``, que es lo que contourpy respeta.
    D, V = curtain["D"], curtain["V"]
    Z = curtain.get("Z_cont", curtain["Z"])
    d, x, y = curtain["d"], curtain["x"], curtain["y"]

    primero = True
    for nivel in niveles:
        for seg in _segmentos_contorno(D, Z, V, nivel):
            d_seg, z_seg = seg[:, 0], seg[:, 1]
            x_seg = np.interp(d_seg, d, x)
            y_seg = np.interp(d_seg, d, y)
            fig.add_trace(
                go.Scatter3d(
                    x=x_seg,
                    y=y_seg,
                    z=z_seg,
                    mode="lines",
                    line=dict(width=grosor, color=color),
                    name=f"Contorno {nivel:g} — {nombre_linea}",
                    legendgroup=f"contorno_{nivel:g}",
                    showlegend=mostrar_leyenda and primero,
                    hovertemplate=f"Isovalor {nivel:g}<extra></extra>",
                )
            )
            primero = False
    return fig


# ---------------------------------------------------------------------------
# Figura principal
# ---------------------------------------------------------------------------
def plot_3d_variable(
    df: pd.DataFrame,
    var_color: str,
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Elevacion",
    line_col: str | None = "Linea",
    fig: go.Figure | None = None,
    mode: str = "surface",
    cmap: str = "espectro_personalizado",
    point_size: int = 4,
    marker_opacity: float = 1.0,
    connect_points: bool = True,
    grid_res: int = 60,
    profundidad_max: float | None = None,
    surface_opacity: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    name_prefix: str | None = None,
    contornos: Iterable[float] | None = None,
    color_contorno: str = "black",
    grosor_contorno: float = 3.0,
    titulo: str = "Síntesis 3D – Verificación de consistencia",
    subtitulo: str | None = None,
    save_html: str | None = None,
) -> go.Figure:
    """
    Crea (o extiende) el gráfico 3D interactivo a partir de UN DataFrame.

    Para agregar más datasets a la misma figura, se vuelve a llamar pasando
    ``fig=`` la figura devuelta::

        fig = plot_3d_variable(df_vp, var_color='Vp',  line_col='Linea_Vp')
        fig = plot_3d_variable(df_vs, var_color='Vs',  line_col='Linea_Vs', fig=fig)

    Parameters
    ----------
    mode : {'surface', 'scatter'}
        ``'surface'`` genera la cortina vertical continua que sigue el
        trazado y la topografía de cada línea; ``'scatter'`` dibuja los
        nodos.  Si un grupo tiene menos de 4 puntos hace fallback a scatter.
    contornos : iterable de float | None
        Valores de ``var_color`` en los que se dibuja una isolínea negra
        sobre la cortina (sólo en ``mode='surface'``).
    """
    vmin, vmax = _rango_color(df, var_color, fig, vmin, vmax)
    escala = resolver_colorscale(cmap)

    nueva_figura = fig is None
    if fig is None:
        fig = go.Figure()

    show_colorbar = not any(
        (getattr(tr, "marker", None) is not None and getattr(tr.marker, "showscale", False))
        or getattr(tr, "showscale", False)
        for tr in fig.data
        if tr.name != NOMBRE_BASE
    )

    if line_col is not None and line_col in df.columns:
        grupos = sorted(df[line_col].unique())
    else:
        grupos = [None]

    # Un prefijo vacío deja el nombre del grupo tal cual: la leyenda es ahora
    # el único control de visibilidad, así que conviene que diga exactamente
    # el nombre de la línea.
    prefix = name_prefix if name_prefix is not None else (line_col or "Datos")
    primer_contorno = True

    for grupo in grupos:
        df_g = df if grupo is None else df.loc[df[line_col] == grupo]
        nombre = (f"{prefix} {grupo}".strip() if grupo is not None else prefix)

        curtain = None
        usar_surface = mode == "surface"
        if usar_surface:
            curtain = construir_curtain(
                df_g, x_col, y_col, z_col, var_color, grid_res, profundidad_max
            )
            usar_surface = curtain is not None

        if usar_surface:
            fig.add_trace(
                go.Surface(
                    x=curtain["X"],
                    y=curtain["Y"],
                    z=curtain["Z"],
                    surfacecolor=curtain["V"],
                    colorscale=escala,
                    cmin=vmin,
                    cmax=vmax,
                    opacity=surface_opacity,
                    showscale=show_colorbar,
                    colorbar=_colorbar_horizontal(f"{var_color} [m/s]")
                    if show_colorbar
                    else None,
                    name=nombre,
                    hovertemplate=(
                        f"{nombre}<br>X: %{{x:.1f}}<br>Y: %{{y:.1f}}"
                        f"<br>Z: %{{z:.1f}}<br>{var_color}: %{{surfacecolor:.2f}}<extra></extra>"
                    ),
                )
            )
            if contornos:
                fig = agregar_contornos_3d(
                    fig,
                    curtain,
                    contornos,
                    nombre_linea=str(grupo) if grupo is not None else prefix,
                    color=color_contorno,
                    grosor=grosor_contorno,
                    mostrar_leyenda=primer_contorno,
                )
                primer_contorno = False
        else:
            modo_trazo = "lines+markers" if connect_points else "markers"
            fig.add_trace(
                go.Scatter3d(
                    x=df_g[x_col],
                    y=df_g[y_col],
                    z=df_g[z_col],
                    mode=modo_trazo,
                    marker=dict(
                        size=point_size,
                        color=df_g[var_color],
                        colorscale=escala,
                        cmin=vmin,
                        cmax=vmax,
                        opacity=marker_opacity,
                        colorbar=_colorbar_horizontal(f"{var_color} [m/s]")
                    if show_colorbar
                    else None,
                        showscale=show_colorbar,
                    ),
                    line=dict(width=2, color="gray") if connect_points else None,
                    name=nombre,
                    hovertemplate=(
                        f"{nombre}<br>X: %{{x:.1f}}<br>Y: %{{y:.1f}}"
                        f"<br>Z: %{{z:.1f}}<br>{var_color}: %{{marker.color:.2f}}<extra></extra>"
                    ),
                )
            )

        show_colorbar = False

    # --- Layout base tipo dashboard --------------------------------------
    if nueva_figura:
        fig.update_layout(
            **_DASHBOARD_LAYOUT,
            scene=dict(
                xaxis_title=f"{x_col} UTM (m)",
                yaxis_title=f"{y_col} UTM (m)",
                zaxis_title="Elevación (m)",
                **_SCENE_STYLE,
            ),
            title=dict(
                text=f"<b>{titulo}</b>"
                + (f"<br><sup>{subtitulo}</sup>" if subtitulo else ""),
                x=0.02,
                xanchor="left",
                y=0.97,
                yanchor="top",
            ),
        )
        fig.update_layout(margin=dict(l=10, r=10, t=155, b=10))
    elif titulo:
        fig.update_layout(title=dict(text=f"<b>{titulo}</b>"))

    fig = _barra_de_controles(fig)

    if save_html:
        fig.write_html(save_html)

    return fig


def _barra_de_controles(fig: go.Figure) -> go.Figure:
    """
    Cabecera con el único control que queda: la vista de cámara.

    Los menús de visibilidad y de selección de línea se retiraron; mostrar u
    ocultar una línea se hace con un click en la leyenda, que es el
    comportamiento nativo de Plotly y no duplica controles.
    """
    botones_camara = [
        dict(label=nombre, method="relayout", args=[{"scene.camera": cam}])
        for nombre, cam in _CAMARAS.items()
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                buttons=botones_camara,
                x=0.0,
                y=0.885,
                xanchor="left",
                yanchor="top",
                showactive=True,
                bgcolor="#ffffff",
                bordercolor="#d7d9dd",
                borderwidth=1,
                font=dict(size=12, color="#2b2b33"),
                pad=dict(l=10, r=10, t=4, b=4),
            )
        ],
        annotations=[
            dict(
                text="Vista de cámara",
                x=0.0,
                y=0.925,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
                font=dict(size=11, color="#6b7280"),
            )
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Imagen base
# ---------------------------------------------------------------------------
def leer_extension_geotiff(fuente) -> dict:
    """
    Extrae la extensión geográfica de un GeoTIFF.

    Intenta ``rasterio`` y, si no está disponible, la etiqueta EXIF
    GeoTIFF mediante ``Pillow`` (ModelTiepoint + ModelPixelScale).

    Returns
    -------
    dict
        ``{'x_min','x_max','y_min','y_max','crs','ancho','alto','motor'}``

    Raises
    ------
    RuntimeError
        Si no se pudo extraer la georreferencia.
    """
    datos = fuente.read() if hasattr(fuente, "read") else fuente

    # --- rasterio ---------------------------------------------------------
    try:  # pragma: no cover
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(datos) as mem, mem.open() as src:
            b = src.bounds
            return {
                "x_min": float(b.left),
                "x_max": float(b.right),
                "y_min": float(b.bottom),
                "y_max": float(b.top),
                "crs": str(src.crs),
                "ancho": int(src.width),
                "alto": int(src.height),
                "motor": "rasterio",
            }
    except Exception:
        pass

    # --- Pillow + etiquetas GeoTIFF --------------------------------------
    try:
        img = Image.open(io.BytesIO(datos))
        tags = getattr(img, "tag_v2", {})
        tiepoint = tags.get(33922)  # ModelTiepointTag
        escala = tags.get(33550)  # ModelPixelScaleTag
        if tiepoint and escala:
            i, j, _, x0, y0, _ = list(tiepoint)[:6]
            sx, sy = float(escala[0]), float(escala[1])
            ancho, alto = img.size
            x_min = float(x0) - float(i) * sx
            y_max = float(y0) + float(j) * sy
            return {
                "x_min": x_min,
                "x_max": x_min + ancho * sx,
                "y_min": y_max - alto * sy,
                "y_max": y_max,
                "crs": "según etiquetas GeoTIFF (no verificado)",
                "ancho": ancho,
                "alto": alto,
                "motor": "pillow",
            }
    except Exception:
        pass

    raise RuntimeError(
        "No se pudo extraer la georreferencia del GeoTIFF. "
        "Instale 'rasterio' (pip install rasterio) o declare las esquinas "
        "manualmente."
    )


def agregar_base_satelital_3d_color(
    fig: go.Figure,
    fuente_img,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_superficie: float,
    opacidad: float = 0.85,
    max_dim: int = 800,
) -> go.Figure:
    """
    Agrega una imagen a color como plano base de la figura 3D.

    Usa indexación de paleta adaptativa (256 colores) para simular textura
    RGB sobre una ``go.Surface``, igual que el módulo original.
    ``fuente_img`` puede ser una ruta, bytes o un objeto con ``.read()``.
    """
    if hasattr(fuente_img, "read"):
        fuente_img = fuente_img.read()
    if isinstance(fuente_img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(fuente_img)))
    else:
        img = Image.open(fuente_img)
    img = img.convert("RGB")

    ancho, alto = img.size
    if ancho > max_dim:
        img = img.resize((max_dim, max(1, int(alto * (max_dim / ancho)))))
    elif alto > max_dim:
        img = img.resize((max(1, int(ancho * (max_dim / alto))), max_dim))

    img_indexada = img.convert("P", palette=Image.ADAPTIVE, colors=256)
    img_array = np.array(img_indexada)

    paleta = np.array(img_indexada.getpalette()[: 256 * 3]).reshape(-1, 3)
    escala = [[i / 255.0, f"rgb({r},{g},{b})"] for i, (r, g, b) in enumerate(paleta)]

    x_coords = np.linspace(x_min, x_max, img_array.shape[1])
    y_coords = np.linspace(y_max, y_min, img_array.shape[0])
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = np.full(X.shape, float(z_superficie))

    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=img_array,
            colorscale=escala,
            cmin=0,
            cmax=255,
            showscale=False,
            opacity=opacidad,
            hoverinfo="skip",
            name=NOMBRE_BASE,
        )
    )
    return fig


def figura_a_html(fig: go.Figure) -> bytes:
    """Serializa la figura como HTML interactivo autocontenido."""
    return fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
