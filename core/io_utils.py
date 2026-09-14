"""
core/io_utils.py
================

Lectura robusta y cacheable de los formatos de entrada del flujo TRS / MASW.

Formatos soportados
-------------------
TRS 2D (.dat / .txt / .csv)
    Tres columnas: X (distancia a lo largo del tendido, m),
    Y (elevación o cota, m) y Z (Vp).  Es el formato que produce el
    software de inversión de refracción (p.ej. ``Los_cuates_1.dat``)::

        X Y  Z
        -1.30000000000000E+0001  1.20000000000000E+0001  6.92991629919705E-0001

    La Vp puede venir en km/s o en m/s; :func:`cargar_trs_2d` lo detecta y
    normaliza siempre a m/s (el notebook original multiplicaba por 1000).

MASW 1D (.txt / .csv)
    Dos columnas: profundidad (m) y Vs (m/s), con o sin encabezado
    (``Depth, Vs_mean`` en los archivos de ejemplo).

Topografía (.txt / .csv)
    ``Zona_Num, Linea_Num, Zona_Nombre, Linea, ID, Xo, X, Y, Z``

Todas las funciones públicas reciben ``bytes`` o una ruta, de modo que se
pueden envolver con ``@st.cache_data`` sin depender de objetos no
serializables de Streamlit.
"""

from __future__ import annotations

import io
import os
from typing import Union

import numpy as np
import pandas as pd

FuenteArchivo = Union[str, bytes, bytearray, io.BytesIO]

# Nombres canónicos que el resto de la app espera.
COLS_TRS = ["X", "Elevacion", "Vp"]
COLS_MASW = ["Profundidad", "Vs"]
COLS_TOPO = [
    "Zona_Num",
    "Linea_Num",
    "Zona_Nombre",
    "Linea",
    "ID",
    "Xo",
    "X",
    "Y",
    "Z",
]

# Sinónimos aceptados al leer archivos con encabezado.
_ALIAS = {
    "x": "X",
    "dist": "X",
    "distancia": "X",
    "xo": "Xo",
    "y": "Elevacion",
    "z": "Vp",
    "elev": "Elevacion",
    "elevacion": "Elevacion",
    "elevación": "Elevacion",
    "cota": "Elevacion",
    "vp": "Vp",
    "vp_ms": "Vp",
    "depth": "Profundidad",
    "prof": "Profundidad",
    "profundidad": "Profundidad",
    "vs": "Vs",
    "vs_mean": "Vs",
    "vs_promedio": "Vs",
    "vs_masw": "Vs",
}


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _a_buffer(fuente: FuenteArchivo) -> io.BytesIO:
    """Normaliza rutas / bytes a un buffer binario reutilizable."""
    if isinstance(fuente, (bytes, bytearray)):
        return io.BytesIO(bytes(fuente))
    if isinstance(fuente, io.BytesIO):
        fuente.seek(0)
        return fuente
    if isinstance(fuente, str) and os.path.exists(fuente):
        with open(fuente, "rb") as fh:
            return io.BytesIO(fh.read())
    raise FileNotFoundError(f"No se pudo interpretar la fuente de datos: {fuente!r}")


def _tiene_encabezado(primera_linea: str) -> bool:
    """Heurística: si el primer renglón no es completamente numérico, es header."""
    tokens = primera_linea.replace(",", " ").replace(";", " ").replace("\t", " ").split()
    if not tokens:
        return False
    for tok in tokens:
        try:
            float(tok)
        except ValueError:
            return True
    return False


def _leer_tabla(fuente: FuenteArchivo) -> pd.DataFrame:
    """
    Lector genérico tolerante a separadores (coma, punto y coma, tabulador,
    espacios múltiples) y a la presencia o ausencia de encabezado.
    """
    buf = _a_buffer(fuente)
    crudo = buf.getvalue().decode("utf-8", errors="replace")
    lineas = [ln for ln in crudo.splitlines() if ln.strip()]
    if not lineas:
        raise ValueError("El archivo está vacío.")

    header = 0 if _tiene_encabezado(lineas[0]) else None

    # El separador se infiere del contenido: coma/;/tab explícitos, o espacios.
    muestra = lineas[1] if len(lineas) > 1 else lineas[0]
    if muestra.count(",") >= 1:
        sep = ","
    elif muestra.count(";") >= 1:
        sep = ";"
    elif "\t" in muestra:
        sep = "\t"
    else:
        sep = r"\s+"

    df = pd.read_csv(
        io.StringIO(crudo),
        sep=sep,
        header=header,
        engine="python",
        comment="#",
    )
    # Si vino sin encabezado, pandas nombra 0,1,2...; se deja así y el
    # llamador asigna nombres posicionalmente.
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _renombrar_por_alias(df: pd.DataFrame) -> pd.DataFrame:
    nuevos = {}
    for c in df.columns:
        clave = str(c).strip().lower()
        if clave in _ALIAS:
            nuevos[c] = _ALIAS[clave]
    return df.rename(columns=nuevos)


def _es_numerico(col: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(col)


# ---------------------------------------------------------------------------
# TRS 2D
# ---------------------------------------------------------------------------
def cargar_trs_2d(fuente: FuenteArchivo, unidad_vp: str = "auto") -> pd.DataFrame:
    """
    Lee un modelo 2D de Vp de sísmica de refracción.

    Parameters
    ----------
    fuente : str | bytes
        Ruta o contenido del archivo (X, Elevación, Vp).
    unidad_vp : {'auto', 'km/s', 'm/s'}
        ``'auto'`` decide por magnitud: si la mediana de Vp es menor a 30 se
        asume km/s y se multiplica por 1000 (igual que el notebook, que
        aplicaba ``*1000`` a la Vp interpolada).

    Returns
    -------
    pd.DataFrame
        Columnas ``X``, ``Elevacion``, ``Vp`` (Vp siempre en m/s).
    """
    df = _leer_tabla(fuente)
    df = _renombrar_por_alias(df)

    if not set(COLS_TRS).issubset(df.columns):
        # Asignación posicional: las 3 primeras columnas numéricas.
        num = [c for c in df.columns if _es_numerico(df[c])]
        if len(num) < 3:
            raise ValueError(
                "El archivo TRS debe tener al menos 3 columnas numéricas "
                f"(X, Elevación, Vp). Se encontraron: {list(df.columns)}"
            )
        df = df[num[:3]].copy()
        df.columns = COLS_TRS
    else:
        df = df[COLS_TRS].copy()

    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

    mediana = float(np.nanmedian(df["Vp"]))
    if unidad_vp == "km/s" or (unidad_vp == "auto" and mediana < 30.0):
        df["Vp"] = df["Vp"] * 1000.0
        df.attrs["conversion_vp"] = "km/s -> m/s (x1000)"
    else:
        df.attrs["conversion_vp"] = "sin conversión (ya en m/s)"

    df.attrs["n_columnas_x"] = int(df["X"].nunique())
    return df


# ---------------------------------------------------------------------------
# MASW 1D
# ---------------------------------------------------------------------------
def cargar_masw_1d(fuente: FuenteArchivo) -> pd.DataFrame:
    """
    Lee un modelo 1D de Vs (MASW).

    Returns
    -------
    pd.DataFrame
        Columnas ``Profundidad`` (m) y ``Vs`` (m/s), ordenadas por profundidad.
    """
    df = _leer_tabla(fuente)
    df = _renombrar_por_alias(df)

    if not set(COLS_MASW).issubset(df.columns):
        num = [c for c in df.columns if _es_numerico(df[c])]
        if len(num) < 2:
            raise ValueError(
                "El archivo MASW debe tener al menos 2 columnas numéricas "
                f"(Profundidad, Vs). Se encontraron: {list(df.columns)}"
            )
        df = df[num[:2]].copy()
        df.columns = COLS_MASW
    else:
        df = df[COLS_MASW].copy()

    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_values("Profundidad").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Topografía
# ---------------------------------------------------------------------------
def cargar_topografia(fuente: FuenteArchivo) -> pd.DataFrame:
    """
    Lee el archivo de topografía con el esquema
    ``Zona_Num, Linea_Num, Zona_Nombre, Linea, ID, Xo, X, Y, Z``.

    Si el archivo viene sin encabezado se asignan esos nombres
    posicionalmente. Las columnas numéricas se convierten a float y se
    descartan renglones incompletos en ``Xo``, ``X``, ``Y``, ``Z``.
    """
    df = _leer_tabla(fuente)

    # Normaliza encabezados quitando acentos/espacios y comparando en minúsculas.
    canon = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    mapa = {}
    for objetivo in COLS_TOPO:
        clave = objetivo.lower()
        if clave in canon:
            mapa[canon[clave]] = objetivo
    df = df.rename(columns=mapa)

    if not set(COLS_TOPO).issubset(df.columns):
        if df.shape[1] >= len(COLS_TOPO):
            df = df.iloc[:, : len(COLS_TOPO)].copy()
            df.columns = COLS_TOPO
        else:
            raise ValueError(
                "El archivo de topografía debe tener las columnas "
                f"{COLS_TOPO}. Se encontraron: {list(df.columns)}"
            )
    else:
        df = df[COLS_TOPO].copy()

    for c in ["Zona_Num", "Linea_Num", "Xo", "X", "Y", "Z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["Zona_Nombre", "Linea", "ID"]:
        df[c] = df[c].astype(str).str.strip()

    df = df.dropna(subset=["Xo", "X", "Y", "Z"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Modelos 2D ya sintetizados (entrada del Tab 3)
# ---------------------------------------------------------------------------
def cargar_modelo_2d(fuente: FuenteArchivo) -> pd.DataFrame:
    """
    Lee un CSV de perfil 2D ya generado (salida del Tab 2 o los
    ``modelos_2d_vp_vs_*.csv`` de los notebooks). No impone esquema: sólo
    limpia la columna índice sobrante y convierte lo numérico.
    """
    df = _leer_tabla(fuente)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    for c in df.columns:
        if df[c].dtype == object:
            convertida = pd.to_numeric(df[c], errors="coerce")
            if convertida.notna().mean() > 0.95:
                df[c] = convertida
    return df


def resumen_trs(df_trs: pd.DataFrame) -> dict:
    """Metadatos útiles para mostrar en la interfaz."""
    return {
        "n_puntos": int(len(df_trs)),
        "n_columnas_x": int(df_trs["X"].nunique()),
        "x_min": float(df_trs["X"].min()),
        "x_max": float(df_trs["X"].max()),
        "x_centro": float((df_trs["X"].min() + df_trs["X"].max()) / 2.0),
        "elev_min": float(df_trs["Elevacion"].min()),
        "elev_max": float(df_trs["Elevacion"].max()),
        "vp_min": float(df_trs["Vp"].min()),
        "vp_max": float(df_trs["Vp"].max()),
        "conversion_vp": df_trs.attrs.get("conversion_vp", "n/d"),
    }
