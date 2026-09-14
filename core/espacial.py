"""
core/espacial.py
================

Asignación espacial entre la topografía y los datos geofísicos, siguiendo
``Combine_Data_Topo`` de ``crear_linea_sintetica.ipynb``:

    merge exacto por ``['Linea', 'Xo']``  →  df_source con
    ``Linea, Xo, Elevacion, Vp, X, Y, Z``

donde ``Z`` es la elevación de la superficie y ``Elevacion`` la coordenada
vertical del nodo de inversión.

Además implementa las dos reglas de emplazamiento que pide el flujo:

* **MASW** — se ancla a la coordenada **central** del tendido TRS.
* **SEV / VES** — se ancla a la **coincidencia exacta** de ``Xo`` en la
  lista de estaciones topográficas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Navegación del catálogo topográfico
# ---------------------------------------------------------------------------
def zonas_disponibles(df_topo: pd.DataFrame) -> list[str]:
    return sorted(df_topo["Zona_Nombre"].dropna().unique().tolist())


def lineas_de_zona(df_topo: pd.DataFrame, zona: str) -> list[str]:
    sub = df_topo.loc[df_topo["Zona_Nombre"] == zona]
    return sorted(sub["Linea"].dropna().unique().tolist())


def topografia_de_linea(df_topo: pd.DataFrame, zona: str, linea: str) -> pd.DataFrame:
    """Estaciones topográficas de una línea, ordenadas por ``Xo``."""
    sub = df_topo.loc[
        (df_topo["Zona_Nombre"] == zona) & (df_topo["Linea"] == linea)
    ].copy()
    return sub.sort_values("Xo").reset_index(drop=True)


def centro_del_tendido(topo_linea: pd.DataFrame) -> dict:
    """
    Coordenada central del tendido: ``Xo`` medio entre el primer y el último
    punto, con ``X``, ``Y`` y ``Z`` interpolados linealmente sobre las
    estaciones reales (no se fuerza a caer en una estaca existente).
    """
    xo = topo_linea["Xo"].to_numpy(float)
    xo_c = float((xo.min() + xo.max()) / 2.0)
    return {
        "Xo": xo_c,
        "X": float(np.interp(xo_c, xo, topo_linea["X"].to_numpy(float))),
        "Y": float(np.interp(xo_c, xo, topo_linea["Y"].to_numpy(float))),
        "Z": float(np.interp(xo_c, xo, topo_linea["Z"].to_numpy(float))),
        "Xo_min": float(xo.min()),
        "Xo_max": float(xo.max()),
        "n_estaciones": int(len(topo_linea)),
        "longitud_m": float(np.sum(np.hypot(np.diff(topo_linea["X"]), np.diff(topo_linea["Y"])))),
    }


def estacion_exacta(topo_linea: pd.DataFrame, xo_objetivo: float) -> pd.Series | None:
    """
    Coincidencia **exacta** de ``Xo`` en la lista de estaciones.
    Es la regla de emplazamiento para SEV / VES.

    Returns
    -------
    pd.Series | None
        La fila de la estación, o ``None`` si no existe esa abscisa.
    """
    coincidencias = topo_linea.loc[
        np.isclose(topo_linea["Xo"].to_numpy(float), float(xo_objetivo))
    ]
    if coincidencias.empty:
        return None
    return coincidencias.iloc[0]


# ---------------------------------------------------------------------------
# Cruce TRS 2D ↔ topografía
# ---------------------------------------------------------------------------
def combinar_trs_con_topografia(
    df_trs: pd.DataFrame,
    topo_linea: pd.DataFrame,
    nombre_linea: str,
    modo: str = "exacto",
    tolerancia: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """
    Cruza el modelo 2D de Vp con la topografía por la posición horizontal.

    Parameters
    ----------
    df_trs : pd.DataFrame
        Columnas ``X`` (= ``Xo`` local del tendido), ``Elevacion``, ``Vp``.
    topo_linea : pd.DataFrame
        Estaciones de la línea (``Xo, X, Y, Z``).
    modo : {'exacto', 'cercano'}
        ``'exacto'`` reproduce el merge del notebook (``how='inner'`` sobre
        ``Xo``): sólo sobreviven las abscisas presentes en ambos archivos.
        ``'cercano'`` asocia cada columna del TRS a la estación más próxima
        dentro de ``tolerancia`` metros, conservando las columnas
        intermedias de la malla de inversión.

    Returns
    -------
    df_source : pd.DataFrame
        ``Linea, Xo, Elevacion, Vp, X, Y, Z, Profundidad``
    info : dict
        Diagnóstico del cruce (filas antes/después, abscisas perdidas).
    """
    inv = df_trs.rename(columns={"X": "Xo"}).copy()
    inv["Xo"] = inv["Xo"].astype(float)
    inv["Linea"] = nombre_linea

    topo = topo_linea[["Xo", "X", "Y", "Z"]].copy()
    topo["Xo"] = topo["Xo"].astype(float)

    n_antes = len(inv)
    xo_inv = np.sort(inv["Xo"].unique())

    if modo == "cercano":
        inv = inv.sort_values("Xo")
        topo_ord = topo.sort_values("Xo")
        df_source = pd.merge_asof(
            inv,
            topo_ord,
            on="Xo",
            direction="nearest",
            tolerance=float(tolerancia),
        )
        df_source = df_source.dropna(subset=["X", "Y", "Z"])
    else:
        df_source = pd.merge(inv, topo, on="Xo", how="inner")

    if df_source.empty:
        raise ValueError(
            "El cruce del modelo 2D con la topografía no produjo resultados. "
            f"Abscisas del TRS: {xo_inv.min():.2f}…{xo_inv.max():.2f}; "
            f"abscisas de la topografía: {topo['Xo'].min():.2f}…{topo['Xo'].max():.2f}. "
            "Revise que 'Xo' sea la misma referencia en ambos archivos "
            "o use el modo 'cercano'."
        )

    df_source["Profundidad"] = df_source["Z"] - df_source["Elevacion"]
    df_source = df_source[
        ["Linea", "Xo", "Elevacion", "Vp", "X", "Y", "Z", "Profundidad"]
    ].reset_index(drop=True)

    xo_conservadas = np.sort(df_source["Xo"].unique())
    info = {
        "modo": modo,
        "filas_antes": n_antes,
        "filas_despues": int(len(df_source)),
        "columnas_trs": int(len(xo_inv)),
        "columnas_cruzadas": int(len(xo_conservadas)),
        "xo_perdidas": sorted(set(np.round(xo_inv, 3)) - set(np.round(xo_conservadas, 3))),
        "xo_min": float(xo_conservadas.min()),
        "xo_max": float(xo_conservadas.max()),
    }
    return df_source, info


# ---------------------------------------------------------------------------
# Emplazamiento de sondeos puntuales
# ---------------------------------------------------------------------------
def emplazar_masw(
    df_masw: pd.DataFrame,
    topo_linea: pd.DataFrame,
    nombre_linea: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Asigna al modelo 1D de Vs la coordenada **central** del tendido TRS y
    convierte profundidad a elevación usando la cota de ese punto.

    Returns
    -------
    df : pd.DataFrame
        ``Linea, Xo, X, Y, Z, Profundidad, Elevacion, Vs``
    centro : dict
        Salida de :func:`centro_del_tendido`.
    """
    centro = centro_del_tendido(topo_linea)

    df = df_masw.copy()
    df["Linea"] = nombre_linea
    df["Xo"] = centro["Xo"]
    df["X"] = centro["X"]
    df["Y"] = centro["Y"]
    df["Z"] = centro["Z"]
    df["Elevacion"] = centro["Z"] - df["Profundidad"]

    cols = ["Linea", "Xo", "X", "Y", "Z", "Profundidad", "Elevacion", "Vs"]
    return df[cols].reset_index(drop=True), centro


def emplazar_sev(
    df_sev: pd.DataFrame,
    topo_linea: pd.DataFrame,
    nombre_linea: str,
    xo_objetivo: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Asigna a un sondeo SEV/VES la estación topográfica cuya ``Xo`` coincide
    **exactamente** con ``xo_objetivo``.

    Raises
    ------
    ValueError
        Si esa abscisa no existe en la lista de estaciones de la línea.
    """
    est = estacion_exacta(topo_linea, xo_objetivo)
    if est is None:
        disponibles = topo_linea["Xo"].tolist()
        raise ValueError(
            f"No hay coincidencia exacta de Xo={xo_objetivo} en la línea "
            f"'{nombre_linea}'. Abscisas disponibles: {disponibles}"
        )

    df = df_sev.copy()
    df["Linea"] = nombre_linea
    df["Xo"] = float(est["Xo"])
    df["X"] = float(est["X"])
    df["Y"] = float(est["Y"])
    df["Z"] = float(est["Z"])
    if "Profundidad" in df.columns:
        df["Elevacion"] = df["Z"] - df["Profundidad"]

    ubicacion = {
        "ID": est.get("ID"),
        "Xo": float(est["Xo"]),
        "X": float(est["X"]),
        "Y": float(est["Y"]),
        "Z": float(est["Z"]),
    }
    return df.reset_index(drop=True), ubicacion


# ---------------------------------------------------------------------------
# Malla objetivo de una línea (real o sintética)
# ---------------------------------------------------------------------------
def malla_objetivo(
    topo_linea: pd.DataFrame,
    prof_max: float = 30.0,
    dz: float = 1.0,
) -> pd.DataFrame:
    """
    Construye la malla vertical de la línea a partir de su topografía,
    idéntica a la de ``synthesize_line``: para cada estación se generan
    nodos desde la cota ``Z`` hasta ``Z − prof_max`` con paso ``dz``.
    """
    filas = []
    for _, fila in topo_linea.iterrows():
        z_sup = float(fila["Z"])
        elevaciones = np.arange(z_sup, z_sup - prof_max - dz, -dz)
        for z_i in elevaciones:
            filas.append(
                [float(fila["Xo"]), float(fila["X"]), float(fila["Y"]), z_sup, float(z_i)]
            )
    df = pd.DataFrame(filas, columns=["Xo", "X", "Y", "Z_superficie", "Elevacion"])
    df["Profundidad"] = df["Z_superficie"] - df["Elevacion"]
    return df
