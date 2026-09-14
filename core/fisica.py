"""
core/fisica.py
==============

Procesamiento físico del flujo TRS (Vp 2D) + MASW (Vs 1D):

1. **Corrección topográfica** — cada traza vertical del modelo 2D se
   referencia a su propia superficie, de modo que todo el perfil queda en
   profundidad relativa con ``z = 0`` en el terreno.
2. **Colapso 1D por promedio ponderado inverso a la distancia (IDW)** al
   centro del tendido.  Sustituye a la extracción de una sola columna
   central que hacía ``procesar_TRS_a_MASW`` en
   ``Interpretacion_MASW_TRS.ipynb``: aquí *todas* las trazas aportan, con
   peso ``w = 1 / (|x - x_centro|^p + eps)``.
3. **Interpolación** del perfil 1D de Vp a las profundidades exactas del
   modelo de Vs (``interp1d`` lineal con extrapolación, igual que el
   notebook).
4. **Módulos elásticos** — se conservan *literalmente* las ecuaciones de
   ``calcular_modulos_elasticos`` del notebook, incluidos los factores de
   escala ``/1e5`` y la densidad de Gardner modificada.

Ecuaciones implementadas
------------------------
Densidad (t/m³, Gardner modificada del notebook):

    rho = 1.2475 + 0.399·(Vp/1000) − 0.026·(Vp/1000)²

Con ``r = Vp/Vs``:

    ν  = (r² − 2) / (2·(r² − 1))            Coeficiente de Poisson
    G  = rho·Vs² / 1e5                      Módulo de corte (= μ, 2ª de Lamé)
    E  = 2·G·(1 + ν)                        Módulo de Young
    K  = E / (3·(1 − 2ν))                   Módulo de Bulk
    λ  = (rho·Vp² − 2·rho·Vs²) / 1e5        1ª constante de Lamé
    fp = ν / ((1 + ν)·(1 − 2ν))             Factor de proporcionalidad

    Check_Vp     = sqrt((λ + 2G)/rho)       Comprobación estructural
    Check_Lambda = E · fp                   Comprobación λ = E·fp
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Columnas de módulos elásticos que genera este módulo, en orden de interés.
MODULOS_ELASTICOS = [
    "Coef_Poisson",
    "Modulo_Corte_G",
    "Modulo_Young_E",
    "Modulo_Bulk_K",
    "Lame_Lambda",
    "Lame_Mu",
    "Fact_Prop",
    "Densidad",
    "Check_Vp",
    "Check_Lambda",
]

# Etiquetas legibles y unidades para la interfaz.
ETIQUETAS_MODULOS = {
    "Coef_Poisson": ("Coeficiente de Poisson (ν)", "adimensional"),
    "Modulo_Corte_G": ("Módulo de Corte (G)", "×10⁵ unidades"),
    "Modulo_Young_E": ("Módulo de Young (E)", "×10⁵ unidades"),
    "Modulo_Bulk_K": ("Módulo de Bulk (K)", "×10⁵ unidades"),
    "Lame_Lambda": ("1ª constante de Lamé (λ)", "×10⁵ unidades"),
    "Lame_Mu": ("2ª constante de Lamé (μ)", "×10⁵ unidades"),
    "Fact_Prop": ("Factor de proporcionalidad", "adimensional"),
    "Densidad": ("Densidad (ρ)", "t/m³"),
    "Check_Vp": ("Comprobación Vp", "—"),
    "Check_Lambda": ("Comprobación λ", "—"),
}


# ---------------------------------------------------------------------------
# 1. Corrección topográfica
# ---------------------------------------------------------------------------
def corregir_topografia(
    df_trs: pd.DataFrame,
    modo: str = "por_columna",
) -> pd.DataFrame:
    """
    Lleva el modelo 2D a profundidad relativa (z = 0 en el terreno).

    Parameters
    ----------
    df_trs : pd.DataFrame
        Columnas ``X``, ``Elevacion``, ``Vp``.
    modo : {'por_columna', 'global'}
        ``'por_columna'`` (recomendado): cada traza vertical se referencia a
        su propia elevación máxima, es decir a la topografía real en esa
        abscisa.  Es lo que "aplana" el relieve.
        ``'global'``: se resta la elevación máxima de todo el tendido; el
        relieve se conserva como profundidad aparente.

    Returns
    -------
    pd.DataFrame
        Copia con las columnas añadidas ``Z_superficie`` (elevación del
        terreno usada como referencia) y ``Profundidad`` (positiva hacia
        abajo).
    """
    df = df_trs.copy()

    if modo == "global":
        z_sup = float(df["Elevacion"].max())
        df["Z_superficie"] = z_sup
    else:
        superficie = df.groupby("X")["Elevacion"].max()
        df["Z_superficie"] = df["X"].map(superficie)

    df["Profundidad"] = df["Z_superficie"] - df["Elevacion"]
    return df


# ---------------------------------------------------------------------------
# 2. Colapso 1D por promedio ponderado inverso a la distancia
# ---------------------------------------------------------------------------
def perfil_1d_idw(
    df_trs: pd.DataFrame,
    potencia: float = 2.0,
    radio: float | None = None,
    n_nodos: int = 200,
    prof_max: float | None = None,
    modo_correccion: str = "por_columna",
    suavizado: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Colapsa el modelo 2D de Vp a un perfil 1D mediante promedio ponderado
    inverso a la distancia horizontal respecto al centro del tendido.

    El procedimiento es:

    1. Corrección topográfica (:func:`corregir_topografia`).
    2. Malla común de profundidad relativa ``0 … prof_max``.
    3. Cada columna ``X`` se interpola linealmente a esa malla **sin
       extrapolar**: donde una traza no tiene datos no aporta peso, lo que
       evita inventar velocidad al fondo de las trazas cortas.
    4. Promedio ponderado con ``w = 1 / (|x − x_centro| + s)^potencia``.

    El término de suavizado ``s`` es lo que mantiene esto como un
    *promedio*: sin él, una columna que cayera exactamente sobre el centro
    del tendido tendría peso infinito y el resultado colapsaría a esa sola
    traza (que es justamente lo que hacía el notebook original).

    Parameters
    ----------
    potencia : float
        Exponente del IDW.  0 ⇒ promedio aritmético simple; valores altos
        ⇒ el resultado tiende a la columna central (el comportamiento del
        notebook original).
    radio : float | None
        Si se indica, sólo participan las columnas con
        ``|x − x_centro| <= radio``.
    suavizado : float | None
        Distancia de suavizado ``s`` en metros.  Por defecto, la mediana
        del espaciamiento entre columnas de la inversión.
    n_nodos : int
        Número de nodos de la malla común de profundidad.
    prof_max : float | None
        Profundidad máxima del perfil.  Por defecto se usa la **mediana**
        de las profundidades máximas por columna, un compromiso entre
        conservar penetración y no quedarse con una sola traza al fondo.
    modo_correccion : str
        Se pasa a :func:`corregir_topografia`.

    Returns
    -------
    perfil : pd.DataFrame
        ``Profundidad``, ``Vp``, ``N_trazas`` (cuántas columnas aportaron a
        ese nodo) y ``Peso_total``.
    info : dict
        Diagnóstico del cálculo (centro, número de columnas, pesos, etc.).
    """
    df = corregir_topografia(df_trs, modo=modo_correccion)

    x_min, x_max = float(df["X"].min()), float(df["X"].max())
    x_centro = (x_min + x_max) / 2.0

    columnas = sorted(df["X"].unique())
    distancias = np.abs(np.array(columnas) - x_centro)

    if radio is not None:
        mascara = distancias <= radio
        if not mascara.any():
            mascara = distancias == distancias.min()
        columnas = list(np.array(columnas)[mascara])
        distancias = distancias[mascara]

    # Profundidad máxima por columna -> malla común
    prof_max_por_col = df.groupby("X")["Profundidad"].max()
    prof_max_por_col = prof_max_por_col.loc[columnas]
    if prof_max is None:
        prof_max = float(np.nanmedian(prof_max_por_col.values))
    prof_max = max(float(prof_max), 1e-3)

    malla = np.linspace(0.0, prof_max, int(n_nodos))

    if suavizado is None:
        espaciamientos = np.diff(np.sort(np.unique(df["X"].to_numpy(float))))
        suavizado = float(np.median(espaciamientos)) if len(espaciamientos) else 1.0
    suavizado = max(float(suavizado), 1e-9)

    pesos = 1.0 / (distancias + suavizado) ** potencia

    acum_valor = np.zeros_like(malla)
    acum_peso = np.zeros_like(malla)
    n_trazas = np.zeros_like(malla)

    for x_col, w in zip(columnas, pesos):
        sub = df.loc[df["X"] == x_col, ["Profundidad", "Vp"]]
        sub = sub.groupby("Profundidad", as_index=False)["Vp"].mean()
        sub = sub.sort_values("Profundidad")
        if len(sub) < 2:
            continue

        vp_col = np.interp(
            malla,
            sub["Profundidad"].to_numpy(),
            sub["Vp"].to_numpy(),
            left=np.nan,
            right=np.nan,
        )
        valido = ~np.isnan(vp_col)
        acum_valor[valido] += w * vp_col[valido]
        acum_peso[valido] += w
        n_trazas[valido] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        vp_1d = np.where(acum_peso > 0, acum_valor / acum_peso, np.nan)

    perfil = pd.DataFrame(
        {
            "Profundidad": malla,
            "Vp": vp_1d,
            "N_trazas": n_trazas.astype(int),
            "Peso_total": acum_peso,
        }
    ).dropna(subset=["Vp"]).reset_index(drop=True)

    info = {
        "x_min": x_min,
        "x_max": x_max,
        "x_centro": x_centro,
        "n_columnas_totales": int(df["X"].nunique()),
        "n_columnas_usadas": len(columnas),
        "potencia_idw": potencia,
        "radio": radio,
        "suavizado": suavizado,
        "prof_max": prof_max,
        "modo_correccion": modo_correccion,
        "peso_max": float(pesos.max()) if len(pesos) else np.nan,
        "peso_min": float(pesos.min()) if len(pesos) else np.nan,
        "frac_peso_columna_central": (
            float(pesos.max() / pesos.sum()) if len(pesos) else np.nan
        ),
    }
    return perfil, info


# ---------------------------------------------------------------------------
# 3. Interpolación a las profundidades del MASW
# ---------------------------------------------------------------------------
def interpolar_vp_a_masw(
    perfil_vp_1d: pd.DataFrame,
    df_masw: pd.DataFrame,
    extrapolar: bool = True,
) -> pd.DataFrame:
    """
    Lleva el perfil 1D de Vp a las profundidades exactas del modelo de Vs.

    Replica la lógica de ``procesar_TRS_a_MASW``: ``interp1d`` lineal,
    ``bounds_error=False`` y ``fill_value='extrapolate'``.  Se eliminan
    profundidades duplicadas (redondeadas a 4 decimales) antes de construir
    el interpolador, tal como en el notebook.

    Returns
    -------
    pd.DataFrame
        ``Profundidad``, ``Vs_MASW_original``, ``Vp_TRS_interpolado`` y
        ``Fuera_de_rango`` (True donde se extrapoló).
    """
    prof_src = perfil_vp_1d["Profundidad"].to_numpy()
    vp_src = perfil_vp_1d["Vp"].to_numpy()

    prof_unicas, idx = np.unique(np.round(prof_src, 4), return_index=True)
    vp_unicas = vp_src[idx]

    fill = "extrapolate" if extrapolar else np.nan
    f_interp = interp1d(
        prof_unicas,
        vp_unicas,
        kind="linear",
        bounds_error=False,
        fill_value=fill,
    )

    prof_masw = df_masw["Profundidad"].to_numpy()
    vp_en_masw = f_interp(prof_masw)

    fuera = (prof_masw < prof_unicas.min()) | (prof_masw > prof_unicas.max())

    return pd.DataFrame(
        {
            "Profundidad": prof_masw,
            "Vs_MASW_original": df_masw["Vs"].to_numpy(),
            "Vp_TRS_interpolado": vp_en_masw,
            "Fuera_de_rango": fuera,
        }
    )


# ---------------------------------------------------------------------------
# 4. Módulos elásticos
# ---------------------------------------------------------------------------
def calcular_modulos_elasticos(
    df_masw: pd.DataFrame,
    densidad_fija: float | None = None,
) -> pd.DataFrame:
    """
    Calcula los parámetros elásticos a partir de ``Vp_TRS_interpolado`` y
    ``Vs_MASW_original``.

    Se conservan **exactamente** las ecuaciones y los factores de escala del
    notebook ``Interpretacion_MASW_TRS.ipynb``.

    Parameters
    ----------
    densidad_fija : float | None
        Si se indica, se usa ese valor constante de densidad en lugar de la
        relación de Gardner modificada.  El notebook dejaba comentada esa
        alternativa (``df['Densidad'] = densidad_estimada``).

    Raises
    ------
    ValueError
        Si faltan las columnas de velocidad requeridas.
    """
    df = df_masw.copy()

    requeridas = {"Vp_TRS_interpolado", "Vs_MASW_original"}
    if not requeridas.issubset(df.columns):
        raise ValueError(
            "El DataFrame debe contener las columnas "
            "'Vp_TRS_interpolado' y 'Vs_MASW_original'."
        )

    vp = df["Vp_TRS_interpolado"]
    vs = df["Vs_MASW_original"]

    if "Densidad" not in df.columns:
        if densidad_fija is not None:
            df["Densidad"] = float(densidad_fija)
        else:
            df["Densidad"] = np.round(
                1.2475 + (0.399 * vp / 1000) - (0.026 * (vp / 1000) ** 2), 3
            )
    rho = df["Densidad"]

    # Relación Vp/Vs
    r = vp / vs

    # Coeficiente de Poisson
    nu = (r**2 - 2) / (2 * (r**2 - 1))

    # Módulo de corte (G = mu)
    G = (rho * (vs**2)) / 100000

    # Módulo de Young
    E = 2 * G * (1 + nu)

    # Módulo de Bulk
    K = (E / (1 - 2 * nu)) / 3

    # Constantes de Lamé
    lame_lambda = ((rho * (vp**2)) - (2 * rho * (vs**2))) / 100000
    lame_mu = (rho * (vs**2)) / 100000

    # Factor de proporcionalidad
    fact_prop = nu / ((1 + nu) * (1 - 2 * nu))

    # Comprobaciones estructurales (se mantiene la escala del notebook)
    vp_check = np.sqrt((lame_lambda + 2 * G) / rho)
    lambda_check = E * fact_prop

    df["Relacion_Vp_Vs"] = r
    df["Coef_Poisson"] = nu
    df["Modulo_Corte_G"] = G
    df["Modulo_Young_E"] = E
    df["Modulo_Bulk_K"] = K
    df["Lame_Lambda"] = lame_lambda
    df["Lame_Mu"] = lame_mu
    df["Fact_Prop"] = fact_prop
    df["Check_Vp"] = vp_check
    df["Check_Lambda"] = lambda_check

    return df


# ---------------------------------------------------------------------------
# Orquestador del Tab 1
# ---------------------------------------------------------------------------
def integrar_trs_masw(
    df_trs: pd.DataFrame,
    df_masw: pd.DataFrame,
    potencia_idw: float = 2.0,
    radio: float | None = None,
    n_nodos: int = 200,
    prof_max: float | None = None,
    modo_correccion: str = "por_columna",
    densidad_fija: float | None = None,
    extrapolar: bool = True,
    suavizado: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Cadena completa Vp 2D + Vs 1D → módulos elásticos.

    Returns
    -------
    df_resultado : pd.DataFrame
        Una fila por profundidad del MASW, con Vs, Vp interpolada, densidad
        y todos los módulos elásticos.
    perfil_vp_1d : pd.DataFrame
        Perfil 1D de Vp ponderado (antes de interpolar al MASW).
    info : dict
        Diagnóstico del colapso IDW.
    """
    perfil_vp_1d, info = perfil_1d_idw(
        df_trs,
        potencia=potencia_idw,
        radio=radio,
        n_nodos=n_nodos,
        prof_max=prof_max,
        modo_correccion=modo_correccion,
        suavizado=suavizado,
    )
    df_int = interpolar_vp_a_masw(perfil_vp_1d, df_masw, extrapolar=extrapolar)
    df_res = calcular_modulos_elasticos(df_int, densidad_fija=densidad_fija)
    return df_res, perfil_vp_1d, info
