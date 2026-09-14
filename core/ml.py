"""
core/ml.py
==========

Clasificación geofísica no supervisada sobre los módulos elásticos,
siguiendo ``clasificacion_geofisica_optimizada`` de
``Interpretacion_MASW_TRS.ipynb``:

    StandardScaler  →  PCA(n_components=0.90)  →  K-Means
    k óptimo por Silhouette Score  →  reordenamiento de clases por Vs medio

Nota sobre una corrección respecto al notebook
----------------------------------------------
El notebook guardaba el mejor *k* como ``mejor_k = k + 1`` dentro del bucle
de búsqueda, de modo que el ``k`` finalmente entrenado **no** era el que
obtuvo el mejor Silhouette (por eso reportaba «3» cuando el máximo se
alcanzó en k=2).  Aquí se corrige (``mejor_k = k``) y se deja el
comportamiento original disponible con ``replicar_notebook=True`` para
poder reproducir resultados históricos.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Atributos físicos del material usados para clasificar.
# Se excluyen Profundidad, Line y las columnas de control matemático.
COLUMNAS_CLASIFICACION = [
    "Vs_MASW_original",
    "Vp_TRS_interpolado",
    "Densidad",
    "Coef_Poisson",
    "Modulo_Corte_G",
    "Modulo_Young_E",
    "Modulo_Bulk_K",
    "Lame_Lambda",
    "Lame_Mu",
]


def clasificacion_geofisica_optimizada(
    df_completo: pd.DataFrame,
    max_clases: int = 8,
    varianza_pca: float = 0.90,
    columnas: list[str] | None = None,
    random_state: int = 42,
    replicar_notebook: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Clasificación no supervisada optimizada con PCA + K-Means.

    Parameters
    ----------
    df_completo : pd.DataFrame
        Salida de :func:`core.fisica.calcular_modulos_elasticos` (puede ser
        la concatenación de varias líneas).
    max_clases : int
        Máximo número de clases a evaluar.  Se acota automáticamente a
        ``n_muestras − 1``.
    varianza_pca : float
        Fracción de varianza explicada a retener (0.90 = 90 %, como pide el
        flujo de trabajo).
    columnas : list[str] | None
        Atributos a usar.  Por defecto :data:`COLUMNAS_CLASIFICACION`.
    replicar_notebook : bool
        Si es True reproduce el ``mejor_k = k + 1`` del notebook original.

    Returns
    -------
    df : pd.DataFrame
        Copia del DataFrame con la columna ``Clase_Geofisica`` (0 = material
        más blando por Vs medio).
    diag : dict
        Diagnóstico: número de componentes, varianza explicada, curva de
        Silhouette, k elegido, centroides y cargas del PCA.
    """
    df = df_completo.copy()

    base = columnas or COLUMNAS_CLASIFICACION
    cols = [c for c in base if c in df.columns]
    if len(cols) < 2:
        raise ValueError(
            "Se requieren al menos 2 atributos para clasificar. "
            f"Disponibles: {cols}"
        )

    X = df[cols].replace([np.inf, -np.inf], np.nan)
    validas = X.notna().all(axis=1)
    if validas.sum() < 3:
        raise ValueError(
            "Muy pocos renglones válidos (sin NaN/inf) para clasificar: "
            f"{int(validas.sum())}."
        )
    X = X.loc[validas]

    # 1. Estandarización (media 0, varianza 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Reducción de dimensionalidad reteniendo la varianza pedida
    pca = PCA(n_components=varianza_pca, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    # 3. Búsqueda del k óptimo por Silhouette
    k_max = int(min(max_clases, len(X_pca) - 1))
    k_max = max(k_max, 2)

    curva = []
    mejor_k, mejor_score = 2, -1.0
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        etiquetas = km.fit_predict(X_pca)
        if len(np.unique(etiquetas)) < 2:
            continue
        score = float(silhouette_score(X_pca, etiquetas))
        curva.append({"k": k, "silhouette": score})
        if score > mejor_score:
            mejor_score = score
            mejor_k = k + 1 if replicar_notebook else k

    mejor_k = int(min(max(mejor_k, 2), len(X_pca) - 1))

    # 4. Clasificación final
    km_opt = KMeans(n_clusters=mejor_k, random_state=random_state, n_init=10)
    etiquetas = km_opt.fit_predict(X_pca)

    df["Clase_Geofisica"] = np.nan
    df.loc[validas, "Clase_Geofisica"] = etiquetas

    # 5. Reordenar: clase 0 = Vs medio más bajo (suelo más blando)
    col_orden = "Vs_MASW_original" if "Vs_MASW_original" in df.columns else cols[0]
    orden = (
        df.dropna(subset=["Clase_Geofisica"])
        .groupby("Clase_Geofisica")[col_orden]
        .mean()
        .sort_values()
        .index
    )
    mapa = {vieja: nueva for nueva, vieja in enumerate(orden)}
    df["Clase_Geofisica"] = df["Clase_Geofisica"].map(mapa)

    # Perfil promedio de cada clase, útil para interpretar litológicamente
    resumen = (
        df.dropna(subset=["Clase_Geofisica"])
        .groupby("Clase_Geofisica")[cols]
        .mean()
        .round(3)
        .reset_index()
    )
    conteo = df["Clase_Geofisica"].value_counts().sort_index()
    resumen["N_muestras"] = resumen["Clase_Geofisica"].map(conteo).astype("Int64")

    cargas = pd.DataFrame(
        pca.components_,
        columns=cols,
        index=[f"PC{i + 1}" for i in range(pca.n_components_)],
    ).round(3)

    diag = {
        "columnas_usadas": cols,
        "n_variables": len(cols),
        "n_componentes": int(pca.n_components_),
        "varianza_explicada": pca.explained_variance_ratio_.tolist(),
        "varianza_acumulada": float(pca.explained_variance_ratio_.sum()),
        "curva_silhouette": pd.DataFrame(curva),
        "k_optimo": int(mejor_k),
        "mejor_silhouette": float(mejor_score),
        "resumen_clases": resumen,
        "cargas_pca": cargas,
        "scores_pca": pd.DataFrame(
            X_pca, columns=[f"PC{i + 1}" for i in range(X_pca.shape[1])]
        ),
        "replicar_notebook": replicar_notebook,
    }
    return df, diag
