"""
core/prediccion.py
==================

Proyección de Vs (MASW) sobre mallas 2D de Vp (TRS) — **port fiel de
``vs_projection_pipeline.py`` (VERSIÓN 2)**, adaptado para ejecutarse
dentro de Streamlit: recibe y devuelve DataFrames en memoria, expone cada
etapa por separado para poder cachearla, y reporta los diagnósticos en la
interfaz en vez de sólo por ``logging``.

Se conservan sin cambios el esquema de atributos, la validación cruzada
espacial, la optimización de hiperparámetros, el Regression Kriging, el
recorte físico y los tres diagnósticos de confiabilidad:

* extrapolación **espacial** (X, Y) vs. **petrofísica** (Vp, Elevación),
* zona de confianza continua por distancia al MASW más cercano,
* importancia de variables como detector de sobreajuste espacial.

Además se incluye :func:`sintetizar_linea_kriging`, la traducción de
``synthesize_line`` de ``crear_linea_sintetica.ipynb`` (kriging ordinario
3D con anisotropía vertical) para generar líneas nuevas o sintéticas.

REGLA ARQUITECTÓNICA heredada del módulo original: en este archivo no se
usa ``elif``; toda evaluación múltiple se escribe como ``else:`` con un
``if:`` anidado.
"""

from __future__ import annotations

import io
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .kriging import (
    PYKRIGE_DISPONIBLE,
    kriging_ordinario_2d,
    kriging_ordinario_3d,
    parametros_por_defecto,
)

# ---------------------------------------------------------------------------
# Dependencias opcionales
# ---------------------------------------------------------------------------
OPTUNA_DISPONIBLE = False
try:
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_DISPONIBLE = True
except ImportError:  # pragma: no cover
    OPTUNA_DISPONIBLE = False

logger = logging.getLogger("VsProjectionPipeline")

MODELOS_DISPONIBLES = ["ridge", "gradient_boosting", "random_forest"]


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Configuración centralizada del pipeline de proyección de Vs."""

    feature_cols: List[str] = field(
        default_factory=lambda: ["X", "Y", "Elevacion", "Vp"]
    )
    target_col: str = "Vs"
    line_col: str = "Linea"
    coord_cols: Tuple[str, str] = ("X", "Y")
    petrofisicas_cols: Tuple[str, str] = ("Vp", "Elevacion")

    use_engineered_features: bool = True

    # "ridge" es la opción recomendada cuando buena parte de la malla 2D cae
    # fuera del footprint espacial de los MASW, porque SÍ extrapola.
    model_type: str = "ridge"
    random_state: int = 42

    spatial_cv_strategy: str = "line"  # "line" | "block"
    n_blocks: int = 5
    min_group_warning: int = 5

    n_trials: int = 15

    apply_residual_kriging: bool = True
    variogram_model: str = "spherical"

    flag_extrapolation: bool = True
    compute_spatial_confidence: bool = True

    clip_predictions: bool = True
    clip_margin_pct: float = 0.15


# ---------------------------------------------------------------------------
# 1. Validación de datos
# ---------------------------------------------------------------------------
def validar_entrenamiento(df: pd.DataFrame, config: PipelineConfig) -> Tuple[pd.DataFrame, List[str]]:
    """
    Valida el dataset de entrenamiento (perfiles 1D consolidados) y devuelve
    las advertencias de confiabilidad para mostrarlas en la interfaz.
    """
    avisos: List[str] = []
    df = df.copy()

    requeridas = set(config.feature_cols) | {config.target_col, config.line_col}
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Dataset de entrenamiento: faltan columnas requeridas: {sorted(faltantes)}"
        )

    n_antes = len(df)
    df = df.dropna(subset=list(requeridas)).reset_index(drop=True)
    if n_antes != len(df):
        avisos.append(
            f"Se eliminaron {n_antes - len(df)} filas con valores nulos en columnas "
            f"clave ({n_antes} → {len(df)})."
        )

    n_lineas = df[config.line_col].nunique()

    if len(df) < 100:
        avisos.append(
            f"**Muestra pequeña**: sólo {len(df)} puntos de entrenamiento. Las "
            "métricas de validación cruzada tendrán alta varianza y deben leerse "
            "como un intervalo orientativo, no como un número exacto."
        )
    if n_lineas < config.min_group_warning:
        avisos.append(
            f"Sólo {n_lineas} líneas/grupos espaciales para GroupKFold. Con tan "
            "pocos grupos el RMSE/MAE es un estimador de **alta varianza** (cada "
            "pliegue = una línea completa). Opciones: más MASW, estrategia "
            "'block' con más subdivisiones, o tratar la métrica como orientativa."
        )

    return df, avisos


def validar_malla(df: pd.DataFrame, config: PipelineConfig) -> Tuple[pd.DataFrame, List[str]]:
    """Valida el dataset de predicción (nodos de los perfiles 2D de Vp)."""
    avisos: List[str] = []
    df = df.copy()

    faltantes = set(config.feature_cols) - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Dataset de predicción: faltan columnas requeridas: {sorted(faltantes)}"
        )

    n_antes = len(df)
    df = df.dropna(subset=config.feature_cols).reset_index(drop=True)
    if n_antes != len(df):
        avisos.append(
            f"Predicción: se eliminaron {n_antes - len(df)} nodos con "
            "Vp/Elevación/X/Y nulos."
        )
    return df, avisos


# ---------------------------------------------------------------------------
# 2. Sanity check geofísico: correlación Vp–Vs
# ---------------------------------------------------------------------------
def check_vp_vs_correlation(df: pd.DataFrame, config: PipelineConfig) -> Tuple[float, Optional[str]]:
    """
    Correlación de Pearson Vp–Vs en el entrenamiento.

    Una correlación baja (< 0.5) avisa temprano de que Vp por sí sola no
    explica bien la variabilidad de Vs y que el modelo se apoyará más en
    X/Y/Elevación — más riesgoso al extrapolar.
    """
    corr = float(df[["Vp", config.target_col]].corr().iloc[0, 1])

    aviso = None
    if abs(corr) < 0.5:
        aviso = (
            f"Correlación Vp–Vs relativamente baja ({corr:.3f}). El modelo dependerá "
            "más de la posición (X/Y/Elevación) que de la física Vp→Vs, lo cual "
            "reduce la confiabilidad de la extrapolación espacial."
        )
    return corr, aviso


# ---------------------------------------------------------------------------
# 3. Ingeniería de atributos
# ---------------------------------------------------------------------------
def engineer_features(
    df: pd.DataFrame,
    config: PipelineConfig,
    centroid: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, Tuple[float, float]]:
    """
    Agrega los atributos derivados de forma **idéntica** para entrenamiento y
    predicción.  El centroide se calcula una sola vez (con el entrenamiento)
    y se reutiliza en la malla: es lo que hace comparable ``dist_centroide``
    entre ambos conjuntos.
    """
    df = df.copy()

    if not config.use_engineered_features:
        return df, centroid if centroid is not None else (np.nan, np.nan)

    x_col, y_col = config.coord_cols

    if centroid is None:
        centroid = (float(df[x_col].mean()), float(df[y_col].mean()))

    x0, y0 = centroid

    df["Vp_x_Elevacion"] = df["Vp"] * df["Elevacion"]
    df["dist_centroide"] = np.sqrt((df[x_col] - x0) ** 2 + (df[y_col] - y0) ** 2)

    return df, centroid


def get_full_feature_list(config: PipelineConfig) -> List[str]:
    """Lista final de atributos: base + derivados."""
    features = list(config.feature_cols)

    if config.use_engineered_features:
        features = features + ["Vp_x_Elevacion", "dist_centroide"]

    return features


# ---------------------------------------------------------------------------
# 4. Grupos espaciales para validación cruzada
# ---------------------------------------------------------------------------
def create_spatial_groups(df: pd.DataFrame, config: PipelineConfig) -> np.ndarray:
    """Etiqueta de grupo espacial usada por ``GroupKFold``."""
    groups = None

    if config.spatial_cv_strategy == "line":
        groups = df[config.line_col].astype(str).values
    else:
        if config.spatial_cv_strategy == "block":
            x_col, y_col = config.coord_cols
            coords = df[[x_col, y_col]].values
            kmeans = KMeans(
                n_clusters=min(config.n_blocks, len(df)),
                random_state=config.random_state,
                n_init=10,
            )
            groups = kmeans.fit_predict(coords)
        else:
            raise ValueError(
                f"spatial_cv_strategy '{config.spatial_cv_strategy}' no soportada. "
                "Use 'line' o 'block'."
            )

    return groups


def get_n_splits(groups: np.ndarray) -> int:
    """Número de pliegues seguro para GroupKFold (máximo 10, mínimo 2)."""
    n_grupos = len(np.unique(groups))
    return int(min(max(n_grupos, 2), 10))


# ---------------------------------------------------------------------------
# 5. Construcción de modelos
# ---------------------------------------------------------------------------
def build_model(model_type: str, params: Dict[str, Any], random_state: int) -> BaseEstimator:
    """Instancia el estimador según ``model_type``."""
    model = None

    if model_type == "random_forest":
        model = RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    else:
        if model_type == "gradient_boosting":
            model = GradientBoostingRegressor(random_state=random_state, **params)
        else:
            if model_type == "ridge":
                model = Ridge(random_state=random_state, **params)
            else:
                raise ValueError(f"model_type '{model_type}' no soportado.")

    return model


def model_supports_extrapolation(model_type: str) -> bool:
    """
    True si el modelo extrapola de forma continua fuera del rango de
    entrenamiento (lineales); False si se aplana en la frontera de los datos
    vistos (árboles: RF y GBM).
    """
    extrapola = False

    if model_type == "ridge":
        extrapola = True
    else:
        extrapola = False

    return extrapola


def _suggest_params(trial: "optuna.Trial", model_type: str) -> Dict[str, Any]:
    """Espacio de búsqueda de hiperparámetros por tipo de modelo."""
    params: Dict[str, Any] = {}

    if model_type == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
    else:
        if model_type == "gradient_boosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
        else:
            if model_type == "ridge":
                params = {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}
            else:
                raise ValueError(f"model_type '{model_type}' no soportado.")

    return params


def _default_param_grid(model_type: str) -> Dict[str, List[Any]]:
    """Grid manual de respaldo si Optuna no está disponible."""
    grid: Dict[str, List[Any]] = {}

    if model_type == "random_forest":
        grid = {
            "n_estimators": [200, 400],
            "max_depth": [6, 12, None],
            "min_samples_leaf": [1, 3, 5],
        }
    else:
        if model_type == "gradient_boosting":
            grid = {
                "n_estimators": [200, 400],
                "max_depth": [2, 3, 4],
                "learning_rate": [0.05, 0.1],
            }
        else:
            if model_type == "ridge":
                grid = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 50.0]}
            else:
                raise ValueError(f"model_type '{model_type}' no soportado.")

    return grid


# ---------------------------------------------------------------------------
# 6. Evaluación espacial
# ---------------------------------------------------------------------------
def _spatial_cv_rmse(
    model_type: str,
    params: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    random_state: int,
) -> float:
    """RMSE promedio en validación cruzada espacial (GroupKFold)."""
    gkf = GroupKFold(n_splits=get_n_splits(groups))

    rmses = []
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        model = build_model(model_type, params, random_state)
        model.fit(X_train, y[train_idx])

        rmses.append(float(np.sqrt(mean_squared_error(y[val_idx], model.predict(X_val)))))

    return float(np.mean(rmses))


# ---------------------------------------------------------------------------
# 7. Optimización de hiperparámetros
# ---------------------------------------------------------------------------
def optimize_hyperparameters(
    df: pd.DataFrame,
    config: PipelineConfig,
    feature_list: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Minimiza el RMSE de validación cruzada espacial.  Usa Optuna (TPE) si
    está instalado; si no, recorre el grid manual de respaldo.

    Returns
    -------
    best_params, diagnostico
    """
    X = df[feature_list].values
    y = df[config.target_col].values
    groups = create_spatial_groups(df, config)

    best_params: Dict[str, Any] = {}
    diagnostico: Dict[str, Any] = {"motor": None, "mejor_rmse": np.nan, "historial": None}

    if OPTUNA_DISPONIBLE:

        def objective(trial: "optuna.Trial") -> float:
            params = _suggest_params(trial, config.model_type)
            return _spatial_cv_rmse(
                config.model_type, params, X, y, groups, config.random_state
            )

        study = optuna.create_study(
            direction="minimize", sampler=TPESampler(seed=config.random_state)
        )
        study.optimize(objective, n_trials=config.n_trials, show_progress_bar=False)

        best_params = study.best_params
        diagnostico["motor"] = "optuna (TPE)"
        diagnostico["mejor_rmse"] = float(study.best_value)
        diagnostico["historial"] = pd.DataFrame(
            [
                {"trial": t.number, "rmse_cv": t.value, **t.params}
                for t in study.trials
                if t.value is not None
            ]
        )
    else:
        grid = _default_param_grid(config.model_type)
        keys = list(grid.keys())
        combinaciones = list(itertools.product(*grid.values()))

        mejor_rmse = np.inf
        historial = []
        for combo in combinaciones:
            params = dict(zip(keys, combo))
            rmse = _spatial_cv_rmse(
                config.model_type, params, X, y, groups, config.random_state
            )
            historial.append({"rmse_cv": rmse, **params})
            if rmse < mejor_rmse:
                mejor_rmse = rmse
                best_params = params

        diagnostico["motor"] = "grid manual (Optuna no disponible)"
        diagnostico["mejor_rmse"] = float(mejor_rmse)
        diagnostico["historial"] = pd.DataFrame(historial)

    diagnostico["n_grupos"] = int(len(np.unique(groups)))
    diagnostico["n_splits"] = get_n_splits(groups)
    return best_params, diagnostico


# ---------------------------------------------------------------------------
# 8. Entrenamiento final y evaluación honesta
# ---------------------------------------------------------------------------
def train_final_model(
    df: pd.DataFrame,
    config: PipelineConfig,
    feature_list: List[str],
    best_params: Dict[str, Any],
) -> Tuple[BaseEstimator, StandardScaler]:
    """Entrena el modelo final sobre el 100 % del dataset de entrenamiento."""
    X = df[feature_list].values
    y = df[config.target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = build_model(config.model_type, best_params, config.random_state)
    model.fit(X_scaled, y)

    return model, scaler


def evaluate_model_spatial(
    df: pd.DataFrame,
    config: PipelineConfig,
    feature_list: List[str],
    best_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Predicciones *out-of-fold* por validación cruzada espacial: cada línea se
    predice con un modelo que **nunca la vio**.  De ahí salen tanto las
    métricas honestas como los residuos que alimentan el Regression Kriging.
    """
    X = df[feature_list].values
    y = df[config.target_col].values
    groups = create_spatial_groups(df, config)
    n_splits = get_n_splits(groups)
    gkf = GroupKFold(n_splits=n_splits)

    y_pred_oof = np.full(len(df), np.nan)
    por_grupo = []

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        model = build_model(config.model_type, best_params, config.random_state)
        model.fit(X_train, y[train_idx])

        y_pred_oof[val_idx] = model.predict(X_val)
        por_grupo.append(
            {
                "grupo": ", ".join(sorted(set(np.asarray(groups)[val_idx].astype(str)))),
                "n": int(len(val_idx)),
                "RMSE": float(np.sqrt(mean_squared_error(y[val_idx], y_pred_oof[val_idx]))),
                "MAE": float(mean_absolute_error(y[val_idx], y_pred_oof[val_idx])),
            }
        )

    residuos = y - y_pred_oof
    y_std = float(np.std(y))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred_oof)))

    metrics = {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y, y_pred_oof)),
        "r2": float(r2_score(y, y_pred_oof)),
        "rmse_sobre_std_objetivo": rmse / y_std if y_std > 0 else np.nan,
        "n_muestras": int(len(df)),
        "n_splits_espaciales": n_splits,
    }

    return {
        "metrics": metrics,
        "y_true": y,
        "y_pred_oof": y_pred_oof,
        "residuos": residuos,
        "por_grupo": pd.DataFrame(por_grupo),
    }


def calificar_rmse_relativo(valor: float) -> str:
    """Lectura cualitativa de RMSE/STD(Vs): <0.5 bueno, 0.5–0.8 aceptable, >0.8 débil."""
    etiqueta = "débil"

    if valor < 0.5:
        etiqueta = "bueno"
    else:
        if valor <= 0.8:
            etiqueta = "aceptable"
        else:
            etiqueta = "débil"

    return etiqueta


# ---------------------------------------------------------------------------
# 9. Kriging de residuos (Regression Kriging)
# ---------------------------------------------------------------------------
def krige_residuals(
    train_df: pd.DataFrame,
    residuos: np.ndarray,
    grid_df: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[np.ndarray, str]:
    """
    Interpola en planta los residuos ``Vs_real − Vs_ML`` del entrenamiento
    sobre las coordenadas de la malla 2D.  La corrección resulta constante a
    lo largo de cada vertical, que es exactamente el comportamiento de los
    modelos ``modelos_2d_vp_vs_*.csv`` de referencia.
    """
    x_col, y_col = config.coord_cols

    correccion, motor = kriging_ordinario_2d(
        train_df[x_col].values,
        train_df[y_col].values,
        residuos,
        grid_df[x_col].values,
        grid_df[y_col].values,
        modelo=config.variogram_model,
    )
    return np.asarray(correccion, float), motor


# ---------------------------------------------------------------------------
# 10. Diagnósticos de extrapolación y confianza
# ---------------------------------------------------------------------------
def flag_out_of_range(
    train_df: pd.DataFrame, grid_df: pd.DataFrame, cols: List[str]
) -> pd.Series:
    """True si CUALQUIERA de ``cols`` cae fuera de [min, max] del entrenamiento."""
    fuera_de_rango = pd.Series(False, index=grid_df.index)

    for col in cols:
        if col in train_df.columns:
            min_val = train_df[col].min()
            max_val = train_df[col].max()
            fuera_de_rango = fuera_de_rango | (
                (grid_df[col] < min_val) | (grid_df[col] > max_val)
            )

    return fuera_de_rango


def diagnose_extrapolation_by_feature(
    train_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    feature_list: List[str],
) -> pd.DataFrame:
    """
    Reporta, **por variable**, cuántos nodos caen fuera del rango de
    entrenamiento.  Esto separa la causa real: un porcentaje alto por X/Y es
    extrapolación espacial (esperada), mientras que por Vp/Elevación es
    extrapolación petrofísica, más riesgosa porque ahí la relación Vp→Vs
    nunca se validó.
    """
    filas = []
    n_total = len(grid_df)

    for col in feature_list:
        if col not in train_df.columns:
            continue
        min_train, max_train = train_df[col].min(), train_df[col].max()
        fuera = (grid_df[col] < min_train) | (grid_df[col] > max_train)
        n_fuera = int(fuera.sum())

        filas.append(
            {
                "variable": col,
                "min_train": min_train,
                "max_train": max_train,
                "min_grid": grid_df[col].min(),
                "max_grid": grid_df[col].max(),
                "pct_nodos_fuera_de_rango": (
                    round(100 * n_fuera / n_total, 1) if n_total > 0 else 0.0
                ),
            }
        )

    return pd.DataFrame(filas).sort_values(
        "pct_nodos_fuera_de_rango", ascending=False
    ).reset_index(drop=True)


def compute_spatial_confidence_zone(
    train_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, float]:
    """
    Distancia de cada nodo al MASW más cercano y zona de confianza continua
    (Alta / Media / Baja), usando como referencia la separación típica
    **entre** los propios MASW (vecino más cercano interno).

    Reemplaza al flag binario: un nodo puede estar fuera del rango de X/Y y
    aun así muy cerca de un dato real, y viceversa.
    """
    x_col, y_col = config.coord_cols

    coords_train = train_df[[x_col, y_col]].values
    coords_grid = grid_df[[x_col, y_col]].values

    tree = cKDTree(coords_train)

    k_interno = min(2, len(coords_train))
    dist_interna, _ = tree.query(coords_train, k=k_interno)
    if dist_interna.ndim == 1:
        escala_referencia = 1.0
    else:
        positivas = dist_interna[:, -1][dist_interna[:, -1] > 0]
        escala_referencia = float(np.median(positivas)) if positivas.size else 1.0
    if escala_referencia <= 0:
        escala_referencia = 1.0

    dist_grid, _ = tree.query(coords_grid, k=1)

    zona = pd.Series("Alta", index=grid_df.index)
    zona[dist_grid > 2 * escala_referencia] = "Media"
    zona[dist_grid > 5 * escala_referencia] = "Baja"

    resultado = pd.DataFrame(
        {
            "dist_al_masw_mas_cercano": dist_grid,
            "zona_confianza_espacial": zona.values,
        },
        index=grid_df.index,
    )
    return resultado, escala_referencia


def report_feature_importance(
    model: BaseEstimator, feature_list: List[str], model_type: str
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Importancia relativa de cada variable.  Diagnóstico clave: si X/Y dominan
    sobre Vp, el modelo está memorizando posiciones en vez de aprender la
    relación física Vp→Vs — síntoma de sobreajuste espacial.
    """
    importancias = None

    if model_type == "ridge":
        importancias = np.abs(model.coef_)
    else:
        importancias = getattr(model, "feature_importances_", None)

    if importancias is None:
        return pd.DataFrame(columns=["variable", "importancia_relativa"]), None

    suma = float(np.sum(importancias))
    if suma > 0:
        importancias = importancias / suma

    reporte = (
        pd.DataFrame(
            {"variable": feature_list, "importancia_relativa": importancias}
        )
        .sort_values("importancia_relativa", ascending=False)
        .reset_index(drop=True)
    )

    aviso = None
    importancia_vp = reporte.loc[reporte["variable"] == "Vp", "importancia_relativa"]
    if len(importancia_vp) > 0:
        if importancia_vp.iloc[0] < 0.25:
            aviso = (
                f"Vp tiene baja importancia relativa ({100 * importancia_vp.iloc[0]:.1f} %) "
                "frente a otras variables: el modelo depende más de la posición "
                "espacial que de la física Vp→Vs, lo que reduce la confiabilidad "
                "al extrapolar fuera del footprint de los MASW."
            )

    return reporte, aviso


def clip_predictions_to_physical_range(
    y_pred: np.ndarray, y_train: np.ndarray, margin_pct: float
) -> Tuple[np.ndarray, int, Tuple[float, float]]:
    """
    Recorta las predicciones al rango físicamente razonable
    ``[min(Vs) − margen, max(Vs) + margen]``, con el límite inferior acotado
    a 0 (Vs no puede ser negativa).  El margen es un porcentaje del rango
    observado, no un límite arbitrario.
    """
    rango = float(y_train.max() - y_train.min())
    margen = rango * margin_pct

    limite_inf = max(float(y_train.min()) - margen, 0.0)
    limite_sup = float(y_train.max()) + margen

    y_pred_clip = np.clip(y_pred, limite_inf, limite_sup)
    n_ajustados = int(np.sum(y_pred != y_pred_clip))

    return y_pred_clip, n_ajustados, (limite_inf, limite_sup)


# ---------------------------------------------------------------------------
# 11. Predicción sobre la malla 2D
# ---------------------------------------------------------------------------
def predict_vs_grid(
    model: BaseEstimator,
    scaler: StandardScaler,
    grid_df: pd.DataFrame,
    feature_list: List[str],
) -> np.ndarray:
    """Predicción base de Vs (componente ML) sobre la malla 2D."""
    return model.predict(scaler.transform(grid_df[feature_list].values))


# ---------------------------------------------------------------------------
# 12. Orquestador
# ---------------------------------------------------------------------------
def run_full_pipeline(
    train_source: pd.DataFrame,
    grid_source: pd.DataFrame,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo y devuelve resultados + diagnósticos.

    Las advertencias que el módulo original emitía por ``logging`` se
    devuelven aquí en ``resultado['avisos']`` para poder mostrarlas en la
    interfaz.
    """
    if config is None:
        config = PipelineConfig()

    avisos: List[str] = []

    # 1. Datos
    train_df, av = validar_entrenamiento(train_source, config)
    avisos += av
    grid_df, av = validar_malla(grid_source, config)
    avisos += av

    # 2. Sanity check geofísico
    correlacion_vp_vs, aviso = check_vp_vs_correlation(train_df, config)
    if aviso:
        avisos.append(aviso)

    # 3. Ingeniería de atributos (centroide del entrenamiento, reutilizado)
    train_df, centroid = engineer_features(train_df, config, centroid=None)
    grid_df, _ = engineer_features(grid_df, config, centroid=centroid)
    feature_list = get_full_feature_list(config)

    nota_extrapolacion = None
    if not model_supports_extrapolation(config.model_type):
        nota_extrapolacion = (
            f"`{config.model_type}` **no extrapola** linealmente: fuera del rango de "
            "entrenamiento las predicciones se aplanan. Si el diagnóstico de "
            "extrapolación muestra porcentajes altos, compare contra `ridge`."
        )
        avisos.append(nota_extrapolacion)

    # 4. Hiperparámetros
    best_params, diag_opt = optimize_hyperparameters(train_df, config, feature_list)

    # 5. Evaluación honesta
    evaluacion = evaluate_model_spatial(train_df, config, feature_list, best_params)

    # 6. Modelo final
    model, scaler = train_final_model(train_df, config, feature_list, best_params)
    importancia_features, aviso = report_feature_importance(
        model, feature_list, config.model_type
    )
    if aviso:
        avisos.append(aviso)

    # 7. Predicción base + Regression Kriging
    grid_df["Vs_predicho_ML"] = predict_vs_grid(model, scaler, grid_df, feature_list)

    correccion_kriging = np.zeros(len(grid_df))
    motor_kriging = None
    if config.apply_residual_kriging:
        correccion_kriging, motor_kriging = krige_residuals(
            train_df, evaluacion["residuos"], grid_df, config
        )
    grid_df["Correccion_residual_kriging"] = correccion_kriging

    vs_predicho = grid_df["Vs_predicho_ML"].values + correccion_kriging

    n_recortados, limites = 0, (np.nan, np.nan)
    if config.clip_predictions:
        vs_predicho, n_recortados, limites = clip_predictions_to_physical_range(
            vs_predicho, train_df[config.target_col].values, config.clip_margin_pct
        )
        if n_recortados > 0:
            avisos.append(
                f"{n_recortados} predicciones se recortaron al rango físico "
                f"[{limites[0]:.1f}, {limites[1]:.1f}] m/s (extrapolaban más allá "
                "de un margen razonable)."
            )
    grid_df["Vs_predicho"] = vs_predicho
    grid_df["Vs"] = vs_predicho  # alias usado por los Tabs 2 y 3

    # 8. Extrapolación
    diagnostico_extrapolacion = pd.DataFrame()
    if config.flag_extrapolation:
        grid_df["flag_extrapolacion_general"] = flag_out_of_range(
            train_df, grid_df, feature_list
        )
        grid_df["flag_extrapolacion_petrofisica"] = flag_out_of_range(
            train_df, grid_df, list(config.petrofisicas_cols)
        )
        diagnostico_extrapolacion = diagnose_extrapolation_by_feature(
            train_df, grid_df, feature_list
        )
    else:
        grid_df["flag_extrapolacion_general"] = False
        grid_df["flag_extrapolacion_petrofisica"] = False

    # 9. Confianza espacial
    escala_referencia = np.nan
    if config.compute_spatial_confidence:
        confianza, escala_referencia = compute_spatial_confidence_zone(
            train_df, grid_df, config
        )
        grid_df["zona_confianza_espacial"] = confianza["zona_confianza_espacial"].values
        grid_df["dist_al_masw_mas_cercano"] = confianza[
            "dist_al_masw_mas_cercano"
        ].values

    return {
        "model": model,
        "scaler": scaler,
        "config": config,
        "feature_list": feature_list,
        "centroide": centroid,
        "best_params": best_params,
        "diagnostico_optimizacion": diag_opt,
        "metrics": evaluacion["metrics"],
        "metricas_por_grupo": evaluacion["por_grupo"],
        "residuos": evaluacion["residuos"],
        "y_true": evaluacion["y_true"],
        "y_pred_oof": evaluacion["y_pred_oof"],
        "correlacion_vp_vs": correlacion_vp_vs,
        "importancia_features": importancia_features,
        "diagnostico_extrapolacion": diagnostico_extrapolacion,
        "motor_kriging": motor_kriging,
        "escala_referencia_masw": escala_referencia,
        "n_predicciones_recortadas": n_recortados,
        "limites_fisicos": limites,
        "train_df_enriquecido": train_df,
        "grid_df_predicho": grid_df,
        "avisos": avisos,
    }


def aplicar_a_nueva_malla(
    resultado: Dict[str, Any],
    grid_source: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aplica un pipeline ya entrenado a otra malla 2D (otra línea de la misma
    campaña) sin reentrenar: reutiliza modelo, escalador, centroide y
    residuos.
    """
    config: PipelineConfig = resultado["config"]
    grid_df, _ = validar_malla(grid_source, config)
    grid_df, _ = engineer_features(grid_df, config, centroid=resultado["centroide"])

    grid_df["Vs_predicho_ML"] = predict_vs_grid(
        resultado["model"], resultado["scaler"], grid_df, resultado["feature_list"]
    )

    correccion = np.zeros(len(grid_df))
    if config.apply_residual_kriging:
        correccion, _ = krige_residuals(
            resultado["train_df_enriquecido"], resultado["residuos"], grid_df, config
        )
    grid_df["Correccion_residual_kriging"] = correccion

    vs = grid_df["Vs_predicho_ML"].values + correccion
    if config.clip_predictions:
        vs, _, _ = clip_predictions_to_physical_range(
            vs,
            resultado["train_df_enriquecido"][config.target_col].values,
            config.clip_margin_pct,
        )
    grid_df["Vs_predicho"] = vs
    grid_df["Vs"] = vs

    grid_df["flag_extrapolacion_general"] = flag_out_of_range(
        resultado["train_df_enriquecido"], grid_df, resultado["feature_list"]
    )
    grid_df["flag_extrapolacion_petrofisica"] = flag_out_of_range(
        resultado["train_df_enriquecido"], grid_df, list(config.petrofisicas_cols)
    )
    confianza, _ = compute_spatial_confidence_zone(
        resultado["train_df_enriquecido"], grid_df, config
    )
    grid_df["zona_confianza_espacial"] = confianza["zona_confianza_espacial"].values
    grid_df["dist_al_masw_mas_cercano"] = confianza["dist_al_masw_mas_cercano"].values

    return grid_df


# ---------------------------------------------------------------------------
# 13. Persistencia de artefactos
# ---------------------------------------------------------------------------
def artefactos_a_bytes(resultado: Dict[str, Any]) -> bytes:
    """Serializa modelo, escalador, configuración y métricas a un joblib."""
    artefactos = {
        "model": resultado["model"],
        "scaler": resultado["scaler"],
        "config": resultado["config"],
        "feature_list": resultado["feature_list"],
        "centroide": resultado["centroide"],
        "best_params": resultado["best_params"],
        "metrics": resultado["metrics"],
    }
    buffer = io.BytesIO()
    joblib.dump(artefactos, buffer)
    return buffer.getvalue()


def save_artifacts(resultado: Dict[str, Any], output_path: str) -> None:
    """Guarda los artefactos en disco (equivalente al del módulo original)."""
    with open(output_path, "wb") as fh:
        fh.write(artefactos_a_bytes(resultado))


def load_artifacts(input_path: str) -> Dict[str, Any]:
    """Carga artefactos previamente guardados."""
    return joblib.load(input_path)


# ---------------------------------------------------------------------------
# 14. Síntesis de línea por kriging 3D (traducción de `synthesize_line`)
# ---------------------------------------------------------------------------
def sintetizar_linea_kriging(
    df_source: pd.DataFrame,
    topo_objetivo: pd.DataFrame,
    variable: str = "Vp",
    modelo_variograma: str = "spherical",
    anisotropy_scaling_z: float = 0.05,
    rango_variograma: float = 500.0,
    prof_max: float = 30.0,
    dz: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Genera una línea 2D nueva (real o sintética) interpolando ``variable``
    desde los puntos fuente por Kriging Ordinario 3D con anisotropía
    vertical, tal como hace ``synthesize_line`` de
    ``crear_linea_sintetica.ipynb``.

    A diferencia del Regression Kriging de arriba (que corrige residuos en
    planta), aquí el kriging **es** el interpolador: se usa cuando no hay
    modelo de velocidades en esa traza y hay que construirlo desde las
    líneas vecinas.
    """
    from .espacial import malla_objetivo

    objetivo = malla_objetivo(topo_objetivo, prof_max=prof_max, dz=dz)

    valores = df_source[variable].to_numpy(float)
    params = parametros_por_defecto(valores, rango_variograma)

    est, var, motor = kriging_ordinario_3d(
        df_source["X"].to_numpy(float),
        df_source["Y"].to_numpy(float),
        df_source["Elevacion"].to_numpy(float),
        valores,
        objetivo["X"].to_numpy(float),
        objetivo["Y"].to_numpy(float),
        objetivo["Elevacion"].to_numpy(float),
        modelo=modelo_variograma,
        parametros=params,
        anisotropy_scaling_z=anisotropy_scaling_z,
    )

    df_final = objetivo[["Xo", "X", "Y", "Elevacion", "Z_superficie", "Profundidad"]].copy()
    df_final[variable] = est
    df_final["Varianza_kriging"] = var
    df_final["Z"] = df_final["Elevacion"]
    df_final = df_final.sort_values(["Xo", "Elevacion"], ascending=[True, False]).reset_index(
        drop=True
    )

    info = {
        "motor_kriging": motor,
        "n_fuente": int(len(df_source)),
        "n_objetivo": int(len(df_final)),
        "variograma": {
            "modelo": modelo_variograma,
            **params,
            "anisotropy_scaling_z": anisotropy_scaling_z,
        },
    }
    return df_final, info


# ---------------------------------------------------------------------------
# 15. Exportación X, Y, Z, V
# ---------------------------------------------------------------------------
def exportar_xyzv(
    df: pd.DataFrame,
    columna_velocidad: str = "Vs",
    usar_elevacion: bool = True,
) -> pd.DataFrame:
    """
    Perfil en el formato de intercambio ``X, Y, Z, V``.

    ``usar_elevacion=True`` escribe la elevación absoluta (msnm) en ``Z``;
    ``False`` escribe la profundidad bajo el terreno (positiva hacia abajo).
    """
    z = None

    if usar_elevacion:
        if "Elevacion" in df.columns:
            z = df["Elevacion"]
        else:
            z = df["Z"]
    else:
        if "Profundidad" in df.columns:
            z = df["Profundidad"]
        else:
            z = df["Z"] - df["Elevacion"]

    return pd.DataFrame(
        {
            "X": df["X"].to_numpy(float),
            "Y": df["Y"].to_numpy(float),
            "Z": np.asarray(z, float),
            "V": df[columna_velocidad].to_numpy(float),
        }
    )
