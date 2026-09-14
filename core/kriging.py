"""
core/kriging.py
===============

Kriging Ordinario 3D con anisotropía vertical, tal como se usa en
``crear_linea_sintetica.ipynb``.

Si ``pykrige`` está instalado se usa :class:`pykrige.ok3d.OrdinaryKriging3D`
(la ruta del notebook).  Si no, se emplea una implementación propia,
equivalente y sin dependencias extra, que resuelve el sistema de kriging
ordinario sobre una **vecindad local** de los ``k`` puntos más cercanos.
Esa variante local es además la única viable cuando la malla objetivo del
perfil 2D tiene decenas de miles de nodos.

Convención de anisotropía
-------------------------
La distancia efectiva es::

    h = sqrt( dx² + dy² + (dz / anisotropy_scaling_z)² )

de modo que el rango vertical resulta ``rango_z = rango_xy ·
anisotropy_scaling_z`` — exactamente lo que documenta ``synthesize_line``
(``anisotropy_scaling_z=0.05`` ⇒ rango vertical = 5 % del horizontal).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

MODELOS_VARIOGRAMA = ["spherical", "exponential", "gaussian", "linear"]

try:  # pragma: no cover - depende del entorno
    from pykrige.ok import OrdinaryKriging  # type: ignore
    from pykrige.ok3d import OrdinaryKriging3D  # type: ignore

    PYKRIGE_DISPONIBLE = True
except Exception:  # pragma: no cover
    OrdinaryKriging = None  # type: ignore
    OrdinaryKriging3D = None  # type: ignore
    PYKRIGE_DISPONIBLE = False


# ---------------------------------------------------------------------------
# Variogramas
# ---------------------------------------------------------------------------
def _gamma(h: np.ndarray, modelo: str, psill: float, rango: float, nugget: float):
    """Semivariograma teórico.  ``gamma(0) = 0`` por definición."""
    h = np.asarray(h, dtype=float)
    a = max(float(rango), 1e-9)

    if modelo == "exponential":
        g = psill * (1.0 - np.exp(-3.0 * h / a))
    elif modelo == "gaussian":
        g = psill * (1.0 - np.exp(-3.0 * (h / a) ** 2))
    elif modelo == "linear":
        g = psill * np.minimum(h / a, 1.0)
    else:  # spherical
        r = np.minimum(h / a, 1.0)
        g = psill * (1.5 * r - 0.5 * r**3)

    g = g + nugget
    return np.where(h <= 1e-12, 0.0, g)


def parametros_por_defecto(valores: np.ndarray, rango: float = 500.0) -> dict:
    """Mismos valores por defecto que ``synthesize_line``."""
    sill = float(np.var(valores))
    return {"psill": sill, "range": float(rango), "nugget": 0.1 * sill}


# ---------------------------------------------------------------------------
# Implementación propia (vecindad local)
# ---------------------------------------------------------------------------
def _kriging_local(
    x_src: np.ndarray,
    y_src: np.ndarray,
    z_src: np.ndarray,
    v_src: np.ndarray,
    x_dst: np.ndarray,
    y_dst: np.ndarray,
    z_dst: np.ndarray,
    modelo: str,
    psill: float,
    rango: float,
    nugget: float,
    anisotropy_scaling_z: float,
    n_vecinos: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Kriging ordinario resuelto punto a punto sobre los k vecinos más cercanos."""
    sz = max(float(anisotropy_scaling_z), 1e-9)

    P = np.column_stack([x_src, y_src, z_src / sz]).astype(float)
    Q = np.column_stack([x_dst, y_dst, z_dst / sz]).astype(float)
    v = np.asarray(v_src, dtype=float)

    arbol = cKDTree(P)
    k = int(min(n_vecinos, len(P)))
    _, idx = arbol.query(Q, k=k)
    if idx.ndim == 1:
        idx = idx[:, None]

    est = np.empty(len(Q))
    var = np.empty(len(Q))

    for i in range(len(Q)):
        ids = idx[i]
        pts = P[ids]
        vals = v[ids]

        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        G = _gamma(d, modelo, psill, rango, nugget)

        A = np.ones((k + 1, k + 1))
        A[:k, :k] = G
        A[k, k] = 0.0

        d0 = np.linalg.norm(pts - Q[i], axis=-1)
        b = np.ones(k + 1)
        b[:k] = _gamma(d0, modelo, psill, rango, nugget)

        try:
            sol = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, b, rcond=None)[0]

        w = sol[:k]
        est[i] = float(w @ vals)
        var[i] = float(max(w @ b[:k] + sol[k], 0.0))

    return est, var


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def kriging_ordinario_3d(
    x_src,
    y_src,
    z_src,
    v_src,
    x_dst,
    y_dst,
    z_dst,
    modelo: str = "spherical",
    parametros: dict | None = None,
    anisotropy_scaling_z: float = 0.05,
    n_vecinos: int = 48,
    forzar_propio: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Interpola ``v_src`` de los puntos fuente a los puntos destino.

    Returns
    -------
    estimacion : np.ndarray
    varianza : np.ndarray
    motor : str
        ``'pykrige'`` o ``'local'``, para poder reportarlo en la interfaz.
    """
    x_src = np.asarray(x_src, float)
    y_src = np.asarray(y_src, float)
    z_src = np.asarray(z_src, float)
    v_src = np.asarray(v_src, float)
    x_dst = np.asarray(x_dst, float)
    y_dst = np.asarray(y_dst, float)
    z_dst = np.asarray(z_dst, float)

    p = dict(parametros or parametros_por_defecto(v_src))
    psill = float(p.get("psill", p.get("sill", np.var(v_src))))
    rango = float(p.get("range", 500.0))
    nugget = float(p.get("nugget", 0.1 * psill))

    # pykrige resuelve el sistema global (matriz n_src × n_src) y lo aplica a
    # todos los destinos: es exacto pero escala muy mal.  Por encima de unos
    # cientos de puntos fuente se usa el motor local, que da el mismo
    # resultado práctico con vecindades y KD-tree.
    usar_pykrige = (
        PYKRIGE_DISPONIBLE
        and not forzar_propio
        and len(x_src) <= 600
        and len(x_dst) <= 20_000
    )
    if usar_pykrige:
        try:  # pragma: no cover
            ok = OrdinaryKriging3D(
                x_src,
                y_src,
                z_src,
                v_src,
                variogram_model=modelo,
                variogram_parameters={
                    "sill": psill + nugget,
                    "range": rango,
                    "nugget": nugget,
                },
                anisotropy_scaling_z=1.0 / max(anisotropy_scaling_z, 1e-9),
                enable_plotting=False,
            )
            est, ss = ok.execute("points", x_dst, y_dst, z_dst)
            return np.asarray(est, float), np.asarray(ss, float), "pykrige"
        except Exception:
            pass  # se cae al motor propio

    est, var = _kriging_local(
        x_src,
        y_src,
        z_src,
        v_src,
        x_dst,
        y_dst,
        z_dst,
        modelo=modelo,
        psill=psill,
        rango=rango,
        nugget=nugget,
        anisotropy_scaling_z=anisotropy_scaling_z,
        n_vecinos=n_vecinos,
    )
    return est, var, "local"


# ---------------------------------------------------------------------------
# Kriging ordinario 2D (corrección de residuos del Regression Kriging)
# ---------------------------------------------------------------------------
def kriging_ordinario_2d(
    x_src,
    y_src,
    v_src,
    x_dst,
    y_dst,
    modelo: str = "spherical",
    n_vecinos: int = 32,
    forzar_propio: bool = False,
) -> tuple[np.ndarray, str]:
    """
    Interpola en planta (X, Y) — el esquema que usa ``krige_residuals`` del
    pipeline de proyección de Vs: los residuos se corrigen espacialmente en
    superficie, no en 3D, de modo que la corrección es constante a lo largo
    de cada vertical.

    Usa ``pykrige.ok.OrdinaryKriging`` si está disponible; si no, un kriging
    ordinario local propio.  En la ruta propia los residuos se promedian
    primero por coordenada única, porque un sondeo MASW aporta decenas de
    nodos exactamente en el mismo (X, Y) y eso vuelve singular el sistema.

    Returns
    -------
    estimacion : np.ndarray
    motor : str
        ``'pykrige'`` o ``'local'``.
    """
    x_src = np.asarray(x_src, float)
    y_src = np.asarray(y_src, float)
    v_src = np.asarray(v_src, float)
    x_dst = np.asarray(x_dst, float)
    y_dst = np.asarray(y_dst, float)

    if not forzar_propio and PYKRIGE_DISPONIBLE:
        try:  # pragma: no cover
            ok = OrdinaryKriging(
                x_src,
                y_src,
                v_src,
                variogram_model=modelo,
                verbose=False,
                enable_plotting=False,
            )
            est, _ = ok.execute("points", x_dst, y_dst)
            est = np.asarray(est, float)
            if np.all(np.isfinite(est)):
                return est, "pykrige"
        except Exception:
            pass

    # --- motor propio: promedio por coordenada única + kriging local ------
    llaves = np.column_stack([x_src, y_src])
    unicas, inverso = np.unique(llaves, axis=0, return_inverse=True)
    suma = np.zeros(len(unicas))
    cuenta = np.zeros(len(unicas))
    np.add.at(suma, inverso, v_src)
    np.add.at(cuenta, inverso, 1.0)
    v_u = suma / np.maximum(cuenta, 1.0)

    if len(unicas) == 1:
        return np.full(len(x_dst), float(v_u[0])), "local"

    psill = float(np.var(v_u))
    if psill <= 0:
        return np.full(len(x_dst), float(v_u.mean())), "local"

    # Rango: mediana de la distancia al vecino más cercano entre los sondeos,
    # escalada — es la escala de correlación que los propios datos sugieren.
    arbol = cKDTree(unicas)
    d_int, _ = arbol.query(unicas, k=min(2, len(unicas)))
    escala = float(np.median(d_int[:, -1])) if d_int.ndim == 2 else 1.0
    rango = max(escala * 3.0, 1e-6)
    nugget = 0.1 * psill

    est, _ = _kriging_local(
        unicas[:, 0],
        unicas[:, 1],
        np.zeros(len(unicas)),
        v_u,
        x_dst,
        y_dst,
        np.zeros(len(x_dst)),
        modelo=modelo,
        psill=psill,
        rango=rango,
        nugget=nugget,
        anisotropy_scaling_z=1.0,
        n_vecinos=min(n_vecinos, len(unicas)),
    )
    return est, "local"
