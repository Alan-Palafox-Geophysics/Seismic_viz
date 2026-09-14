"""
check_entorno.py
================

Diagnóstico del entorno antes de ejecutar la aplicación.

Detecta el problema más común al instalar sobre un entorno conda existente:
paquetes compilados (matplotlib, scipy, scikit-learn, pykrige…) construidos
contra una versión de NumPy distinta a la instalada.  El síntoma es::

    ImportError: numpy.core.multiarray failed to import

Uso::

    python check_entorno.py
"""

from __future__ import annotations

import importlib
import sys

# (módulo, nombre en pip, obligatorio)
PAQUETES = [
    ("numpy", "numpy", True),
    ("pandas", "pandas", True),
    ("scipy", "scipy", True),
    ("sklearn", "scikit-learn", True),
    ("matplotlib", "matplotlib", True),
    ("plotly", "plotly", True),
    ("PIL", "Pillow", True),
    ("streamlit", "streamlit", True),
    ("joblib", "joblib", True),
    ("optuna", "optuna", False),
    ("pykrige", "pykrige", False),
    ("rasterio", "rasterio", False),
]

# Paquetes con extensiones compiladas que enlazan contra la ABI de NumPy.
COMPILADOS = {"pandas", "scipy", "sklearn", "matplotlib", "pykrige", "rasterio"}

SINTOMA_ABI = (
    "numpy.core.multiarray failed to import",
    "numpy.dtype size changed",
    "binary incompatibility",
    "compiled using numpy",
    "_ARRAY_API not found",
)


def _es_error_abi(mensaje: str) -> bool:
    bajo = mensaje.lower()
    return any(s.lower() in bajo for s in SINTOMA_ABI)


def main() -> int:
    print("=" * 72)
    print("Diagnóstico del entorno — TRS + MASW")
    print("=" * 72)
    print(f"Python     : {sys.version.split()[0]}")
    print(f"Ejecutable : {sys.executable}")
    print("-" * 72)

    fallas_abi: list[str] = []
    faltantes_obligatorios: list[str] = []
    faltantes_opcionales: list[str] = []
    version_numpy = None

    for modulo, paquete, obligatorio in PAQUETES:
        try:
            mod = importlib.import_module(modulo)
            version = getattr(mod, "__version__", "?")
            if modulo == "numpy":
                version_numpy = version
            marca = "OK  "
            print(f"{marca} {paquete:<16} {version}")
        except ImportError as exc:
            mensaje = str(exc)
            if _es_error_abi(mensaje) and modulo in COMPILADOS:
                fallas_abi.append(paquete)
                print(f"ABI  {paquete:<16} INCOMPATIBLE CON NUMPY  → {mensaje[:60]}")
            else:
                if obligatorio:
                    faltantes_obligatorios.append(paquete)
                    print(f"FALTA {paquete:<15} (obligatorio)")
                else:
                    faltantes_opcionales.append(paquete)
                    print(f"--   {paquete:<16} no instalado (opcional)")
        except Exception as exc:  # pragma: no cover
            print(f"ERR  {paquete:<16} {type(exc).__name__}: {str(exc)[:60]}")

    print("-" * 72)

    if fallas_abi:
        print("\n>>> CONFLICTO DE BINARIOS CON NUMPY <<<\n")
        print(
            f"NumPy instalado: {version_numpy}. Los paquetes "
            f"{', '.join(fallas_abi)} fueron compilados contra otra versión "
            "mayor de NumPy.\n"
            "Esto ocurre al instalar con pip sobre un entorno conda que ya\n"
            "traía esos paquetes: quedan dos cadenas de binarios mezcladas.\n"
        )
        print("Solución recomendada — entorno limpio:\n")
        print("    conda create -n trs python=3.11 -y")
        print("    conda activate trs")
        print("    pip install -r requirements.txt\n")
        print("Alternativa — reparar el entorno actual sin mezclar gestores:\n")
        print("    pip install --force-reinstall --no-cache-dir \\")
        print("        numpy pandas scipy scikit-learn matplotlib\n")
        print(
            "Si alguna otra herramienta de ese entorno exige NumPy 1.x, fije\n"
            "esa rama en su lugar:\n"
        )
        print('    pip install --force-reinstall --no-cache-dir "numpy<2" \\')
        print("        pandas scipy scikit-learn matplotlib\n")
        return 1

    if faltantes_obligatorios:
        print("\nFaltan paquetes obligatorios:", ", ".join(faltantes_obligatorios))
        print("\n    pip install -r requirements.txt\n")
        return 1

    print("\nEntorno correcto. La aplicación puede ejecutarse:\n")
    print("    streamlit run app.py\n")

    if faltantes_opcionales:
        print("Paquetes opcionales ausentes:", ", ".join(faltantes_opcionales))
        degradacion = {
            "optuna": "búsqueda de hiperparámetros en grid manual",
            "pykrige": "motor de kriging propio (vecindad local con KD-tree)",
            "rasterio": "georreferencia del GeoTIFF por etiquetas Pillow o esquinas manuales",
        }
        for p in faltantes_opcionales:
            print(f"  · sin {p}: {degradacion.get(p, 'funcionalidad reducida')}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
