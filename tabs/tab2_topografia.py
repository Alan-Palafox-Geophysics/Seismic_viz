"""
tabs/tab2_topografia.py
=======================

**Tab 2 — Topografía y predicción espacial de Vs.**

1. Carga del catálogo topográfico (``Zona_Num, Linea_Num, Zona_Nombre,
   Linea, ID, Xo, X, Y, Z``) y navegación por zona y línea.
2. «Agregar»: cruza la posición horizontal ``Xo`` de la topografía con los
   datos geofísicos registrados en el Tab 1 — el MASW se ancla a la
   coordenada central del tendido TRS y los SEV/VES a la coincidencia
   exacta de ``Xo`` en la lista de estaciones.
3. «Generar Perfil 2D»: ejecuta el pipeline de ``vs_projection_pipeline.py``
   — optimización de hiperparámetros sobre validación cruzada espacial
   (Leave-One-Line-Out), métricas honestas *out-of-fold*, Regression
   Kriging de los residuos, recorte físico y diagnósticos de extrapolación
   y confianza — y proyecta el resultado sobre toda la malla 2D de Vp.
4. Exportación en formato ``X, Y, Z, V`` y de los artefactos del modelo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import espacial, servicios
from core.kriging import MODELOS_VARIOGRAMA, PYKRIGE_DISPONIBLE
from core.plots_2d import CMAPS_DISPONIBLES
from core.prediccion import (
    MODELOS_DISPONIBLES,
    OPTUNA_DISPONIBLE,
    artefactos_a_bytes,
    calificar_rmse_relativo,
    exportar_xyzv,
    model_supports_extrapolation,
)

# Columnas que el pipeline necesita en el conjunto de entrenamiento.
COLS_ENTRENAMIENTO = ["Linea", "X", "Y", "Z", "Profundidad", "Elevacion", "Vs", "Vp"]


def _rerun() -> None:
    """
    Relanza el script con el nombre que tenga la API en esta versión.

    ``st.rerun()`` existe desde Streamlit 1.27; antes era
    ``st.experimental_rerun()``.  Si no hay ninguna de las dos, se continúa
    sin relanzar: el perfil ya quedó registrado y aparecerá en cuanto haya
    cualquier otra interacción.
    """
    for nombre in ("rerun", "experimental_rerun"):
        fn = getattr(st, nombre, None)
        if callable(fn):
            fn()
            return


def _registrar_perfil(nombre: str, df, topo_linea, origen: str) -> None:
    """
    Guarda un perfil 2D en la sesión y lo deja seleccionado en el Tab 3.

    El multiselect del Tab 3 conserva su valor entre ejecuciones, así que su
    ``default`` se ignora una vez que el usuario tocó el control: hay que
    añadir el perfil nuevo a la selección almacenada para que se vea sin
    tener que elegirlo a mano.
    """
    st.session_state.setdefault("perfiles_2d", {})
    st.session_state["perfiles_2d"][nombre] = {
        "nombre": nombre,
        "df": df,
        "topo_linea": topo_linea,
        "origen": origen,
    }
    if "t3_sel" in st.session_state:
        seleccion = list(st.session_state["t3_sel"])
        if nombre not in seleccion:
            seleccion.append(nombre)
        st.session_state["t3_sel"] = seleccion


def _etiqueta_origen(df: pd.DataFrame, base: str) -> str:
    """Describe un perfil por las variables de velocidad que ya contiene."""
    presentes = [c for c in ("Vp", "Vs") if c in df.columns]
    return f"{base} {'+'.join(presentes)}" if presentes else base


def _fusionar_sintetica(
    nombre: str,
    df_nuevo: pd.DataFrame,
    variable: str,
    topo_linea: pd.DataFrame,
) -> str | None:
    """
    Agrega la variable recién sintetizada al perfil de esa línea, **sin borrar
    lo que ya tuviera**.

    Sintetizar Vp y después Vs sobre la misma línea objetivo produce dos
    mallas con idéntica geometría (misma topografía, misma profundidad y
    mismo ``dz``), así que la segunda sólo aporta columnas nuevas.  Se
    empalman por ``(Xo, Elevacion)``; si el usuario cambió la geometría entre
    una síntesis y otra las mallas no casan, y entonces el perfil se
    reemplaza avisándolo en vez de mezclar nodos que no se corresponden.

    Returns
    -------
    str | None
        Aviso para el usuario cuando hubo que reemplazar, o ``None``.
    """
    perfiles = st.session_state.setdefault("perfiles_2d", {})
    previo = perfiles.get(nombre)
    aviso = None

    claves = ["Xo", "Elevacion"]
    fusionado = None

    if previo is not None and isinstance(previo.get("df"), pd.DataFrame):
        df_previo = previo["df"]
        if all(c in df_previo.columns for c in claves) and all(
            c in df_nuevo.columns for c in claves
        ):
            aportadas = [variable] + [
                c for c in df_nuevo.columns if c.startswith("Varianza_")
            ]
            izq = df_previo.drop(columns=aportadas, errors="ignore").copy()
            izq["_k"] = list(zip(izq["Xo"].round(4), izq["Elevacion"].round(4)))

            der = df_nuevo.copy()
            der["_k"] = list(zip(der["Xo"].round(4), der["Elevacion"].round(4)))
            mapa = der.set_index("_k")[aportadas]
            mapa = mapa[~mapa.index.duplicated(keep="first")]

            unido = izq.join(mapa, on="_k")
            cobertura = float(unido[variable].notna().mean())

            if cobertura >= 0.95:
                fusionado = unido.drop(columns=["_k"])
            else:
                aviso = (
                    f"La malla de esta síntesis no coincide con la del perfil "
                    f"«{nombre}» que ya existía (sólo empalma el "
                    f"{100 * cobertura:.0f} % de los nodos): probablemente cambió "
                    "la profundidad o el paso vertical. El perfil se reemplazó. "
                    "Para conservar ambas variables, vuelva a sintetizarlas con "
                    "la misma profundidad y el mismo dz."
                )

    df_final = fusionado if fusionado is not None else df_nuevo
    df_final = df_final.copy()
    df_final["Linea"] = nombre

    _registrar_perfil(
        nombre,
        df_final,
        topo_linea,
        _etiqueta_origen(df_final, "sintética") + " · kriging 3D",
    )
    return aviso


def _sincronizar_seleccion_3d() -> None:
    """Depura de la selección del Tab 3 los perfiles que ya no existen."""
    if "t3_sel" not in st.session_state:
        return
    disponibles = set(st.session_state.get("perfiles_2d", {}))
    st.session_state["t3_sel"] = [
        n for n in st.session_state["t3_sel"] if n in disponibles
    ]


# ---------------------------------------------------------------------------
# Carga de topografía
# ---------------------------------------------------------------------------
def _catalogo_topografia():
    archivo = st.file_uploader(
        "Catálogo topográfico  ·  Zona_Num, Linea_Num, Zona_Nombre, Linea, ID, Xo, X, Y, Z",
        type=["txt", "csv", "dat"],
        key="t2_topo",
    )
    if archivo is None:
        return None
    try:
        return servicios.cargar_topografia(archivo.getvalue())
    except Exception as exc:
        st.error(f"No se pudo leer la topografía: {exc}")
        return None


# ---------------------------------------------------------------------------
# Asignación espacial
# ---------------------------------------------------------------------------
def _panel_agregar(df_topo: pd.DataFrame) -> None:
    zonas = espacial.zonas_disponibles(df_topo)
    c1, c2, c3, c4 = st.columns([1.2, 1, 1.3, 1])

    with c1:
        zona = st.selectbox("Zona_Nombre", zonas, key="t2_zona")
    with c2:
        lineas = espacial.lineas_de_zona(df_topo, zona)
        linea = st.selectbox("Línea", lineas, key="t2_linea")

    topo_linea = espacial.topografia_de_linea(df_topo, zona, linea)
    centro = espacial.centro_del_tendido(topo_linea)

    registradas = list(st.session_state.get("lineas", {}).keys())
    with c3:
        if not registradas:
            st.info("Registre al menos una línea en el Tab 1.")
            fuente = None
        else:
            fuente = st.selectbox(
                "Datos geofísicos (del Tab 1)", registradas, key="t2_fuente"
            )
    with c4:
        tipo = st.selectbox("Tipo de sondeo puntual", ["MASW", "SEV / VES"], key="t2_tipo")

    with st.expander("Opciones del cruce Xo ↔ topografía", expanded=False):
        o1, o2, o3 = st.columns(3)
        with o1:
            modo = st.radio(
                "Emparejamiento de Xo",
                ["exacto", "cercano"],
                index=0,
                key="t2_modo",
                help=(
                    "‘exacto’ reproduce el merge de `Combine_Data_Topo` (inner join "
                    "sobre Xo). ‘cercano’ asocia cada columna de inversión a la "
                    "estaca más próxima dentro de la tolerancia."
                ),
            )
        with o2:
            tolerancia = st.number_input(
                "Tolerancia (m)",
                0.01,
                50.0,
                0.5,
                0.25,
                key="t2_tol",
                disabled=(modo == "exacto"),
            )
        with o3:
            xo_sev = st.number_input(
                "Xo del SEV/VES",
                value=float(topo_linea["Xo"].iloc[0]),
                step=1.0,
                key="t2_xo_sev",
                disabled=(tipo == "MASW"),
                help="Debe coincidir exactamente con una estación de la línea.",
            )

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Estaciones", centro["n_estaciones"])
    i2.metric("Longitud", f"{centro['longitud_m']:.1f} m")
    i3.metric("Xo central", f"{centro['Xo']:.1f} m")
    i4.metric("Cota en el centro", f"{centro['Z']:.2f} m")

    if fuente is None:
        return

    if st.button("➕ Agregar", type="primary"):
        linea_datos = st.session_state["lineas"][fuente]
        nombre = f"{zona} · {linea}"
        try:
            df_source, info_cruce = servicios.combinar_trs_con_topografia(
                linea_datos["df_trs"], topo_linea, nombre, modo, tolerancia
            )
        except Exception as exc:
            st.error(str(exc))
            return

        df_res = linea_datos["df_resultado"].rename(
            columns={"Vs_MASW_original": "Vs", "Vp_TRS_interpolado": "Vp"}
        )

        if tipo == "MASW":
            puntos_vs, ubic = espacial.emplazar_masw(df_res, topo_linea, nombre)
            regla = "coordenada central del tendido TRS"
        else:
            try:
                puntos_vs, ubic = espacial.emplazar_sev(
                    df_res, topo_linea, nombre, xo_sev
                )
            except ValueError as exc:
                st.error(str(exc))
                return
            regla = f"coincidencia exacta en Xo = {xo_sev:g}"

        puntos_vs["Vp"] = df_res["Vp"].to_numpy()
        puntos_vs = puntos_vs[[c for c in COLS_ENTRENAMIENTO if c in puntos_vs.columns]]

        st.session_state.setdefault("asignadas", {})
        st.session_state["asignadas"][nombre] = {
            "nombre": nombre,
            "zona": zona,
            "linea": linea,
            "fuente": fuente,
            "tipo": tipo,
            "topo_linea": topo_linea,
            "df_source": df_source,
            "puntos_vs": puntos_vs,
            "centro": centro,
            "ubicacion": ubic,
            "info_cruce": info_cruce,
        }
        st.success(
            f"«{nombre}» agregada — {info_cruce['columnas_cruzadas']} de "
            f"{info_cruce['columnas_trs']} columnas del TRS cruzaron con la "
            f"topografía ({info_cruce['filas_despues']} nodos). "
            f"Sondeo anclado por {regla}."
        )
        if info_cruce["xo_perdidas"]:
            st.caption(
                "Abscisas del TRS sin estación topográfica: "
                f"{info_cruce['xo_perdidas'][:12]}"
                + (" …" if len(info_cruce["xo_perdidas"]) > 12 else "")
            )


# ---------------------------------------------------------------------------
# Pipeline predictivo
# ---------------------------------------------------------------------------
def _panel_prediccion() -> None:
    asignadas = st.session_state.get("asignadas", {})
    if not asignadas:
        st.info("Agregue al menos una línea con topografía para entrenar el modelo.")
        return

    st.markdown("##### Líneas asignadas")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Línea": v["nombre"],
                    "Tipo": v["tipo"],
                    "Nodos 2D (Vp)": len(v["df_source"]),
                    "Nodos 1D (Vs)": len(v["puntos_vs"]),
                    "Columnas cruzadas": v["info_cruce"]["columnas_cruzadas"],
                    "Xo": f"{v['info_cruce']['xo_min']:g} … {v['info_cruce']['xo_max']:g}",
                }
                for v in asignadas.values()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        objetivo = st.multiselect(
            "Líneas del bloque a modelar (entrenamiento y malla)",
            list(asignadas.keys()),
            default=list(asignadas.keys()),
            key="t2_entrena",
            help=(
                "El modelo se entrena con los MASW de estas líneas y se aplica a sus "
                "propias mallas 2D de Vp, igual que los bloques Iyotla / Tiquimil "
                "de `Vs_Modelling.ipynb`."
            ),
        )
    with c2:
        st.write("")
        if st.button("🗑️ Vaciar lista", use_container_width=True):
            st.session_state["asignadas"] = {}
            _rerun()

    with st.expander("Configuración del pipeline", expanded=True):
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            model_type = st.selectbox(
                "Modelo",
                MODELOS_DISPONIBLES,
                index=0,
                key="t2_model",
                help=(
                    "‘ridge’ es la opción recomendada cuando buena parte de la malla "
                    "cae fuera del footprint de los MASW: los árboles NO extrapolan, "
                    "se aplanan en la frontera del rango de entrenamiento."
                ),
            )
            n_trials = st.slider("Iteraciones de búsqueda", 5, 200, 15, 5, key="t2_iter")
        with h2:
            cv_strategy = st.radio(
                "Validación cruzada espacial",
                ["line", "block"],
                index=0,
                key="t2_cv",
                help="‘line’ = Leave-One-Line-Out sobre la columna Linea.",
            )
            n_blocks = st.slider(
                "Bloques (si 'block')", 2, 12, 5, 1, key="t2_blocks",
                disabled=(cv_strategy == "line"),
            )
        with h3:
            kriging_residual = st.checkbox(
                "Regression Kriging de residuos", value=True, key="t2_krig"
            )
            variograma = st.selectbox(
                "Variograma", MODELOS_VARIOGRAMA, index=0, key="t2_var",
                disabled=not kriging_residual,
            )
        with h4:
            recortar = st.checkbox("Recorte físico de Vs", value=True, key="t2_clip")
            margen = st.slider(
                "Margen del recorte", 0.0, 0.5, 0.15, 0.05, key="t2_margen",
                disabled=not recortar,
            )
            semilla = st.number_input("Semilla", 0, 9999, 42, key="t2_seed")

        st.markdown("**Guardas de identificabilidad de los atributos**")
        g1, g2, g3 = st.columns(3)
        with g1:
            guardas = st.checkbox(
                "Descartar atributos no identificables",
                value=True,
                key="t2_guardas",
                help=(
                    "Elimina los atributos que la geometría de los sondeos no puede "
                    "sostener: los que son casi constantes entre MASW pero recorren "
                    "un rango amplio a lo largo del perfil. Sin esta guarda, el "
                    "modelo produce una rampa lateral en vez de un perfil "
                    "estratificado."
                ),
            )
        with g2:
            apalancamiento = st.slider(
                "Apalancamiento máximo",
                1.0,
                20.0,
                5.0,
                0.5,
                key="t2_lev",
                disabled=not guardas,
                help="rango en la malla / (2 × desviación en el entrenamiento).",
            )
        with g3:
            interaccion = st.checkbox(
                "Atributos derivados (Vp·Elevación, dist. al centroide)",
                value=True,
                key="t2_deriv",
                help=(
                    "Desactívelo si Vp y Elevación salen casi colineales y el perfil "
                    "muestra estructura lateral que no está en la Vp."
                ),
            )

        st.caption(
            f"Optimizador: {'Optuna (TPE)' if OPTUNA_DISPONIBLE else 'grid manual'} · "
            f"Kriging: {'pykrige' if PYKRIGE_DISPONIBLE else 'motor local propio'}"
        )
        if not model_supports_extrapolation(model_type):
            st.caption(
                f"⚠ `{model_type}` no extrapola: fuera del rango de entrenamiento las "
                "predicciones se aplanan. Compare contra `ridge` si el diagnóstico "
                "de extrapolación sale alto."
            )

    if not objetivo:
        return

    zonas_sel = {asignadas[n]["zona"] for n in objetivo}
    if len(zonas_sel) > 1:
        st.warning(
            f"El bloque mezcla {len(zonas_sel)} zonas ({', '.join(sorted(zonas_sel))}). "
            "Como X e Y entran como atributos, un modelo entrenado sobre sitios "
            "separados por kilómetros extrapola de forma muy inestable al dejar una "
            "línea fuera — sobre todo con `ridge`. Modele **una zona por bloque**, "
            "igual que los bloques Iyotla y Tiquimil de `Vs_Modelling.ipynb`."
        )

    if st.button("🚀 Generar Perfil 2D", type="primary"):
        with st.spinner("Optimizando hiperparámetros y proyectando Vs…"):
            try:
                _generar(
                    objetivo,
                    asignadas,
                    model_type,
                    cv_strategy,
                    int(n_blocks),
                    int(n_trials),
                    kriging_residual,
                    variograma,
                    recortar,
                    float(margen),
                    int(semilla),
                    guardas,
                    float(apalancamiento),
                    interaccion,
                )
            except Exception as exc:
                st.error(f"Error al generar el perfil: {exc}")
                return

    _mostrar_resultados()


def _generar(
    objetivo,
    asignadas,
    model_type,
    cv_strategy,
    n_blocks,
    n_trials,
    kriging_residual,
    variograma,
    recortar,
    margen,
    semilla,
    guardas,
    apalancamiento,
    interaccion,
) -> None:
    entrenamiento = pd.concat(
        [asignadas[n]["puntos_vs"] for n in objetivo], ignore_index=True
    )
    malla = pd.concat(
        [asignadas[n]["df_source"] for n in objetivo], ignore_index=True
    )

    resultado = servicios.proyectar_vs(
        entrenamiento,
        malla,
        model_type,
        cv_strategy,
        n_blocks,
        n_trials,
        kriging_residual,
        variograma,
        recortar,
        margen,
        semilla,
        guardas,
        apalancamiento,
        interaccion,
    )

    st.session_state["resultado_vs"] = resultado
    st.session_state.setdefault("perfiles_2d", {})

    grid = resultado["grid_df_predicho"]
    for nombre in objetivo:
        df_linea = grid.loc[grid["Linea"] == nombre].copy()
        if df_linea.empty:
            continue
        _registrar_perfil(
            nombre,
            df_linea.reset_index(drop=True),
            asignadas[nombre]["topo_linea"],
            "pipeline Vs",
        )


def _mostrar_resultados() -> None:
    resultado = st.session_state.get("resultado_vs")
    if resultado is None:
        return

    st.divider()
    st.markdown("##### Desempeño del modelo (validación cruzada espacial, out-of-fold)")

    met = resultado["metrics"]
    rel = met["rmse_sobre_std_objetivo"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Modelo", resultado["config"].model_type)
    m2.metric("RMSE", f"{met['rmse']:.1f} m/s")
    m3.metric("MAE", f"{met['mae']:.1f} m/s")
    m4.metric("R²", f"{met['r2']:.3f}")
    m5.metric(
        "RMSE / σ(Vs)",
        f"{rel:.2f}",
        help="<0.5 bueno · 0.5–0.8 aceptable · >0.8 débil",
    )
    st.caption(
        f"Calidad: **{calificar_rmse_relativo(rel)}** · "
        f"{met['n_muestras']} muestras · {resultado['descripcion_cv']} · "
        f"correlación Pearson Vp–Vs = "
        f"{resultado['correlacion_vp_vs']:.3f} · "
        f"hiperparámetros: {resultado['best_params']} "
        f"({resultado['diagnostico_optimizacion']['motor']}, "
        f"RMSE CV = {resultado['diagnostico_optimizacion']['mejor_rmse']:.2f})"
        + (
            f" · kriging: {resultado['motor_kriging']}"
            if resultado["motor_kriging"]
            else ""
        )
    )

    for aviso in resultado["avisos"]:
        st.warning(aviso)

    usados = resultado["feature_list"]
    descartados = [c for c in resultado["feature_list_completa"] if c not in usados]
    st.caption(
        f"Atributos usados por el modelo: **{', '.join(usados)}**"
        + (f" · descartados: {', '.join(descartados)}" if descartados else "")
    )
    with st.expander(
        "Identificabilidad de los atributos — por qué se descartó cada uno",
        expanded=bool(descartados),
    ):
        st.caption(
            "`apalancamiento = rango en la malla / (2 × desviación en el "
            "entrenamiento)`. Un valor alto significa que el atributo apenas "
            "varía entre los sondeos pero recorre un rango amplio a lo largo del "
            "perfil: el escalador lo normaliza con una desviación diminuta y su "
            "contribución se amplifica sin control al predecir."
        )
        st.dataframe(
            resultado["reporte_atributos"].round(3),
            use_container_width=True,
            hide_index=True,
        )

    e1, e2 = st.columns(2)
    with e1:
        st.caption("Error por línea excluida (Leave-One-Line-Out)")
        st.dataframe(
            resultado["metricas_por_grupo"].round(2),
            use_container_width=True,
            hide_index=True,
            height=190,
        )
    with e2:
        st.caption(
            "Importancia de variables — si X/Y dominan sobre Vp, el modelo "
            "memoriza posiciones en lugar de la física Vp→Vs"
        )
        st.dataframe(
            resultado["importancia_features"].round(4),
            use_container_width=True,
            hide_index=True,
            height=190,
        )

    if not resultado["diagnostico_extrapolacion"].empty:
        st.caption(
            "Diagnóstico de extrapolación por variable — un porcentaje alto en X/Y es "
            "extrapolación espacial (esperada); en Vp/Elevación es petrofísica, "
            "más riesgosa"
        )
        st.dataframe(
            resultado["diagnostico_extrapolacion"],
            use_container_width=True,
            hide_index=True,
            height=230,
        )


def _panel_perfiles() -> None:
    """
    Catálogo de perfiles 2D disponibles — tanto los que produjo el pipeline de
    Vs como las líneas sintetizadas por kriging.  Es independiente de que haya
    corrido el modelo: una línea sintética se ve aquí en cuanto se genera.
    """
    perfiles = st.session_state.get("perfiles_2d", {})
    if not perfiles:
        return

    resultado = st.session_state.get("resultado_vs")

    st.divider()
    st.markdown("##### Perfiles 2D generados")

    c1, c2 = st.columns([3, 1])
    with c1:
        nombre = st.selectbox(
            "Perfil",
            list(perfiles.keys()),
            key="t2_perfil_sel",
            format_func=lambda n: (
                f"{n}  ·  {perfiles[n].get('origen', 'pipeline Vs')}"
            ),
        )
    entrada = perfiles[nombre]
    df = entrada["df"]
    origen = entrada.get("origen", "pipeline Vs")
    with c2:
        st.write("")
        if st.button("🗑️ Eliminar este perfil", use_container_width=True):
            st.session_state["perfiles_2d"].pop(nombre, None)
            _sincronizar_seleccion_3d()
            _rerun()

    numericas = [
        c
        for c in ["Vs", "Vs_predicho", "Vs_predicho_ML", "Vp"]
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numericas:
        st.warning(f"El perfil «{nombre}» no tiene ninguna variable de velocidad.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Nodos", len(df))
    k2.metric("Origen", origen)
    principal = numericas[0]
    k3.metric(
        f"Rango de {principal}",
        f"{df[principal].min():.0f} – {df[principal].max():.0f} m/s",
    )
    if "flag_extrapolacion_general" in df.columns:
        k4.metric(
            "Extrapolación general",
            f"{100 * df['flag_extrapolacion_general'].mean():.1f} %",
        )

    if "zona_confianza_espacial" in df.columns and resultado is not None:
        reparto = (
            100 * df["zona_confianza_espacial"].value_counts(normalize=True)
        ).round(1)
        st.caption(
            "Zona de confianza espacial (separación típica entre MASW = "
            f"{resultado['escala_referencia_masw']:.1f} m): {reparto.to_dict()}"
        )

    # ------------------------------------------------------------ vista 2D
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        variable = st.selectbox("Variable", numericas, key="t2_var_vista")
    with v2:
        prof = st.number_input(
            "Profundidad del corte (m)", 1.0, 300.0, 30.0, 1.0, key="t2_prof_vista"
        )
    with v3:
        cmap = st.selectbox("Escala", CMAPS_DISPONIBLES, index=0, key="t2_cmap_vista")
    with v4:
        texto_cont = st.text_input("Contornos", value="300, 720", key="t2_cont_vista")

    contornos = []
    for token in texto_cont.split(","):
        token = token.strip()
        if token:
            try:
                contornos.append(float(token))
            except ValueError:
                pass

    col_x = "Xo" if "Xo" in df.columns else "X"
    col_y = "Elevacion" if "Elevacion" in df.columns else "Z"
    try:
        png = servicios.corte_2d_png(
            df,
            col_x,
            col_y,
            variable,
            "linear",
            float(df[variable].min()),
            float(df[variable].max()),
            float(prof),
            cmap,
            tuple(contornos),
            200,
            f"{nombre} — {variable}",
        )
        st.image(png, use_container_width=True)
    except Exception as exc:
        png = None
        st.warning(f"No se pudo dibujar la sección: {exc}")

    with st.expander("Tabla de datos", expanded=False):
        st.dataframe(df.head(300), use_container_width=True, height=260)

    # --------------------------------------------------------- exportación
    base = nombre.replace(" · ", "_").replace(" ", "_")
    x1, x2, x3 = st.columns(3)
    with x1:
        eje_z = st.radio(
            "Eje Z del archivo",
            ["Elevación (msnm)", "Profundidad (m)"],
            index=0,
            key="t2_ejez",
        )
    with x2:
        st.write("")
        st.download_button(
            "⬇️ X, Y, Z, V (CSV)",
            servicios.dataframe_a_csv(
                exportar_xyzv(df, variable, eje_z.startswith("Elevación"))
            ),
            file_name=f"{base}_{variable}_XYZV.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Perfil completo (CSV)",
            servicios.dataframe_a_csv(df),
            file_name=f"{base}_modelo_2d.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with x3:
        st.write("")
        if png is not None:
            st.download_button(
                "⬇️ Sección 2D (PNG, 300 dpi)",
                png,
                file_name=f"Seccion_{base}_{variable}.png",
                mime="image/png",
                use_container_width=True,
            )
        if resultado is not None:
            st.download_button(
                "⬇️ Modelo completo del bloque (CSV)",
                servicios.dataframe_a_csv(resultado["grid_df_predicho"]),
                file_name="modelos_2d_vp_vs.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "⬇️ Artefactos del modelo (joblib)",
                artefactos_a_bytes(resultado),
                file_name="modelo_vs.joblib",
                mime="application/octet-stream",
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Síntesis de línea nueva por kriging 3D
# ---------------------------------------------------------------------------
def _panel_sintetica() -> None:
    asignadas = st.session_state.get("asignadas", {})
    df_topo = st.session_state.get("topografia")
    if not asignadas or df_topo is None:
        return

    st.divider()
    with st.expander(
        "Sintetizar una línea nueva por Kriging 3D  (`synthesize_line` de "
        "`crear_linea_sintetica.ipynb`)",
        expanded=False,
    ):
        st.caption(
            "Para trazas donde no hay modelo de velocidades: se interpola desde las "
            "líneas vecinas con kriging ordinario 3D y anisotropía vertical "
            "(rango_z = rango_xy × factor)."
        )
        s1, s2, s3 = st.columns(3)
        with s1:
            fuentes = st.multiselect(
                "Líneas fuente",
                list(asignadas.keys()),
                default=list(asignadas.keys()),
                key="t2s_fuentes",
            )
            variable = st.selectbox("Variable", ["Vp", "Vs"], key="t2s_var")
        with s2:
            zona_s = st.selectbox(
                "Zona de la línea objetivo",
                espacial.zonas_disponibles(df_topo),
                key="t2s_zona",
            )
            linea_s = st.selectbox(
                "Línea objetivo",
                espacial.lineas_de_zona(df_topo, zona_s),
                key="t2s_linea",
            )
        with s3:
            prof_max = st.number_input(
                "Profundidad máx. (m)", 5.0, 200.0, 30.0, 1.0, key="t2s_prof"
            )
            dz = st.number_input("Paso vertical dz (m)", 0.25, 10.0, 1.0, 0.25, key="t2s_dz")
            aniso = st.number_input(
                "Anisotropía vertical",
                0.005,
                1.0,
                0.05,
                0.005,
                format="%.3f",
                key="t2s_aniso",
            )
            rango = st.number_input(
                "Rango horizontal (m)", 10.0, 5000.0, 500.0, 10.0, key="t2s_rango"
            )

        if st.button("Generar línea sintética") and fuentes:
            topo_obj = espacial.topografia_de_linea(df_topo, zona_s, linea_s)

            partes = []
            for n in fuentes:
                if variable == "Vp":
                    partes.append(asignadas[n]["df_source"])
                else:
                    pf = st.session_state.get("perfiles_2d", {}).get(n)
                    if pf is None:
                        st.error(
                            f"La línea «{n}» aún no tiene Vs 2D. Genere primero el "
                            "perfil 2D."
                        )
                        return
                    partes.append(pf["df"])
            df_source = pd.concat(partes, ignore_index=True)

            with st.spinner("Kriging ordinario 3D…"):
                df_sint, info = servicios.sintetizar_linea_kriging(
                    df_source,
                    topo_obj,
                    variable,
                    st.session_state.get("t2_var", "spherical"),
                    float(aniso),
                    float(rango),
                    float(prof_max),
                    float(dz),
                )
            # El nombre del perfil es el de la «Línea objetivo», tal cual.
            # Si esa línea ya se sintetizó en la otra variable, ambas se
            # acumulan en un único perfil en vez de sobrescribirse.
            nombre = str(linea_s)
            aviso_fusion = _fusionar_sintetica(nombre, df_sint, variable, topo_obj)

            df_guardado = st.session_state["perfiles_2d"][nombre]["df"]
            variables = ", ".join(c for c in ("Vp", "Vs") if c in df_guardado.columns)
            st.session_state["aviso_sintetica"] = (
                f"Línea sintética «{nombre}» — {variable} generada con "
                f"{info['n_objetivo']} nodos a partir de {info['n_fuente']} puntos "
                f"fuente (motor: {info['motor_kriging']}). "
                f"El perfil contiene ahora: {variables}."
            )
            st.session_state["aviso_fusion"] = aviso_fusion
            _rerun()


# ---------------------------------------------------------------------------
def render() -> None:
    st.subheader("Topografía y predicción espacial de Vs")
    st.caption(
        "Georreferenciación de los modelos geofísicos y proyección de Vs sobre "
        "las mallas 2D de Vp."
    )

    df_topo = _catalogo_topografia()

    if df_topo is None:
        st.info(
            "Cargue el catálogo topográfico para continuar. Acepta separadores por "
            "tabulador, coma o espacios, con o sin encabezado."
        )
    else:
        st.session_state["topografia"] = df_topo
        st.success(
            f"{len(df_topo)} estaciones · {df_topo['Zona_Nombre'].nunique()} zonas · "
            f"{df_topo.groupby(['Zona_Nombre', 'Linea']).ngroups} líneas."
        )

        aviso = st.session_state.pop("aviso_sintetica", None)
        if aviso:
            st.success(aviso)
        aviso_fusion = st.session_state.pop("aviso_fusion", None)
        if aviso_fusion:
            st.warning(aviso_fusion)

        st.divider()
        _panel_agregar(df_topo)
        st.divider()
        _panel_prediccion()
        _panel_sintetica()

    # El catálogo de perfiles va al final y no depende ni de la topografía ni
    # de que haya corrido el pipeline: una línea sintética aparece aquí en
    # cuanto se genera.
    _panel_perfiles()
