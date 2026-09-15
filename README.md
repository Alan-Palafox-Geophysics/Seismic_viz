# TRS + MASW — Integración, modelado y visualización

Aplicación Streamlit para procesar, modelar y visualizar datos de **Sísmica de
Refracción (TRS)** y **MASW**. Toda la lógica se extrajo de los notebooks y
módulos del proyecto y se reorganizó en funciones modulares y cacheables.

## Instalación

Instale en un entorno **limpio**. Mezclar `pip` sobre un entorno conda que ya
traía matplotlib, scipy o scikit-learn deja dos cadenas de binarios y rompe la
ABI de NumPy.

```
conda create -n trs python=3.11 -y
conda activate trs
pip install -r requirements.txt

python check_entorno.py     # verifica versiones y conflictos
streamlit run app.py
```

### Si aparece `ImportError: numpy.core.multiarray failed to import`

No es un fallo de la aplicación. Significa que un paquete compilado
(matplotlib, scipy, scikit-learn, pykrige…) fue construido contra una versión
mayor de NumPy distinta de la instalada. `python check_entorno.py` identifica
cuáles y sugiere el comando exacto. Las dos salidas posibles:

```
# Recomendado: entorno limpio
conda create -n trs python=3.11 -y && conda activate trs
pip install -r requirements.txt

# Reparar el entorno actual, sin mezclar gestores
pip install --force-reinstall --no-cache-dir numpy pandas scipy scikit-learn matplotlib
```

Si alguna otra herramienta de ese entorno exige NumPy 1.x, fije esa rama y
reconstruya el resto contra ella:

```
pip install --force-reinstall --no-cache-dir "numpy<2" pandas scipy scikit-learn matplotlib
```

La aplicación funciona igual con NumPy 1.x y 2.x; lo que no tolera es tener
binarios de ambas ramas conviviendo.

---

## Estructura

```
masw_trs_app/
├── app.py                      Punto de entrada, estado de sesión, barra lateral
├── check_entorno.py            Diagnóstico de dependencias y conflictos de ABI
├── requirements.txt
├── core/                       Núcleo independiente de Streamlit
│   ├── io_utils.py             Lectores tolerantes (TRS .dat, MASW, topografía, modelos 2D)
│   ├── fisica.py               Corrección topográfica, colapso 1D por IDW, módulos elásticos
│   ├── ml.py                   Clasificación no supervisada PCA + K-Means
│   ├── prediccion.py           Pipeline de proyección de Vs + síntesis por kriging 3D
│   ├── kriging.py              Kriging ordinario 2D y 3D (pykrige + motor propio)
│   ├── espacial.py             Cruce Xo ↔ topografía, emplazamiento MASW / SEV
│   ├── plots_1d.py             Figura de dos paneles del Tab 1
│   ├── plots_2d.py             Cortes estáticos recortados al relieve
│   ├── plots_3d.py             Cortinas 3D, contornos e imagen base
│   └── servicios.py            Capa @st.cache_data
├── tabs/
│   ├── tab1_integracion.py
│   ├── tab2_topografia.py
│   └── tab3_visualizacion.py
└── datos_ejemplo/              Los Cuates (TRS, MASW) + catálogo topográfico
```

El núcleo no importa Streamlit: se puede usar desde un notebook o un script por
lotes. La caché vive únicamente en `core/servicios.py`.

---

## Tab 1 — Integración 1D/2D y módulos elásticos

**Entradas.** Modelo 2D de Vp (`X`, `Elevación`, `Vp`) y modelo 1D de Vs
(`Profundidad`, `Vs`). La Vp se detecta en km/s o m/s y se normaliza a m/s.

**Corrección topográfica.** Cada traza vertical se referencia a la topografía en
su propia abscisa, de modo que todo el perfil queda en profundidad relativa con
`z = 0` en el terreno (`modo = 'por_columna'`). La alternativa `'global'` resta
la cota máxima del tendido completo.

**Colapso 1D por IDW.** A diferencia de `procesar_TRS_a_MASW`, que extraía
únicamente la columna central, aquí participan **todas** las trazas con peso

```
w = 1 / (|x − x_centro| + s)^p
```

El término de suavizado `s` (por defecto, la mediana del espaciamiento entre
columnas) es lo que mantiene esto como un promedio: sin él, una traza que caiga
exactamente en el centro tendría peso infinito y el resultado colapsaría a esa
única columna. Con `p` alto el comportamiento tiende al del notebook original.
Cada columna se interpola a la malla común **sin extrapolar**, de manera que una
traza corta no aporta velocidad inventada al fondo del perfil.

**Interpolación y física.** El perfil 1D se lleva a las profundidades exactas del
MASW con `interp1d` lineal y extrapolación, y se calculan los módulos elásticos
con las ecuaciones textuales del notebook, incluidos los factores de escala:

| Parámetro | Ecuación |
|---|---|
| Densidad ρ | `1.2475 + 0.399·(Vp/1000) − 0.026·(Vp/1000)²` |
| Poisson ν | `(r² − 2) / (2(r² − 1))`, con `r = Vp/Vs` |
| Corte G = μ | `ρ·Vs² / 1e5` |
| Young E | `2G(1 + ν)` |
| Bulk K | `E / (3(1 − 2ν))` |
| Lamé λ | `(ρ·Vp² − 2ρ·Vs²) / 1e5` |
| Factor prop. | `ν / ((1 + ν)(1 − 2ν))` |
| Comprobaciones | `√((λ + 2G)/ρ)` y `E·fp` |

**Clasificación.** `StandardScaler → PCA(0.90) → K-Means`, con *k* óptimo por
Silhouette y reordenamiento de clases por Vs medio (clase 0 = material más
blando).

> **Corrección respecto al notebook.** El original guardaba `mejor_k = k + 1`
> dentro del bucle de búsqueda, de modo que el *k* entrenado no era el que
> maximizaba la silueta (por eso reportaba 3 cuando el máximo estaba en 2). Aquí
> se corrige, y la casilla *«Replicar el k+1 del notebook original»* permite
> reproducir resultados históricos.

**Interfaz.** Dos subplots con eje Y compartido, *radio button* global para
invertir la profundidad, selector de módulo en el panel derecho (por defecto
Coeficiente de Poisson con límite X de 0 a 5, editable) y descarga de resultados
y del perfil 1D de Vp.

---

## Tab 2 — Topografía y predicción espacial de Vs

**Catálogo topográfico.** `Zona_Num, Linea_Num, Zona_Nombre, Linea, ID, Xo, X, Y, Z`,
con menús desplegables para zona y línea.

**Asignación espacial** (botón «Agregar»), siguiendo `Combine_Data_Topo`:

- El modelo 2D se cruza con la topografía por `Xo` — modo `exacto` (el *inner
  join* del notebook) o `cercano` con tolerancia, que conserva las columnas
  intermedias de la malla de inversión.
- **MASW**: se ancla a la coordenada central del tendido TRS, interpolada sobre
  las estaciones reales.
- **SEV / VES**: se ancla a la coincidencia **exacta** de `Xo` en la lista de
  estaciones; si esa abscisa no existe, la operación falla con el listado de
  abscisas disponibles en vez de aproximar en silencio.

**Pipeline predictivo** (botón «Generar Perfil 2D») — port fiel de
`vs_projection_pipeline.py` v2:

1. *Sanity check* geofísico: correlación Pearson Vp–Vs, con aviso si cae de 0.5.
2. Atributos: `X, Y, Elevacion, Vp` + `Vp_x_Elevacion` y `dist_centroide`, con el
   centroide calculado en el entrenamiento y reutilizado en la malla.
3. Optimización de hiperparámetros sobre el RMSE de validación cruzada espacial
   (`GroupKFold` por línea = Leave-One-Line-Out). Usa Optuna (TPE) si está
   instalado; si no, un grid manual.
4. Métricas honestas *out-of-fold*: RMSE, MAE, R² y **RMSE/σ(Vs)** con su lectura
   cualitativa (<0.5 bueno, 0.5–0.8 aceptable, >0.8 débil), más el error por
   línea excluida.
5. **Regression Kriging**: los residuos *out-of-fold* se interpolan en planta
   (X, Y) y se suman a la predicción, por lo que la corrección es constante a lo
   largo de cada vertical.
6. Recorte físico al rango de Vs observado más un margen porcentual.
7. Diagnósticos: extrapolación **por variable** — que separa la espacial (X/Y,
   esperada) de la petrofísica (Vp/Elevación, más riesgosa) —, zona de confianza
   continua por distancia al MASW más cercano, e importancia de variables como
   detector de sobreajuste espacial.

Modelos disponibles: `ridge` (recomendado, **sí extrapola**),
`gradient_boosting` y `random_forest` (los árboles se aplanan fuera del rango de
entrenamiento; la interfaz lo advierte).

**Síntesis por Kriging 3D.** Panel aparte con la traducción de `synthesize_line`:
malla vertical desde la cota de cada estación hasta `−prof_max` con paso `dz`, e
interpolación por kriging ordinario 3D con anisotropía vertical
(`rango_z = rango_xy × factor`, 0.05 por defecto). Es la herramienta para trazas
donde no hay modelo de velocidades.

**Vp y Vs se sintetizan en un solo proceso**, sobre la misma malla y con los
mismos parámetros, de modo que el perfil sintético queda completo de una vez y
no hay forma de que una pasada sobrescriba a la otra. La Vp se interpola desde
las mallas 2D de las líneas fuente y la Vs desde sus perfiles 2D ya modelados,
así que conviene correr antes «Generar Perfil 2D»; si ninguna fuente tiene Vs
todavía, se sintetiza sólo Vp y se avisa. El perfil se guarda con el nombre de
la «Línea objetivo» y aparece de inmediato en el catálogo del Tab 2 y en el
Tab 3.

**Exportación.** Lista desplegable de perfiles generados, CSV en formato
`X, Y, Z, V` (con `Z` en elevación o profundidad), perfil completo, modelo del
bloque y artefactos `joblib` del modelo entrenado.

### Identificabilidad de los atributos espaciales

El pipeline original usa `X`, `Y` y `dist_centroide` como atributos. Eso funciona
cuando los MASW están genuinamente repartidos por el área — como en Iyotla, con
cuatro sondeos separados decenas de metros en ambas direcciones. **No funciona
cuando hay un MASW por línea, ubicado en el centro del tendido.**

El mecanismo: si dos líneas vecinas corren casi paralelas, la `Y` de sus dos
sondeos difiere en centímetros, mientras que a lo largo de cada perfil la `Y`
recorre decenas de metros. El `StandardScaler` normaliza `Y` con esa desviación
diminuta, de modo que a lo largo del perfil el valor escalado alcanza ±58. Un
coeficiente que en la tabla de importancias parece insignificante queda
multiplicado por eso, y cualquier diferencia real de Vs entre las dos líneas se
atribuye íntegramente a la posición y se extrapola sobre toda la sección. El
resultado es una **rampa lateral** en lugar de un perfil estratificado.

Medido con Los Cuates L1 + L2, comparando la desviación de Vs entre bandas de
elevación contra la desviación entre bandas de `Xo` (la Vp medida da 9.5):

| Diferencia entre los dos MASW | vertical / lateral | Rango de Vs |
|---|---|---|
| 0 % | 9.2 | 201 – 582 |
| 5 % | **0.52** | −93 – 839 |
| 15 % | **0.18** | −731 – 1353 |
| 30 % | **0.09** | −1687 – 2125 |

Con un 5 % de diferencia la variación lateral ya supera a la vertical.

La aplicación aplica dos guardas antes de entrenar, y muestra la tabla completa
en *«Identificabilidad de los atributos»*:

1. **Identificabilidad espacial** — con `n` atributos de posición hacen falta al
   menos `n + 2` sitios distintos de MASW. Con dos sitios, un plano queda
   determinado exactamente y no hay residuo del que aprender.
2. **Apalancamiento** — `rango en la malla / (2 × desviación en el entrenamiento)`.
   Por encima del umbral (5 por defecto) el atributo se descarta.

`Vp` y `Elevación` nunca se descartan: son la física que el modelo debe
aprender. La estructura espacial entra entonces por el kriging de residuos, que
es donde corresponde. Con las guardas activas el resultado deja de depender de
la diferencia entre MASW (vertical/lateral = 3.6 en los cuatro casos de la
tabla).

Puede desactivarlas para reproducir el comportamiento anterior.

### Colinealidad Vp – Elevación

En un sondeo 1D la velocidad crece de forma casi monótona con la profundidad: en
Los Cuates, `corr(Vp, Elevacion) = −0.995`. El modelo no puede separar «Vs sube
porque Vp sube» de «Vs sube porque bajamos». Con `alpha` pequeño, Ridge reparte
la contribución en coeficientes grandes y opuestos (−833 para Elevación, −331
para Vp, +398 para la interacción), y ese reparto arbitrario reaparece como
variación lateral al aplicarse a la malla 2D, donde Vp **sí** varía a
profundidad fija.

La aplicación lo detecta y lo avisa. Si el perfil muestra estructura lateral que
no está en la Vp, suba la regularización o desactive *«Atributos derivados»*.

### Una zona por bloque

`X` e `Y` entran como atributos en coordenadas UTM absolutas. Entrenar con líneas
separadas por kilómetros hace que Leave-One-Line-Out extrapole de forma muy
inestable — con `ridge` el error puede crecer dos órdenes de magnitud. Modele
**una zona por bloque**, como los bloques Iyotla y Tiquimil de
`Vs_Modelling.ipynb`. La aplicación lo advierte al detectar la mezcla.

---

## Tab 3 — Visualización 3D y mapas base

- Cortinas 3D (`mode='surface'`) construidas sobre una **malla que sigue el
  terreno**: para cada posición a lo largo de la línea la vertical arranca en la
  topografía de esa abscisa y baja la profundidad indicada. Antes la malla era
  un rectángulo `z.min()…z.max()` rellenado por vecino más cercano, que pintaba
  material por encima del terreno; ahora el recorte es el mismo criterio que
  usa el corte 2D. Por debajo del dato se abre hueco en vez de extrapolar.
- **Misma rampa de color que los cortes 2D.** El selector del 3D ofrece la misma
  lista que el 2D, incluida la rampa personalizada *morado → … → rojo oscuro*,
  que se traduce a una `colorscale` de Plotly con `colormap_a_plotly`. La
  profundidad del corte es un único control compartido por la cortina 3D y la
  sección estática, de modo que ambas son la misma figura vista de dos maneras.
- Menús de cámara y visibilidad, y contornos parametrizables.
- **Contornos parametrizables**: las isolíneas se calculan sobre la malla
  distancia–elevación de la cortina y se proyectan al trazado real, de modo que
  los mismos valores (p. ej. 300 y 720 m/s) aparecen en 3D y en los cortes 2D.
- **Imagen base**: GeoTIFF con extracción automática de su extensión (rasterio,
  o etiquetas GeoTIFF vía Pillow) y JPG/PNG con esquinas inferior-izquierda y
  superior-derecha declaradas. Se indexa a una paleta adaptativa de 256 colores
  para renderizarla como textura sobre una `go.Surface`.
- **Exportación**: HTML interactivo autocontenido y cortes estáticos 2D a 300 dpi
  con la estética de `plot_2d_puentes.ipynb` — recorte al relieve por arriba y a
  la profundidad indicada por abajo, 256 niveles continuos y la rampa
  personalizada *morado → azul → cian → verde → amarillo → naranja → rojo →
  rojo oscuro*.

---

## Dependencias opcionales

La aplicación funciona sin ellas, con degradación controlada y anunciada en la
interfaz:

| Paquete | Si falta |
|---|---|
| `optuna` | Búsqueda de hiperparámetros en grid manual |
| `pykrige` | Motor de kriging propio (vecindad local con KD-tree) |
| `rasterio` | Georreferencia del GeoTIFF por etiquetas Pillow, o esquinas manuales |

El motor de kriging propio se usa además **por diseño** cuando hay más de ~600
puntos fuente en 3D: pykrige resuelve el sistema global (matriz *n×n*) y escala
mal, mientras que la vecindad local da el mismo resultado práctico en segundos.

---

## Datos de ejemplo

`datos_ejemplo/` incluye `Los_cuates_1.dat` (TRS 2D, 41 columnas × 22 nodos, Vp
en km/s), `Cerro_LosCuates_1_promedio.csv` (MASW, 20 profundidades) y
`Topografia.txt` (432 estaciones, 7 zonas, 18 líneas).
