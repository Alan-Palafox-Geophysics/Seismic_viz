# Documentación técnica

## Integración TRS + MASW, modelado de Vs y visualización 3D

**Sísmica de refracción · MASW · Parámetros elásticos · Geoestadística · Aprendizaje supervisado**

---

## Resumen ejecutivo

Esta aplicación convierte dos productos de campo independientes —un modelo **2D de velocidad de onda P** obtenido por tomografía de refracción sísmica (TRS) y un modelo **1D de velocidad de onda S** obtenido por análisis multicanal de ondas superficiales (MASW)— en un modelo geotécnico continuo del subsuelo: perfiles verticales de parámetros elásticos, secciones 2D de Vs a lo largo de cada tendido, líneas sintéticas en trazas no levantadas, y visualización tridimensional georreferenciada.

El problema de fondo es una **asimetría de muestreo**. La refracción entrega cobertura lateral densa pero no resuelve la onda de corte; el MASW resuelve la onda de corte pero solo bajo un punto. Toda la aplicación se organiza alrededor de esa asimetría: primero colapsa la información lateral de Vp a la vertical donde existe Vs, ahí calcula la física que requiere ambas velocidades, y después usa esa relación aprendida para devolver Vs a toda la extensión lateral que la Vp sí cubre.

| Etapa | Entrada | Técnica | Salida |
|---|---|---|---|
| 1. Perfil 1D de Vp | Malla 2D de Vp | Corrección topográfica + IDW ponderado | Vp(z) en la vertical del MASW |
| 2. Integración | Vp(z), Vs(z) | Interpolación lineal a nodos MASW | Tabla acoplada Vp–Vs |
| 3. Parámetros elásticos | Vp, Vs, ρ | Elasticidad lineal isótropa | ν, G, E, K, λ, μ |
| 4. Clasificación | Módulos elásticos | Estandarización → PCA → K-Means | Unidades geofísicas |
| 5. Georreferenciación | Topografía + geofísica | Cruce por abscisa Xo | Modelo en coordenadas UTM |
| 6. Proyección de Vs | Vp 2D + Vs 1D | Regresión + Regression Kriging | Sección 2D de Vs |
| 7. Línea sintética | Líneas vecinas | Kriging ordinario 3D anisótropo | Traza nueva en Vp y Vs |
| 8. Visualización | Modelos 2D/3D | Recorte al relieve, cortinas 3D | Secciones e interactivos |

---

## 1. Fundamento físico

### 1.1 Las dos ondas de cuerpo

En un medio elástico, homogéneo, isótropo e infinito, la ecuación de Navier–Cauchy admite exactamente dos soluciones de onda de cuerpo, cuyas velocidades dependen solo de las constantes elásticas y la densidad:

```
Vp = √( (λ + 2μ) / ρ ) = √( (K + 4μ/3) / ρ )      onda P, compresional
Vs = √( μ / ρ )                                     onda S, de corte
```

La distinción es la clave interpretativa del método. La onda P comprime el medio y **se propaga por el esqueleto y por el fluido de poro**; la onda S lo cizalla y, como los fluidos no tienen rigidez al corte (μ = 0), **solo se propaga por el esqueleto sólido**. De ahí que:

- **Vs es el descriptor mecánico honesto del terreno.** No la altera la saturación. Por eso las normativas sísmicas (NEHRP, Eurocódigo 8, y en México el MDOC-CFE) clasifican el sitio por Vs30 y no por Vp.
- **Vp sí se altera con la saturación.** Por debajo del nivel freático la Vp salta a ≈ 1500 m/s (la velocidad del sonido en agua) casi con independencia del material. Un contraste fuerte de Vp sin contraste de Vs suele ser el nivel freático, no un cambio litológico.

Esta complementariedad es la que justifica todo el flujo: una sola velocidad no determina el estado mecánico; las dos juntas sí.

### 1.2 Relación Vp/Vs y coeficiente de Poisson

Definiendo la razón **r = Vp/Vs**, la relación con el coeficiente de Poisson se despeja directamente de las dos ecuaciones anteriores:

```
ν = (r² − 2) / ( 2·(r² − 1) )
```

Es una función monótona creciente de r, con significado geológico directo:

| r = Vp/Vs | ν | Interpretación típica |
|---|---|---|
| 1.41 (√2) | 0.00 | Límite teórico inferior práctico |
| 1.73 (√3) | 0.25 | Sólido de Poisson: roca cristalina sana |
| 2.0 | 0.33 | Roca fracturada, suelo compacto |
| 3.0 | 0.44 | Suelo blando, aluvión |
| > 4.0 | > 0.47 | Saturación, material muy blando |
| → ∞ | → 0.50 | Fluido (incompresible al corte) |

Los límites termodinámicos son −1 < ν < 0.5. En materiales geológicos reales el rango útil es 0.15–0.49. **Un ν fuera de ese rango, o negativo, es un diagnóstico de datos inconsistentes**, no un hallazgo: normalmente significa que Vp y Vs no corresponden al mismo punto del subsuelo, o que r < √2, lo cual es físicamente imposible en un sólido isótropo.

Esta comprobación es la razón de que la aplicación sea tan cuidadosa con la etapa 1: si la Vp que se acopla a la Vs no viene de la misma vertical, el ν resultante carece de sentido físico.

---

## 2. Extracción del perfil 1D de Vp

> **Esta es la etapa crítica de todo el flujo.** Todos los parámetros elásticos dependen del acoplamiento correcto entre una Vp y una Vs que representen el mismo volumen de roca. Un error aquí se propaga, amplificado y sin aviso, a todo lo demás.

### 2.1 Planteamiento del problema

El modelo de refracción es una nube de nodos `(X, Elevación, Vp)`: una malla 2D que cubre todo el tendido, típicamente varias decenas de columnas verticales separadas 2–3 m. El modelo MASW es un perfil `(Profundidad, Vs)` que representa **un solo punto**, convencionalmente el centro del tendido, porque el método promedia las propiedades bajo la extensión del arreglo receptor.

Hay que reducir la malla 2D a una sola columna representativa de ese punto. Existen tres estrategias, y la elección no es inocua:

| Estrategia | Qué hace | Problema |
|---|---|---|
| Columna central | Toma la traza más cercana al centro | Hereda todo el ruido de inversión de una sola traza |
| Promedio aritmético | Promedia todas las columnas | Diluye con material que el MASW nunca vio |
| **IDW (implementada)** | Pondera por cercanía al centro | Equilibra representatividad y estabilidad |

El flujo original tomaba la columna central. La aplicación implementa la ponderación por distancia inversa, que es más coherente con la física del MASW: el método **es** un promedio espacial bajo el arreglo, con sensibilidad decreciente hacia los extremos.

### 2.2 Corrección topográfica

Antes de promediar hay que resolver un problema de referencia. Los nodos del modelo 2D están en **elevación absoluta**, pero el MASW está en **profundidad bajo el terreno**. Si el tendido tiene relieve, promediar por elevación mezcla nodos que están a profundidades muy distintas bajo la superficie: el nodo a cota 10 m es superficial donde el terreno está a 10.5 m, pero está 2 m enterrado donde el terreno está a 12 m.

La corrección referencia cada traza vertical a **la topografía de su propia abscisa**:

```
z_sup(x) = máx( Elevación | X = x )        techo del dato en esa columna
Profundidad(x, e) = z_sup(x) − e
```

Con esto el relieve queda "aplanado" y todas las trazas comparten el origen z = 0 en el terreno. Es el modo `por_columna`, el recomendado. El modo alternativo `global` resta la cota máxima de todo el tendido y conserva el relieve como profundidad aparente; solo tiene sentido en terreno prácticamente horizontal.

**Regla geológica implícita:** las unidades litológicas y los frentes de meteorización tienden a ser subparalelos a la topografía en los primeros metros. Referenciar a la superficie local, no a un datum arbitrario, es lo que hace comparables nodos de columnas distintas.

### 2.3 Ponderación por distancia inversa

El centro del tendido se define como el punto medio del recorrido de abscisas:

```
x_centro = ( mín(X) + máx(X) ) / 2
```

Cada columna recibe un peso que decae con su distancia horizontal a ese centro. La formulación es la de **Shepard (1968)**, con un término de suavizado:

```
d_i = | x_i − x_centro |

w_i = 1 / ( d_i + s )^p
```

**Sobre el exponente p.** Controla cuán rápido se pierde influencia con la distancia. Con p = 0 todas las trazas pesan igual (promedio aritmético); con p alto el resultado converge a la columna central, reproduciendo el comportamiento del flujo original. El valor por defecto **p = 2** es el estándar en interpolación espacial y equivale a una ponderación por el inverso del cuadrado, análoga al decaimiento de una influencia puntual en dos dimensiones.

**Sobre el suavizado s — y por qué no es opcional.** La formulación clásica de Shepard usa `w = 1/d^p`. Con ella, una columna que caiga **exactamente** sobre el centro del tendido tiene d = 0 y peso infinito: el "promedio ponderado" colapsa a esa única traza y todas las demás se anulan. No es un caso raro — ocurre siempre que el número de columnas es impar o que el centro coincide con una estaca. El término aditivo `s` acota el peso máximo y preserva el carácter de promedio. Su valor por defecto es **la mediana del espaciamiento entre columnas**, que es la escala natural del problema: a distancia cero el peso equivale al de una columna vecina inmediata, no a infinito.

Opcionalmente se puede imponer un **radio de influencia** que excluya las columnas más allá de cierta distancia, replicando el hecho de que el MASW no tiene sensibilidad fuera de la longitud de su arreglo.

### 2.4 Malla común y regla de no extrapolación

El promedio se evalúa sobre una malla regular de profundidad de `n_nodos` = 200 puntos, entre 0 y una profundidad máxima que por defecto es **la mediana de las profundidades máximas por columna**. La mediana, y no el máximo, porque las trazas de la tomografía tienen penetración desigual: usar el máximo haría que el fondo del perfil dependiera de una sola columna afortunada.

Cada columna se interpola linealmente a esa malla **sin extrapolar**. Donde una traza no tiene dato, no aporta peso:

```
Vp(z) = Σ wᵢ · Vpᵢ(z) / Σ wᵢ      solo sobre las columnas con dato en z
```

Esta restricción es deliberada y tiene una justificación de método: en tomografía de refracción, la resolución se degrada con la profundidad porque decrece la cobertura de rayos. Extrapolar una traza corta hacia abajo equivale a inventar velocidad en una zona que ningún rayo iluminó. El perfil devuelve, junto a Vp, el número de trazas `N_trazas` que contribuyeron a cada nodo y el `Peso_total` acumulado: **una caída brusca de N_trazas marca el punto donde el perfil deja de estar bien soportado**.

### 2.5 Diagnóstico entregado

| Indicador | Lectura |
|---|---|
| `x_centro` | Abscisa usada como referencia |
| `n_columnas_usadas` | Trazas que participaron |
| `frac_peso_columna_central` | Fracción del peso total que aporta la traza más cercana al centro. Cerca de 1 indica colapso a una sola columna |
| `N_trazas(z)` | Soporte del promedio a cada profundidad |

---

## 3. Acoplamiento Vp–Vs

El perfil 1D de Vp se lleva a las profundidades **exactas** del modelo MASW mediante interpolación lineal por tramos, con extrapolación opcional en los extremos:

```
Vp_en_MASW(z) = interp1d( z_perfil, Vp_perfil, kind='linear' )( z_MASW )
```

Se eliminan previamente las profundidades duplicadas (redondeo a 4 decimales) porque la interpolación exige abscisas estrictamente crecientes.

La discretización del MASW no es uniforme: los modelos de inversión de ondas superficiales usan capas de espesor creciente con la profundidad, porque la resolución del método decae con la longitud de onda. Respetar esas profundidades exactas —en lugar de remuestrear la Vs— evita introducir suavizado artificial en el modelo de corte, que es el que más pesa en la interpretación geotécnica.

Los nodos del MASW que caen fuera del rango del perfil de Vp se marcan con la bandera `Fuera_de_rango`. **Un módulo elástico calculado sobre un nodo extrapolado debe tratarse como orientativo**, porque su Vp no proviene de una medición sino de la prolongación lineal de la tendencia.

---

## 4. Parámetros elásticos

### 4.1 Densidad

La densidad no se mide en campo; se estima a partir de Vp mediante una relación empírica. La aplicación usa un polinomio de segundo grado con Vp en km/s y ρ en t/m³:

```
ρ = 1.2475 + 0.399·(Vp/1000) − 0.026·(Vp/1000)²
```

Esta relación pertenece a la familia de las correlaciones velocidad–densidad, cuyo exponente más conocido es el de **Gardner et al. (1974)**, `ρ = 0.31·Vp^0.25`, calibrado para rocas sedimentarias a profundidad de exploración petrolera. La forma polinómica empleada aquí está calibrada para el rango de baja velocidad de la geotecnia somera (600–3000 m/s), donde la ley potencial de Gardner subestima sistemáticamente.

Comportamiento y validez:

| Vp (m/s) | ρ (t/m³) | Material |
|---|---|---|
| 600 | 1.48 | Suelo suelto, relleno |
| 1000 | 1.62 | Suelo compacto |
| 2000 | 1.94 | Roca alterada |
| 3000 | 2.21 | Roca sana fracturada |

La derivada se anula en Vp = 7.67 km/s, muy por encima del rango de trabajo, de modo que la relación es **monótona creciente en todo el dominio de aplicación** y no produce inversiones espurias de densidad. Aun así, sigue siendo una estimación: si se dispone de densidades de laboratorio, la aplicación permite fijar un valor constante en su lugar.

### 4.2 Ecuaciones de la elasticidad lineal isótropa

Un sólido elástico lineal e isótropo queda descrito por **dos** constantes independientes. Todas las demás se derivan. Conocidas Vp, Vs y ρ:

| Parámetro | Ecuación | Significado mecánico |
|---|---|---|
| Coeficiente de Poisson | `ν = (r² − 2) / (2(r² − 1))`, r = Vp/Vs | Deformación transversal por unidad de deformación axial |
| Módulo de corte (G = μ) | `G = ρ·Vs²` | Resistencia a la distorsión sin cambio de volumen |
| Módulo de Young | `E = 2G(1 + ν)` | Rigidez axial bajo carga uniaxial |
| Módulo de Bulk | `K = E / (3(1 − 2ν))` | Resistencia al cambio de volumen bajo presión hidrostática |
| 1ª constante de Lamé | `λ = ρ·Vp² − 2ρ·Vs²` | Constante de acoplamiento volumétrico |
| 2ª constante de Lamé | `μ = ρ·Vs²` | Idéntica a G |
| Factor de proporcionalidad | `fp = ν / ((1 + ν)(1 − 2ν))` | Relación λ = E·fp |

**Nota sobre el dominio de deformación.** Estos son módulos **dinámicos de pequeña deformación** (ε < 10⁻⁵), medidos por el paso de una onda elástica. No son los módulos estáticos de un ensayo de laboratorio: los dinámicos son sistemáticamente mayores, típicamente por un factor de 2 a 10 en suelos, porque el suelo se comporta de forma más rígida a deformaciones muy pequeñas. La conversión a módulos de diseño requiere curvas de degradación `G/G₀` propias del material y del nivel de deformación esperado. **Usar estos valores directamente como módulos de diseño estructural es un error de método.**

### 4.3 Unidades

Los módulos se calculan con ρ en t/m³ y V en m/s, dividiendo por 10⁵. El resultado queda en unidades de **10⁸ Pa**:

```
Valor en la tabla × 100 = MPa
Valor en la tabla × 0.1  = GPa
```

Ejemplo de verificación: Vs = 550 m/s con ρ = 1.9 t/m³ da G = 5.75 en la tabla, es decir **575 MPa**, coherente con un material rígido pero no rocoso.

### 4.4 Comprobaciones internas

Se calculan dos columnas de control que permiten detectar inconsistencias numéricas:

- **`Check_Vp`** reconstruye Vp a partir de λ, G y ρ. Por el factor de escala aplicado, se cumple `Check_Vp × √10⁵ = Check_Vp × 316.23 = Vp` en m/s. Cualquier desviación indica corrupción de datos.
- **`Check_Lambda`** verifica la identidad `λ = E · fp`, que debe cumplirse exactamente por construcción algebraica.

---

## 5. Clasificación geofísica no supervisada

Los módulos elásticos están fuertemente correlacionados entre sí: todos derivan de las mismas dos velocidades. Clasificar directamente sobre ellos daría un peso desproporcionado a la dimensión redundante. El procedimiento implementado resuelve esa multicolinealidad en tres pasos.

**1. Estandarización.** Cada atributo se lleva a media 0 y varianza 1. Es obligatorio antes de cualquier método basado en distancias: sin ello, Vp (orden 10³) dominaría sobre ν (orden 10⁻¹) por pura escala numérica, no por contenido informativo.

**2. Análisis de componentes principales.** Se retienen las componentes que explican el **90 % de la varianza acumulada**. El PCA rota el espacio a ejes ortogonales ordenados por varianza, eliminando la redundancia. En la práctica, nueve atributos elásticos se reducen típicamente a **dos componentes** que explican más del 99 % — confirmación cuantitativa de que la información independiente del sistema es, esencialmente, bidimensional (las dos constantes elásticas independientes de la teoría).

**3. K-Means con selección objetiva de k.** Se evalúa k desde 2 hasta el máximo indicado y se elige el que maximiza el **coeficiente de silueta** (Rousseeuw, 1987):

```
s(i) = ( b(i) − a(i) ) / máx( a(i), b(i) )
```

donde a(i) es la distancia media al propio grupo y b(i) la distancia media al grupo vecino más próximo. Varía entre −1 y 1; por encima de 0.5 la estructura de grupos se considera razonable.

Finalmente las clases se **reordenan por Vs media ascendente**, de modo que la clase 0 corresponde siempre al material más blando y la última al más rígido. Esto convierte una etiqueta arbitraria del algoritmo en una secuencia con significado estratigráfico directo.

> **Corrección respecto al flujo original.** El código de referencia guardaba `mejor_k = k + 1` dentro del bucle de búsqueda, de modo que el k finalmente entrenado no era el que maximizaba la silueta (reportaba 3 cuando el máximo estaba en 2). La aplicación lo corrige, y ofrece una casilla para reproducir el comportamiento histórico cuando se necesite comparar con resultados anteriores.

---

## 6. Georreferenciación y asignación espacial

El catálogo topográfico tiene el esquema `Zona_Num, Linea_Num, Zona_Nombre, Linea, ID, Xo, X, Y, Z`, donde `Xo` es la abscisa local del tendido y `(X, Y, Z)` las coordenadas UTM y la cota de cada estaca.

**Cruce del modelo 2D con la topografía.** Se empareja por la abscisa `Xo`, en dos modos:

- **exacto** — reproduce el *inner join* del flujo original: solo sobreviven las abscisas presentes en ambos archivos. Conservador y trazable.
- **cercano** — asocia cada columna de inversión a la estaca más próxima dentro de una tolerancia, conservando las columnas intermedias de la malla de inversión.

**Reglas de emplazamiento de sondeos puntuales.** Son distintas porque los métodos tienen soporte espacial distinto:

- **MASW** — se ancla a la **coordenada central del tendido**, interpolada linealmente sobre las estacas reales. El método promedia bajo todo el arreglo, de modo que su punto representativo es el centroide, no una estaca concreta.
- **SEV / VES** — se ancla a la **coincidencia exacta** de `Xo` en la lista de estaciones. Un sondeo eléctrico vertical tiene una posición de centro definida en campo; aproximarla sería falsear el dato. Si esa abscisa no existe, la operación falla y devuelve el listado de abscisas disponibles, en lugar de aproximar en silencio.

---

## 7. Proyección de Vs sobre el modelo 2D

El objetivo es devolver Vs a toda la sección: aprender la relación Vp → Vs donde ambas se conocen (la vertical del MASW) y aplicarla donde solo se conoce Vp (toda la malla 2D).

### 7.1 Verificación previa

Antes de modelar se calcula la **correlación de Pearson entre Vp y Vs** en el conjunto de entrenamiento. Es un control de viabilidad: si la correlación es baja (< 0.5), Vp por sí sola no explica la variabilidad de Vs en ese sitio, y el modelo se apoyará en la posición espacial más que en la petrofísica — lo cual degrada gravemente la fiabilidad de cualquier extrapolación. La aplicación lo advierte explícitamente.

### 7.2 Atributos

| Atributo | Naturaleza | Función |
|---|---|---|
| `Vp` | Petrofísico | Portador principal de la señal |
| `Elevacion` | Geométrico vertical | Tendencia de compactación con la profundidad |
| `X`, `Y` | Geométrico horizontal | Tendencia lateral regional |
| `Vp_x_Elevacion` | Derivado | Interacción velocidad–profundidad |
| `dist_centroide` | Derivado | Distancia radial al centroide del levantamiento |

El centroide se calcula **una sola vez, sobre el entrenamiento**, y se reutiliza al aplicar el modelo a la malla. Calcularlo dos veces rompería la comparabilidad del atributo entre ambos conjuntos — un error sutil y frecuente en este tipo de pipeline.

### 7.3 Guardas de identificabilidad

> Esta sección documenta la corrección más importante respecto al flujo original, diagnosticada sobre datos reales.

Los atributos de posición solo pueden sostener una tendencia si la geometría del muestreo lo permite. Se aplican dos guardas antes de entrenar:

**Guarda 1 — Identificabilidad espacial.** Con *n* atributos espaciales hacen falta al menos *n + 2* sitios distintos de MASW. Con dos sitios, un plano queda determinado **exactamente**: el ajuste no deja residuo del que aprender, y cualquier diferencia de Vs entre líneas se atribuye íntegramente a la posición.

**Guarda 2 — Apalancamiento.** Se define como

```
apalancamiento = recorrido en la malla / ( 2 × desviación en el entrenamiento )
```

y se descartan los atributos que superen el umbral (5 por defecto). El mecanismo del fallo es el siguiente: si dos líneas vecinas corren casi paralelas, la coordenada `Y` de sus dos sondeos difiere en centímetros, mientras que a lo largo de cada perfil recorre decenas de metros. El estandarizador normaliza `Y` con esa desviación diminuta, de modo que el valor escalado alcanza ±58 a lo largo del perfil. Un coeficiente que en la tabla de importancias parece insignificante queda multiplicado por eso, y el modelo produce una **rampa lateral** en lugar de un perfil estratificado.

Medido sobre dos líneas reales, comparando la variación de Vs entre bandas de profundidad frente a la variación entre bandas de abscisa (la Vp medida da 9.5):

| Diferencia entre los dos MASW | vertical / lateral | Rango de Vs |
|---|---|---|
| 0 % | 9.2 | 201 – 582 |
| 5 % | **0.52** | −93 – 839 |
| 15 % | **0.18** | −731 – 1353 |

Con apenas un 5 % de diferencia entre sondeos, la variación lateral espuria ya supera a la estratificación real. Con las guardas activas el resultado deja de depender de esa diferencia. `Vp` y `Elevacion` nunca se descartan: son la física que el modelo debe aprender.

**Diagnóstico adicional de colinealidad.** En un sondeo 1D la velocidad crece de forma casi monótona con la profundidad — se mide `corr(Vp, Elevacion) = −0.995`. El modelo no puede separar «Vs sube porque Vp sube» de «Vs sube porque bajamos». La aplicación lo detecta y lo advierte, porque ese reparto arbitrario de coeficientes reaparece como variación lateral al aplicarse a la malla 2D, donde Vp **sí** varía a profundidad fija.

### 7.4 Validación cruzada espacial

Los datos geofísicos violan el supuesto de independencia de la validación cruzada clásica: dos nodos contiguos de un mismo sondeo están casi perfectamente correlacionados. Una partición aleatoria pondría nodos vecinos a ambos lados de la división, y el error medido sería optimista por fuga de información.

La aplicación usa **GroupKFold agrupando por línea** — *Leave-One-Line-Out*: cada línea se predice con un modelo que nunca la vio. Es el estándar en modelado espacial (Roberts et al., 2017). La alternativa `block` agrupa por conglomerados espaciales (K-Means sobre X, Y) cuando hay pocas líneas pero buena extensión.

Las predicciones así obtenidas son *out-of-fold*, y de ellas salen tanto las métricas honestas como los residuos que alimenta la corrección geoestadística.

### 7.5 Métricas

| Métrica | Lectura |
|---|---|
| RMSE | Error cuadrático medio, en m/s |
| MAE | Error absoluto medio, menos sensible a valores extremos |
| R² | Fracción de varianza explicada |
| **RMSE / σ(Vs)** | Error relativo a la variabilidad del objetivo |

La última es la más informativa: normaliza el error por la dispersión natural del dato. Su lectura convencional es **< 0.5 bueno · 0.5–0.8 aceptable · > 0.8 débil**. Un RMSE de 100 m/s es excelente en un sitio con Vs de 150 a 900, y pésimo en uno homogéneo de 400 a 450.

Se reporta además el error **por línea excluida**, que revela si el modelo falla de forma homogénea o si hay una línea anómala arrastrando la métrica global.

### 7.6 Optimización de hiperparámetros y elección de modelo

La búsqueda minimiza el RMSE de validación cruzada espacial usando **Optuna** con muestreador TPE (*Tree-structured Parzen Estimator*), una estrategia bayesiana que concentra las evaluaciones en las regiones prometedoras del espacio de búsqueda. Sin Optuna instalado, la aplicación cae a una rejilla manual equivalente.

| Modelo | Espacio de búsqueda | Extrapola |
|---|---|---|
| **Ridge** (recomendado) | α ∈ [10⁻³, 10²], logarítmico | **Sí** |
| Gradient Boosting | n_estimators, profundidad, tasa de aprendizaje, submuestreo | No |
| Random Forest | n_estimators, profundidad, hojas mínimas, atributos | No |

**Por qué Ridge es la recomendación.** Los modelos de árbol **no extrapolan**: fuera del rango de entrenamiento sus predicciones se aplanan en el valor de la hoja frontera. Como buena parte de la malla 2D cae fuera del envolvente petrofísico del MASW, esa saturación produce artefactos planos en las zonas de velocidad extrema. Un modelo lineal regularizado extrapola de forma continua, que es preferible a saturar. La regularización L2 de Ridge es además lo que estabiliza los coeficientes ante la colinealidad Vp–Elevación descrita antes.

### 7.7 Regression Kriging de los residuos

El modelo aprende la relación petrofísica global, pero no la estructura local del sitio. El **Regression Kriging** (Hengl et al., 2007) recupera esa componente: descompone la variable en tendencia más residuo espacialmente correlacionado,

```
Vs(u) = m(u) + ε(u)
```

donde `m(u)` es la predicción del modelo y `ε(u)` el residuo interpolado geoestadísticamente. La aplicación ofrece tres modos:

| Modo | Qué hace | Cuándo usarlo |
|---|---|---|
| **por línea (constante)** | Desplaza cada línea por el residuo medio de su MASW | **Por defecto.** Un sondeo por tendido |
| en planta (kriging) | Kriging ordinario en (X, Y) | Varios sondeos repartidos por el área |
| sin corrección | `Vs = Vs_predicho_ML` | Diagnóstico |

> **Por qué «por línea» es el modo por defecto.** El kriging ordinario es un **interpolador exacto**: reproduce el dato en su localización. Con un solo sondeo por línea hay muy pocas localizaciones distintas, el variograma ajustado tiene un rango diminuto, y la corrección resulta nula en casi toda la sección salvo un hundimiento abrupto de decenas de m/s en los pocos metros alrededor del sondeo. Como el MASW se emplaza en el centro del tendido, ese **cráter aparece justo en el centro del perfil**, cortando la continuidad horizontal de los estratos.
>
> Medido sobre una línea real: la corrección valía 0.00 en toda la sección y −40.36 m/s en cuatro columnas consecutivas alrededor de Xo = 34.5, con un salto de +39.5 m/s entre columnas contiguas.
>
> El modo «por línea» aplica un único desplazamiento constante por línea, igual al residuo medio de su sondeo. Corrige el sesgo sistemático —que es real y vale la pena corregir— sin inventar estructura lateral que el muestreo no puede sostener. Reproduce además el comportamiento del flujo original, donde las coordenadas eran constantes dentro de cada línea y el kriging solo podía devolver un valor por línea.

### 7.8 Recorte físico

La predicción final se acota al rango de Vs observado más un margen porcentual (15 % por defecto), con el límite inferior fijado en 0 porque **una velocidad negativa no tiene existencia física**. El margen es proporcional al rango observado, no un umbral arbitrario. La aplicación informa cuántos nodos fueron recortados: una fracción alta indica que el modelo está extrapolando más allá de lo razonable.

### 7.9 Diagnósticos de confiabilidad

**Extrapolación por variable.** Se reporta, para cada atributo, qué porcentaje de nodos cae fuera del rango de entrenamiento. Separa dos causas de naturaleza distinta:

- **Extrapolación espacial** (X, Y) — esperada y poco grave: es el propósito del ejercicio.
- **Extrapolación petrofísica** (Vp, Elevación) — más seria: ahí la relación Vp → Vs nunca se validó.

**Zona de confianza espacial.** En lugar de una bandera binaria, se calcula la distancia de cada nodo al MASW más cercano y se clasifica con una escala derivada de los propios datos: la **mediana de la distancia al vecino más próximo entre los sondeos**. Los umbrales son 2× y 5× esa escala de referencia (Alta / Media / Baja). El criterio es autoajustable: un levantamiento denso y uno disperso se juzgan cada uno con su propia escala.

**Importancia de variables.** Si `X` e `Y` dominan sobre `Vp`, el modelo está memorizando posiciones en vez de aprender la física Vp → Vs. Es el síntoma clásico de sobreajuste espacial, y la aplicación lo advierte cuando la importancia de Vp cae por debajo del 25 %.

---

## 8. Síntesis de líneas por Kriging Ordinario 3D

Para trazas donde no hay modelo de velocidades —una línea proyectada, o una orientación transversal a las levantadas— la aplicación genera el modelo interpolando desde las líneas vecinas. **Vp y Vs se sintetizan en un solo proceso**, sobre la misma malla y con los mismos parámetros, de modo que el perfil queda completo y consistente de una vez.

### 8.1 Malla objetivo

Se construye a partir de la topografía de la línea destino: para cada estaca se generan nodos desde su cota `Z` hasta `Z − prof_max`, con paso `dz`. La malla **sigue el terreno** por construcción, heredando el relieve real de la traza.

### 8.2 El variograma

El kriging requiere un modelo de continuidad espacial. El **semivariograma** mide cómo se pierde la semejanza con la distancia:

```
γ(h) = ½ · E[ ( Z(u) − Z(u+h) )² ]
```

Los modelos disponibles, con meseta (*sill*) `c`, alcance (*range*) `a` y efecto pepita (*nugget*) `c₀`:

| Modelo | γ(h) | Carácter |
|---|---|---|
| **Esférico** | `c·(1.5·h/a − 0.5·(h/a)³)` para h ≤ a; `c` después | Alcance finito, transición neta |
| Exponencial | `c·(1 − e^(−3h/a))` | Aproximación asintótica, contactos graduales |
| Gaussiano | `c·(1 − e^(−3(h/a)²))` | Muy suave en el origen, fenómenos continuos |
| Lineal | `c·mín(h/a, 1)` | Sin estructura definida |

Los parámetros por defecto se derivan de los datos: meseta igual a la varianza muestral, alcance horizontal de 500 m, y pepita al 10 % de la meseta. El **esférico** es el predeterminado porque su alcance finito representa bien unidades geológicas con contactos definidos, que es la situación habitual en geotecnia somera.

### 8.3 Anisotropía vertical

Los medios sedimentarios son **fuertemente anisótropos**: las propiedades varían mucho más rápido en la vertical (atravesando capas) que en la horizontal (a lo largo de ellas). Ignorarlo produce mezcla artificial entre estratos.

La anisotropía se introduce deformando la métrica de distancia:

```
h = √( Δx² + Δy² + ( Δz / factor )² )
```

de modo que **rango_vertical = rango_horizontal × factor**. Con el valor por defecto de 0.05, un alcance horizontal de 500 m corresponde a 25 m verticales — una relación 20:1 congruente con la geometría tabular de los depósitos sedimentarios.

### 8.4 El sistema de kriging ordinario

El kriging ordinario (Matheron, 1963) es el **mejor estimador lineal insesgado** (BLUE): minimiza la varianza del error sujeto a la condición de insesgamiento `Σwᵢ = 1`. Esa restricción se impone con un multiplicador de Lagrange, dando el sistema:

```
⎡ Γ  1 ⎤ ⎡ w ⎤   ⎡ γ₀ ⎤
⎢      ⎥ ⎢   ⎥ = ⎢    ⎥
⎣ 1ᵀ 0 ⎦ ⎣ ψ ⎦   ⎣ 1  ⎦
```

donde Γ es la matriz de semivarianzas entre puntos de dato, γ₀ el vector hacia el punto a estimar, w los pesos y ψ el multiplicador. La estimación es `Ẑ = Σ wᵢ·Zᵢ`.

A diferencia del IDW, los pesos **no dependen solo de la distancia**: incorporan la estructura de continuidad del variograma y el efecto de apantallamiento entre datos agrupados. El kriging entrega además la **varianza de estimación**, un mapa de incertidumbre que la aplicación conserva en las columnas `Varianza_Vp` y `Varianza_Vs`.

### 8.5 Motores de cálculo

| Motor | Cuándo se usa | Característica |
|---|---|---|
| pykrige | ≤ 600 puntos fuente | Sistema global, solución exacta |
| Local propio | > 600 puntos, o sin pykrige | Vecindad de los 48 vecinos más cercanos vía KD-tree |

El motor global resuelve una matriz *n × n* y escala mal: con varios miles de nodos fuente resulta impracticable. El motor local restringe cada estimación a su vecindad —práctica estándar en geoestadística aplicada, porque los puntos lejanos reciben peso despreciable— y da el mismo resultado práctico en segundos.

---

## 9. Visualización

### 9.1 Secciones 2D recortadas al relieve

El procedimiento interpola la variable sobre una malla regular distancia × elevación mediante triangulación de Delaunay (`scipy.griddata`), y luego aplica una **doble máscara topográfica**:

- por arriba, la superficie real extraída como `máx(Elevación)` por abscisa;
- por abajo, esa misma superficie desplazada la profundidad de corte.

El relleno usa 256 niveles continuos. La escala de color por defecto es una rampa personalizada **morado → azul → cian → verde → amarillo → naranja → rojo → rojo oscuro**, con mayor resolución perceptual en el rango bajo-medio, donde se concentra la información geotécnica útil.

Sobre el relleno se superponen **líneas de contorno parametrizables**, calculadas sobre la misma malla interpolada. Su uso habitual es marcar umbrales normativos: el contorno de Vs = 720–760 m/s separa las clases de sitio B y C de la clasificación NEHRP, y el de 360 m/s las clases C y D.

### 9.2 Cortinas 3D

Cada línea se representa como una superficie vertical que sigue su trazado real. La parametrización usa la **distancia acumulada a lo largo de la traza**:

```
d = Σ √( Δx² + Δy² )
```

lo que permite representar correctamente líneas quebradas, con cambios de azimut. La malla **sigue el terreno**: para cada posición a lo largo de la línea, la vertical arranca en la topografía de esa abscisa y baja la profundidad indicada. Por debajo del dato se abre hueco en lugar de extrapolar.

Los contornos 3D se calculan en el plano paramétrico (distancia, elevación) y se proyectan al trazado real, de modo que la isolínea sigue estrictamente la geometría de la línea. El 3D y el corte 2D comparten escala de color, contornos y profundidad: son la misma figura vista de dos maneras.

### 9.3 Contexto espacial

Se admite imagen base georreferenciada, con la extensión extraída automáticamente de un GeoTIFF o declarada por esquinas para JPG/PNG. La imagen se indexa a una paleta adaptativa de 256 colores para renderizarse como textura sobre una superficie 3D a cota constante.

---

## 10. Supuestos, límites y buenas prácticas

### 10.1 Supuestos del modelo físico

1. **Isotropía y homogeneidad local.** Las ecuaciones de la elasticidad lineal isótropa suponen que el medio no tiene dirección preferente. Los materiales estratificados son anisótropos transversalmente; los módulos calculados son valores efectivos, no propiedades intrínsecas.
2. **Pequeña deformación.** Los módulos son dinámicos. Su conversión a valores de diseño exige curvas de degradación.
3. **Coherencia de soporte.** Vp y Vs deben representar el mismo volumen. La etapa 1 está enteramente dedicada a garantizarlo.
4. **Densidad estimada.** Deriva de una correlación empírica. Un error del 10 % en ρ se traslada linealmente a G, E, K y λ.

### 10.2 Límites del modelado de Vs

1. **Una zona por bloque.** X e Y entran en coordenadas UTM absolutas. Entrenar con líneas separadas por kilómetros hace que Leave-One-Line-Out sea pura extrapolación; con Ridge el error puede crecer dos órdenes de magnitud. La aplicación lo advierte al detectar la mezcla.
2. **Tamaño muestral.** Con un MASW por línea, un bloque de cuatro líneas da del orden de 80 puntos. Las métricas tienen varianza alta y deben leerse como intervalo orientativo.
3. **Extrapolación petrofísica.** Fuera del rango de Vp muestreado, la relación Vp → Vs no está validada.

### 10.3 Secuencia de trabajo recomendada

1. Procesar cada línea en el Tab 1 y revisar `frac_peso_columna_central` y `N_trazas` antes de dar por bueno el perfil 1D.
2. Verificar que ν caiga en 0.15–0.49 en todo el perfil. Fuera de ahí, revisar el acoplamiento Vp–Vs.
3. Cargar topografía y agregar **las líneas de una sola zona** por bloque.
4. Generar el perfil 2D con Ridge y corrección de residuos «por línea». Leer RMSE/σ(Vs) y la tabla de extrapolación por variable.
5. Sintetizar las líneas faltantes, con la línea objetivo previamente incorporada al catálogo topográfico.
6. Visualizar y exportar con la misma escala y los mismos contornos en 2D y 3D.

---

## 11. Glosario de columnas de salida

| Columna | Descripción | Unidad |
|---|---|---|
| `Profundidad` | Profundidad bajo el terreno | m |
| `Elevacion` | Cota absoluta | m s.n.m. |
| `Xo` | Abscisa local del tendido | m |
| `X`, `Y` | Coordenadas UTM | m |
| `Vp_TRS_interpolado` | Vp acoplada a la profundidad del MASW | m/s |
| `Vs_MASW_original` | Vs del modelo de inversión | m/s |
| `Densidad` | Densidad estimada | t/m³ |
| `Coef_Poisson` | Coeficiente de Poisson ν | adimensional |
| `Modulo_Corte_G`, `Lame_Mu` | Módulo de corte | 10⁸ Pa |
| `Modulo_Young_E` | Módulo de Young | 10⁸ Pa |
| `Modulo_Bulk_K` | Módulo de compresibilidad | 10⁸ Pa |
| `Lame_Lambda` | 1ª constante de Lamé | 10⁸ Pa |
| `Check_Vp` | Control: × 316.23 reconstruye Vp | — |
| `Clase_Geofisica` | Unidad geofísica, 0 = más blando | entero |
| `Vs_predicho_ML` | Predicción del modelo, sin corrección | m/s |
| `Correccion_residual_kriging` | Corrección geoestadística del residuo | m/s |
| `Vs` | Predicción final, recortada | m/s |
| `Varianza_Vp`, `Varianza_Vs` | Varianza de estimación del kriging | (m/s)² |
| `flag_extrapolacion_general` | Fuera del rango de entrenamiento | booleano |
| `flag_extrapolacion_petrofisica` | Vp o Elevación fuera de rango | booleano |
| `zona_confianza_espacial` | Alta / Media / Baja | categórica |
| `dist_al_masw_mas_cercano` | Distancia en planta al sondeo | m |

---

## 12. Referencias

**Elasticidad y sísmica**

- Sheriff, R. E. & Geldart, L. P. (1995). *Exploration Seismology*, 2ª ed. Cambridge University Press.
- Aki, K. & Richards, P. G. (2002). *Quantitative Seismology*, 2ª ed. University Science Books.
- Gardner, G. H. F., Gardner, L. W. & Gregory, A. R. (1974). Formation velocity and density: the diagnostic basics for stratigraphic traps. *Geophysics*, 39(6), 770–780.

**MASW y refracción**

- Park, C. B., Miller, R. D. & Xia, J. (1999). Multichannel analysis of surface waves. *Geophysics*, 64(3), 800–808.
- Xia, J., Miller, R. D. & Park, C. B. (1999). Estimation of near-surface shear-wave velocity by inversion of Rayleigh waves. *Geophysics*, 64(3), 691–700.

**Interpolación y geoestadística**

- Shepard, D. (1968). A two-dimensional interpolation function for irregularly-spaced data. *Proc. 23rd ACM National Conference*, 517–524.
- Matheron, G. (1963). Principles of geostatistics. *Economic Geology*, 58(8), 1246–1266.
- Deutsch, C. V. & Journel, A. G. (1998). *GSLIB: Geostatistical Software Library*, 2ª ed. Oxford University Press.
- Hengl, T., Heuvelink, G. B. M. & Rossiter, D. G. (2007). About regression-kriging: from equations to case studies. *Computers & Geosciences*, 33(10), 1301–1315.

**Estadística y aprendizaje automático**

- Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *J. Computational and Applied Mathematics*, 20, 53–65.
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, 2ª ed. Springer.
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929.
- Akiba, T. et al. (2019). Optuna: a next-generation hyperparameter optimization framework. *KDD 2019*.

**Normativa**

- FEMA P-750 / NEHRP (2009). *Recommended Seismic Provisions for New Buildings and Other Structures*.
- CFE (2015). *Manual de Diseño de Obras Civiles — Diseño por Sismo*. Comisión Federal de Electricidad, México.

---

## Apéndice — Exportación a PDF

Este documento se convierte a PDF conservando formato, estilo e identidad corporativa mediante el script `exportar_pdf.py`, incluido en el paquete:

```
python exportar_pdf.py DOCUMENTACION.md
python exportar_pdf.py DOCUMENTACION.md --proteger
```

La conversión es en dos etapas: Markdown → HTML con hoja de estilo de impresión → PDF mediante el motor de maquetación de Chromium. El resultado respeta tipografía, tablas con encabezado sombreado, citas destacadas, saltos de página por capítulo y márgenes de documento técnico.

### Identidad corporativa

| Elemento | Archivo | Comportamiento |
|---|---|---|
| Logotipo de encabezado | `assets/logo_encabezado.png` | Repetido en el margen superior de todas las páginas |
| Marca de agua | `assets/marca_agua.png` | Centrada, rotada −30°, al 96 % del ancho y 5 % de opacidad |

Ambas se incrustan como *data URI*, de modo que el HTML intermedio es autocontenido y conserva la identidad aunque se mueva de carpeta o se imprima desde otro equipo. Los parámetros están al principio del script: `ANCHO_MARCA_PCT`, `OPACIDAD_MARCA` y `ROTACION_MARCA_DEG`. El rango utilizable de opacidad es 0.04–0.10; por debajo la marca desaparece en impresión, por encima estorba la lectura.

**Nota técnica sobre el encabezado.** En la impresión de Chromium, los elementos CSS `position: fixed` se repiten en todas las páginas pero solo se anclan dentro del área de contenido: cualquier desplazamiento negativo hacia el margen los reubica al pie, donde chocan con el texto. Por eso el logotipo repetido no se resuelve con CSS sino con la **plantilla de encabezado nativa** del motor de impresión, que sí dibuja dentro del margen. Requiere Playwright:

```
pip install playwright && playwright install chromium
```

Sin Playwright el script recurre a Chromium por línea de comandos, y entonces el logotipo aparece como membrete de la portada en lugar de repetirse.

### Sobre el alcance de la marca de agua

Conviene ser preciso con lo que una marca de agua puede y no puede hacer:

- **No impide capturas de pantalla.** Nada que se dibuje en la página puede impedirlo. Una marca de agua es un mecanismo de *atribución y trazabilidad*, no de control de acceso.
- **Sí dificulta la apropiación silenciosa.** Al cubrir el 96 % del ancho y cruzar el bloque de texto, no puede recortarse sin mutilar el contenido, de modo que cualquier fragmento que circule sigue llevando la identidad encima.
- **Hay una tensión real entre transparencia y protección.** Cuanta más transparencia, menos estorba la lectura pero más fácil resulta ignorarla o eliminarla por procesamiento de imagen. El ajuste por defecto privilegia la legibilidad.
- **La opción `--proteger`** cifra el PDF dejándolo de lectura libre pero sin permiso de extracción de texto ni de edición. Impide el copiar-pegar en los lectores que respetan los permisos; no resiste a herramientas que los ignoran deliberadamente, y tampoco impide capturas. Es una barrera de fricción y una declaración de intención.

Si el objetivo real es el control de distribución, la vía efectiva no es gráfica sino organizativa: entrega nominativa con marca de agua personalizada por destinatario, que permite identificar el origen de una filtración.

### Alternativas

| Herramienta | Comando | Nota |
|---|---|---|
| **Script incluido** | `python exportar_pdf.py DOCUMENTACION.md` | Identidad corporativa completa |
| Pandoc + LaTeX | `pandoc DOCUMENTACION.md -o doc.pdf --pdf-engine=xelatex` | Mejor calidad tipográfica; requiere distribución TeX |
| VS Code | Extensión *Markdown PDF* | Inmediato, sin identidad corporativa |
| Typora / Obsidian | Exportar → PDF | Conserva el tema visual de la aplicación |

---

*Documento generado para el flujo de trabajo de caracterización geofísica del subsuelo mediante sísmica de refracción y MASW.*
