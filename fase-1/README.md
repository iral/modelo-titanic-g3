# Fase 1 — Análisis, Modelado y Evaluación

Documentación de la **Fase 1** del proyecto *Modelo Predictivo de Supervivencia en el Titanic*.
Contiene el método completo, los resultados finales y las decisiones de diseño.

Para instalar y ejecutar el proyecto, ver la sección **Cómo ejecutar el proyecto** del
[README principal](../README.md).

---

## 1. Descripción del problema

El hundimiento del RMS Titanic el 15 de abril de 1912 es uno de los naufragios más conocidos de la
historia. El objetivo del proyecto es construir un modelo predictivo supervisado que determine la
probabilidad de supervivencia de un pasajero a partir de sus características personales y de viaje
(edad, género, clase socioeconómica y tarifa pagada).

## 2. Fuente del conjunto de datos y conjuntos utilizados

* **Origen:** Competencia de Kaggle — [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).
* **Variable objetivo:** `Survived` (clasificación binaria: `0` = no sobrevivió, `1` = sobrevivió).

Para evitar ambigüedades, en toda la documentación y en el notebook se usan estos nombres. Las
palabras «test» y «prueba» solo designan al archivo `test.csv`.

| Nombre | Filas | ¿Tiene `Survived`? | Propósito |
|--------|------:|:------------------:|-----------|
| `train.csv` | 891 | Sí | Único conjunto etiquetado: **todo** el material disponible para aprender y evaluar de forma supervisada. |
| **Desarrollo** (`X_dev`, `y_dev`) | 712 | Sí | Entrenamiento y selección del modelo mediante validación cruzada. |
| **Validación** (pliegues de la CV) | ~142 por pliegue | Sí | Pliegues de validación cruzada **dentro** de `X_dev`. |
| **Evaluación final** (`X_final`, `y_final`) | 179 | Sí | Reservado; se usa **una sola vez** con el modelo ya fijado. |
| `test.csv` | 418 | **No** | Conjunto externo **sin etiquetas**: solo sirve para generar predicciones. No permite calcular métricas supervisadas y **no amplía** los 891 casos etiquetados. No participa en particiones, validación cruzada ni selección. |

## 3. Objetivo del modelo

Desarrollar un pipeline de Machine Learning reproducible y documentado que preprocese los datos
adecuadamente para evitar la fuga de información, entrene un modelo supervisado de clasificación y
supere el desempeño de un modelo base.

## 4. Datos utilizados

### 4.1 Variable objetivo

* `Survived`: indica si el pasajero sobrevivió (`1`) o no (`0`).

### 4.2 Variables predictoras

| Variable   | Tipo       |
| ---------- | ---------- |
| `Pclass`   | Numérica   |
| `Age`      | Numérica   |
| `SibSp`    | Numérica   |
| `Parch`    | Numérica   |
| `Fare`     | Numérica   |
| `Sex`      | Categórica |
| `Embarked` | Categórica |

Durante la exploración se identificaron valores faltantes en `Age` (~20%), `Cabin` (~77%) y
`Embarked` (2 registros). Para el modelo final se utilizan las variables de la tabla anterior y los
valores faltantes se tratan dentro del pipeline de preprocesamiento.

### 4.3 Tickets compartidos y grupos de viaje

El EDA analiza los pasajeros que comparten `Ticket` (familias y acompañantes):

* 681 tickets distintos; **134** son compartidos y afectan a **344 de 891 pasajeros**.
* En **63,4 %** de esos grupos todos los miembros tuvieron el mismo desenlace, frente a ~42,9 %
  esperable si las etiquetas fueran independientes del grupo (rango 95 %: 35,1 %–50,4 %).
* `Ticket` y familia (`SibSp + Parch + 1`) **no son equivalentes**: 75 pasajeros comparten ticket
  sin familiares declarados y 85 declaran familiares con ticket propio.
* **Qué no se concluye:** compartir ticket no es fuga de información en sí (el modelo no recibe
  `Ticket`); es un riesgo de evaluación optimista si compañeros de viaje quedan repartidos entre
  entrenamiento y evaluación.

## 5. Partición y preprocesamiento

### 5.1 Partición por grupos

`train.csv` se divide **antes** de cualquier transformación con `StratifiedGroupKFold(n_splits=5,
shuffle=True, random_state=42)` agrupando por `Ticket`: un pliegue (~20 %) forma la **evaluación
final** (179 pasajeros) y el resto el **desarrollo** (712). Cada `Ticket` queda completo en uno de
los dos lados (0 tickets compartidos, comprobado con `assert`) y se conserva la proporción de
supervivientes (38,3 % en desarrollo, 38,5 % en evaluación final).

Como referencia, la partición aleatoria 80/20 usada anteriormente compartía **61 tickets** entre
entrenamiento y evaluación (67 de los 179 pasajeros de evaluación tenían algún compañero de ticket
en entrenamiento).

**Escenario de generalización.** La partición por grupos estima el rendimiento sobre **grupos de
viaje no vistos**, la estimación conservadora; la partición aleatoria estimaría el rendimiento con
compañeros presentes en el entrenamiento. Se elige la primera. Matiz: en `test.csv` hay pasajeros
cuyo ticket aparece en `train.csv` (152 de 418), de modo que el escenario real de ese archivo queda
entre ambos. Una agrupación por `Ticket` no captura todos los vínculos familiares: en la evaluación
final, 14 de 80 pasajeros con familiares declarados comparten apellido con alguien de desarrollo.

### 5.2 Preprocesamiento

Las variables reciben tratamientos distintos:

* **Numéricas** (`Pclass`, `Age`, `SibSp`, `Parch`, `Fare`): imputación por **mediana**.
* **Categóricas** (`Sex`, `Embarked`): imputación por **categoría más frecuente** y codificación con
  `OneHotEncoder(handle_unknown="ignore")`.
* `Cabin`, `PassengerId`, `Name` y `Ticket` no entran al modelo (`Ticket` se usa solo para agrupar).

El preprocesamiento vive dentro del `Pipeline` (`ColumnTransformer`), de modo que en cada pliegue de
la validación cruzada las imputaciones y la codificación se ajustan únicamente con los datos de
entrenamiento de ese pliegue: **sin fuga de información**. La evaluación final no participa en el
entrenamiento ni en la selección.

## 6. Modelo base

`DummyClassifier` con estrategia `most_frequent`: siempre predice la clase mayoritaria. Sirve como
punto de comparación para comprobar si el modelo predictivo aprende información útil.

## 7. Algoritmo definitivo y selección

**`RandomForestClassifier` dentro de un `Pipeline` de Scikit-Learn.**

### 7.1 Métrica principal y procedimiento de selección

La métrica principal es el **F1-Score de la clase superviviente** (la clase está desbalanceada:
38,4 % / 61,6 %). La selección es coherente con ella: `GridSearchCV(..., scoring={f1, accuracy,
precision, recall, roc_auc}, refit="f1")`, con validación cruzada `StratifiedGroupKFold` de 5
pliegues y `groups=Ticket` (ningún grupo queda a la vez en entrenamiento y validación, comprobado
por pliegue), **exclusivamente sobre `X_dev` / `y_dev`**. Las demás métricas solo se registran.
Se registra también la F1 de entrenamiento para vigilar el sobreajuste.

### 7.2 Flujo del pipeline

```
X (datos crudos)
   └─ preprocessor (ColumnTransformer)
        ├─ num: Pclass, Age, SibSp, Parch, Fare → SimpleImputer(median)
        └─ cat: Sex, Embarked                    → SimpleImputer(most_frequent) + OneHotEncoder
   └─ clf: RandomForestClassifier
   └─ (envoltura) FixedThresholdClassifier: aplica el umbral de decisión guardado
```

No se requiere escalado: un bosque aleatorio no depende de la escala de las características.

### 7.3 Mejora del recall (solo con validación)

Regla fijada de antemano: una alternativa se adopta solo si mejora la F1 media de validación en
≥ 0,01. Resultados en validación cruzada por grupos sobre `X_dev`:

| Configuración | F1 (media ± desv.) | Precision | Recall | Accuracy | ROC-AUC | F1 entrenamiento |
|---------------|-------------------:|----------:|-------:|---------:|--------:|-----------------:|
| RF base (24 combinaciones) | 0,7268 ± 0,0351 | 0,7973 | 0,6700 | 0,8076 | 0,8348 | 0,8650 |
| **RF + `class_weight` (48 combinaciones)** | **0,7491 ± 0,0480** | 0,7781 | 0,7248 | 0,8145 | 0,8375 | 0,8759 |

* **`class_weight="balanced"` se adopta** (+0,0223 de F1): sube el recall a costa de algo de
  precision. La ganancia es menor que la desviación entre pliegues: evidencia moderada.
* **Ajuste de umbral: no se adopta.** El umbral que maximiza F1 fuera de muestra es 0,5, y en la
  validación anidada ajustar el umbral empeora la F1 media (0,7191 frente a 0,7491).
* Con la configuración base, fuera de muestra y umbral 0,5, el modelo detectaba 183 de 273
  supervivientes (67,0 %) y omitía 90.

### 7.4 Hiperparámetros seleccionados

| Hiperparámetro | Valor |
| -------------- | -----: |
| `class_weight` | `balanced` |
| `n_estimators` | 200 |
| `max_depth` | `None` |
| `min_samples_split` | 5 |
| `min_samples_leaf` | 2 |
| `random_state` | 42 |
| Umbral de decisión | 0,5 |

### 7.5 Variables utilizadas

* **Numéricas:** `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`.
* **Categóricas:** `Sex`, `Embarked`.
* **Descartadas:** `PassengerId`, `Name`, `Ticket` (solo agrupación), `Cabin` y `Survived`.

## 8. Resultados finales

Métricas calculadas **una sola vez** sobre la **evaluación final** (`X_final`, 179 pasajeros,
69 supervivientes), con modelo, hiperparámetros y umbral ya fijados por validación. Clase positiva:
`1 = sobrevivió`.

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------ | -------: | --------: | -----: | -------: | ------: |
| `DummyClassifier` (base, `most_frequent`) | 0.6145 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| `RandomForest` procedimiento anterior (accuracy, sin grupos) — *referencia* | 0.8324 | 0.7910 | 0.7681 | 0.7794 | 0.9067 |
| **`RandomForest` definitivo (F1, por grupos)** | **0.8603** | **0.8143** | **0.8261** | **0.8201** | **0.9157** |

La fila de referencia aplica el procedimiento anterior a esta **misma** evaluación final y no
interviene en ninguna decisión. Las cifras publicadas en la versión anterior (accuracy 0.8156,
F1 0.7179, recall 0.6087) se calcularon sobre **otros** 179 pasajeros (partición aleatoria) y **no
son comparables** con las de esta tabla.

### 8.1 Matriz de confusión (evaluación final)

| | Predice "no sobrevivió" | Predice "sobrevivió" |
| --- | ---: | ---: |
| **Real: no sobrevivió** | 97 (correcto) | 13 (falso positivo) |
| **Real: sobrevivió** | 12 (falso negativo) | 57 (correcto) |

El modelo identifica a **57 de los 69 supervivientes reales** (recall 0.83) con 13 falsos positivos
(precision 0.81).

### 8.2 Cómo leer estas cifras

* Con 69 supervivientes, un pasajero equivale a 1,4 puntos de recall: los intervalos son amplios.
* La F1 de la evaluación final (0,82) supera a la estimada por validación cruzada (0,75 ± 0,05):
  esta evaluación final parece haber sido favorable. Como estimación del rendimiento esperado se
  recomienda la de validación cruzada (~0,75); la evaluación final es una confirmación de alta varianza.
* Las métricas de validación y de entrenamiento (sección 7.3) no deben confundirse con las finales.
* Se almacenan en `metricas_base.json` y `metricas_modelo.json` (esta incluye validación, evaluación
  final, partición y versiones de las librerías).

## 9. Modelo almacenado y verificación

El modelo completo se guarda con `joblib` en `fase-1/modelo.joblib`: un `FixedThresholdClassifier`
que contiene el pipeline entrenado (imputación, codificación y `RandomForestClassifier`) y el umbral
de decisión. La última celda del notebook verifica, con `assert`:

1. El esquema de entrada (columnas y tipos), incluido el rechazo de una entrada sin `Sex`.
2. Predicciones sobre 5 pasajeros en su formato original.
3. Que el modelo cargado reproduce las predicciones y probabilidades del modelo en memoria sobre
   los 179 pasajeros de evaluación final, y su F1 reportado.
4. Que un **proceso Python nuevo** obtiene las mismas predicciones.
5. Que `test.csv` (418 pasajeros sin etiquetas) puede predecirse (solo se cuentan las predicciones).

**Compatibilidad:** `joblib` no garantiza compatibilidad entre versiones de scikit-learn. El
archivo debe cargarse con la versión con la que se generó (registrada en `metricas_modelo.json`);
`uv sync` instala la fijada en `uv.lock`. Si cambia el entorno, regenere el modelo ejecutando el notebook.

## 10. Limitaciones conocidas

* Solo 891 registros etiquetados: las métricas sobre 179 registros tienen varianza alta.
* La mejora de `class_weight` (+0,022 de F1 en validación) es menor que la desviación entre pliegues.
* El bosque presenta sobreajuste moderado (F1 de entrenamiento ~0,88 frente a ~0,75 en validación).
* La agrupación por `Ticket` no captura todos los vínculos familiares, y el escenario evaluado
  (grupos no vistos) es conservador respecto a `test.csv`.
* `Name`, `Ticket` (como variable) y `Cabin` se descartan; el título, el tamaño familiar y la
  cubierta son información potencialmente útil aún no explotada.
* La validación del umbral es aproximadamente anidada (los hiperparámetros se eligieron con todo `X_dev`).

Las mejoras propuestas se detallan en la sección **Reflexión técnica** del notebook.
