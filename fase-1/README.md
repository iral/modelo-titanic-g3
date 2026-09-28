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

## 2. Fuente del conjunto de datos

* **Origen:** Competencia de Kaggle — [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).
* **Variable objetivo:** `Survived` (clasificación binaria: `0` = no sobrevivió, `1` = sobrevivió).
* **Tamaño:** 891 registros de entrenamiento y 418 de prueba.

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

## 5. Preprocesamiento

Los datos se dividen en un conjunto de entrenamiento y uno de prueba **antes** de cualquier
transformación, con `train_test_split(test_size=0.2, stratify=y, random_state=42)`. La estratificación
preserva la proporción de supervivientes (~38.4%) en ambos subconjuntos.

Las variables reciben tratamientos distintos:

* **Numéricas** (`Pclass`, `Age`, `SibSp`, `Parch`, `Fare`): imputación por **mediana**.
* **Categóricas** (`Sex`, `Embarked`): imputación por **categoría más frecuente** y codificación con
  `OneHotEncoder(handle_unknown="ignore")`.
* `Cabin` se descarta por su 77% de valores faltantes.

El preprocesamiento vive dentro del `Pipeline` (`ColumnTransformer`), de modo que en cada pliegue de
la validación cruzada las imputaciones y la codificación se ajustan únicamente con los datos de
entrenamiento de ese pliegue: **sin fuga de información**. El conjunto de prueba no participa en el
entrenamiento ni en la selección de hiperparámetros.

## 6. Modelo base

`DummyClassifier` con estrategia `most_frequent`: siempre predice la clase mayoritaria. Sirve como
punto de comparación para comprobar si el modelo predictivo aprende información útil.

## 7. Algoritmo definitivo

**`RandomForestClassifier` dentro de un `Pipeline` de Scikit-Learn**, con hiperparámetros
seleccionados mediante `GridSearchCV` y validación cruzada estratificada de 5 pliegues
(24 combinaciones evaluadas, exclusivamente sobre `X_train` / `y_train`).

### 7.1 Flujo del pipeline

```
X (datos crudos)
   └─ preprocessor (ColumnTransformer)
        ├─ num: Pclass, Age, SibSp, Parch, Fare → SimpleImputer(median)
        └─ cat: Sex, Embarked                    → SimpleImputer(most_frequent) + OneHotEncoder
   └─ clf: RandomForestClassifier
```

No se requiere escalado de variables numéricas: un bosque aleatorio no depende de la escala de las
características.

### 7.2 Hiperparámetros seleccionados

| Hiperparámetro | Valor |
| -------------- | -----: |
| `n_estimators` | 100 |
| `max_depth` | 5 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 2 |
| `random_state` | 42 |

### 7.3 Variables utilizadas

* **Numéricas:** `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`.
* **Categóricas:** `Sex`, `Embarked`.
* **Descartadas:** `PassengerId`, `Name`, `Ticket`, `Cabin` y la propia variable objetivo `Survived`.

## 8. Resultados finales

Las métricas se calculan sobre el **conjunto de prueba retenido** (`X_test`, 179 pasajeros que nunca
participaron en entrenamiento ni en la selección de hiperparámetros). La clase positiva es
`1 = sobrevivió`.

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------ | -------: | --------: | -----: | -------: | ------: |
| `DummyClassifier` (base, `most_frequent`) | 0.6145 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| **`RandomForestClassifier` (definitivo)** | **0.8156** | **0.8750** | **0.6087** | **0.7179** | **0.8356** |
| **Diferencia (Δ)** | **+0.2011** | +0.8750 | +0.6087 | +0.7179 | +0.3356 |

* **Accuracy media en validación cruzada (5 pliegues estratificados):** 0.8202.
* Las métricas se almacenan en `metricas_base.json` y `metricas_modelo.json`.

### 8.1 Matriz de confusión

| | Predice "no sobrevivió" | Predice "sobrevivió" |
| --- | ---: | ---: |
| **Real: no sobrevivió** | 104 (correcto) | 6 (falso positivo) |
| **Real: sobrevivió** | 27 (falso negativo) | 42 (correcto) |

El modelo identifica a **42 de los 69 supervivientes reales** (recall 0.61) cometiendo solo 6 falsos
positivos (precision 0.88), frente a un modelo base que clasifica bien a los 110 no sobrevivientes
pero **no detecta a ningún superviviente**.

### 8.2 Métrica principal: F1-Score

La variable objetivo está desbalanceada (38.4% sobrevivió frente a 61.6% no sobrevivió), por lo que la
accuracy premia demasiado acertar la clase mayoritaria: el modelo base ya alcanza 0.6145 sin aprender
nada. Por eso la métrica principal del proyecto es el **F1-Score de la clase superviviente**, que
combina *precision* y *recall* y obliga a balancear los dos tipos de error (no perder supervivientes
que sí sobrevivieron ni declarar supervivencia donde no la hubo). El **ROC-AUC** se reporta como
métrica complementaria, por ser independiente del umbral de decisión.

## 9. Modelo almacenado

El pipeline completo y entrenado se guarda con `joblib` en `fase-1/modelo.joblib`, e incluye tanto el
preprocesamiento (imputación y codificación ajustadas) como el `RandomForestClassifier`. La última
celda del notebook lo recupera desde disco y ejecuta una predicción sobre una muestra de 5 pasajeros
en su formato original, comprobando con `assert` que reproduce las predicciones del modelo en
memoria.

Guardar el pipeline completo permite reutilizar el modelo sobre nuevos datos sin reconstruir
manualmente las transformaciones realizadas durante el entrenamiento.

## 10. Limitaciones conocidas

* Solo 891 registros de entrenamiento: las métricas sobre 179 registros de prueba tienen varianza
  alta ante pequeños cambios de partición.
* El recall de supervivientes (~0.61) deja fuera a uno de cada tres supervivientes reales.
* `Name`, `Ticket` y `Cabin` se descartan; el título del pasajero, el tamaño familiar y la cubierta
  son información potencialmente útil que aún no se explota.
* El umbral de decisión se deja en 0.5, sin optimizar según el costo de los falsos negativos.

Las mejoras propuestas (ingeniería de características, ajuste de umbral, modelos adicionales,
validación anidada y calibración de probabilidades) se detallan en la sección **Reflexión técnica**
del notebook.
