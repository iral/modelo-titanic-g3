# Modelo Predictivo de Supervivencia en el Titanic

## 1. Integrantes del Equipo
* **Santiago Acevedo González**
* **Ivan Rene Acosta Lallemand**
* **Daniela Andrea Gallego Díaz**

## 2. Descripción del Problema
El hundimiento del RMS Titanic es uno de los naufragios más conocidos de la historia. El objetivo principal de este proyecto es construir un modelo predictivo supervisado capaz de determinar la probabilidad de supervivencia de los pasajeros a partir de sus características personales y de viaje (como edad, género, clase socioeconómica y tarifa pagada).

## 3. Fuente del Conjunto de Datos
* **Origen:** Competencia de Kaggle - [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).
* **Variable Objetivo:** `Survived` (Clasificación binaria: `0` = No sobrevivió, `1` = Sobrevivió).

## 4. Objetivo del Modelo
Desarrollar un pipeline de Machine Learning reproducible y documentado que preprocese los datos adecuadamente para evitar la fuga de información, entrene un modelo supervisado de clasificación y supere el desempeño de un modelo base.

## 5. Estructura del Proyecto
```
modelo-titanic-g3/
├── data/
│   └── raw/                    # Datos originales sin modificar (train.csv, test.csv)
├── docs/
│   └── TECNICO.md              # Guía técnica: entorno, ejecución y resolución de problemas
├── fase-1/
│   ├── notebook.ipynb          # Notebook ejecutado: EDA, preprocesamiento,
│   │                           # modelo base, modelo predictivo, evaluación y verificación
│   ├── modelo.joblib           # Pipeline completo entrenado (preprocesamiento + Random Forest)
│   ├── metricas_base.json      # Métricas del modelo base sobre el conjunto de prueba
│   └── metricas_modelo.json    # Métricas del modelo predictivo sobre el conjunto de prueba
├── pyproject.toml              # Definición del entorno y dependencias
├── uv.lock                     # Bloqueo de versiones de dependencias
└── README.md
```

## 6. Datos utilizados

El proyecto utiliza el conjunto de datos de pasajeros del Titanic proporcionado para el proyecto.

### 6.1 Variable objetivo

* `Survived`: indica si el pasajero sobrevivió (`1`) o no sobrevivió (`0`).

### 6.2 Variables predictoras

| Variable   | Tipo       |
| ---------- | ---------- |
| `Pclass`   | Numérica   |
| `Age`      | Numérica   |
| `SibSp`    | Numérica   |
| `Parch`    | Numérica   |
| `Fare`     | Numérica   |
| `Sex`      | Categórica |
| `Embarked` | Categórica |

Durante la exploración se identificaron valores faltantes en `Age`, `Cabin` y `Embarked`. Para el modelo final se utilizan las variables seleccionadas anteriormente y los valores faltantes se tratan mediante el pipeline de preprocesamiento.

## 7. Preprocesamiento

Los datos se dividen en un conjunto de entrenamiento y un conjunto de prueba antes de realizar el ajuste del preprocesamiento.

Las variables numéricas y categóricas reciben tratamientos diferentes:

* Las variables numéricas utilizan imputación mediante la mediana.
* Las variables categóricas utilizan imputación mediante la categoría más frecuente y codificación mediante `OneHotEncoder`.

El preprocesamiento forma parte del `Pipeline` utilizado por el modelo. De esta manera, durante la validación cruzada, los parámetros de transformación se ajustan únicamente con los datos de entrenamiento de cada partición, evitando fuga de información (*data leakage*).

El conjunto de prueba se mantiene separado y no participa en el entrenamiento ni en la selección de hiperparámetros.

## 8. Modelo base

Como referencia se utiliza un `DummyClassifier` con la estrategia `most_frequent`.

Este modelo siempre predice la clase mayoritaria y permite establecer un punto de comparación para determinar si el modelo predictivo aprende información útil a partir de las variables disponibles.

Las métricas del modelo base se almacenan en:

`fase-1/metricas_base.json`

## 9. Algoritmo definitivo

El algoritmo definitivo del proyecto es un **`RandomForestClassifier` dentro de un `Pipeline` de
Scikit-Learn**, cuyos hiperparámetros se seleccionan mediante `GridSearchCV` con validación cruzada
estratificada de 5 pliegues (24 combinaciones evaluadas).

La búsqueda se realiza exclusivamente sobre el conjunto de entrenamiento (`X_train`, `y_train`),
mientras que el conjunto de prueba (`X_test`, `y_test`) se reserva para la evaluación final.

### 9.1 Flujo del pipeline

```
X (datos crudos)
   └─ preprocessor (ColumnTransformer)
        ├─ num: Pclass, Age, SibSp, Parch, Fare → SimpleImputer(median)
        └─ cat: Sex, Embarked                    → SimpleImputer(most_frequent) + OneHotEncoder
   └─ clf: RandomForestClassifier
```

No se requiere escalado de variables numéricas: un bosque aleatorio no depende de la escala de las
características.

### 9.2 Hiperparámetros seleccionados

| Hiperparámetro | Valor |
| -------------- | -----: |
| `n_estimators` | 100 |
| `max_depth` | 5 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 2 |
| `random_state` | 42 |

### 9.3 Variables utilizadas

* **Numéricas:** `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`.
* **Categóricas:** `Sex`, `Embarked`.
* **Descartadas:** `PassengerId`, `Name`, `Ticket`, `Cabin` (77% de valores faltantes) y la propia
  variable objetivo `Survived`.

## 10. Resultados finales

Las métricas se calculan sobre el **conjunto de prueba retenido** (`X_test`, 179 pasajeros que nunca
participaron en entrenamiento ni en la selección de hiperparámetros). La clase positiva es
`1 = sobrevivió`.

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------ | -------: | --------: | -----: | -------: | ------: |
| `DummyClassifier` (base, `most_frequent`) | 0.6145 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| **`RandomForestClassifier` (definitivo)** | **0.8156** | **0.8750** | **0.6087** | **0.7179** | **0.8356** |
| **Diferencia (Δ)** | **+0.2011** | +0.8750 | +0.6087 | +0.7179 | +0.3356 |

* **Accuracy media en validación cruzada (5 pliegues estratificados):** 0.8202.
* Las métricas se almacenan en `fase-1/metricas_base.json` y `fase-1/metricas_modelo.json`.

### 10.1 Matriz de confusión

| | Predice "no sobrevivió" | Predice "sobrevivió" |
| --- | ---: | ---: |
| **Real: no sobrevivió** | 104 (correcto) | 6 (falso positivo) |
| **Real: sobrevivió** | 27 (falso negativo) | 42 (correcto) |

El modelo identifica a **42 de los 69 supervivientes reales** (recall 0.61) cometiendo solo 6 falsos
positivos (precision 0.88), frente a un modelo base que clasifica bien a los 110 no sobrevivientes
pero **no detecta a ningún superviviente**.

### 10.2 Métrica principal: F1-Score

La variable objetivo está desbalanceada (38.4% sobrevivió frente a 61.6% no sobrevivió), por lo que la
accuracy premia demasiado acertar la clase mayoritaria: el modelo base ya alcanza 0.6145 sin aprender
nada. Por eso la métrica principal del proyecto es el **F1-Score de la clase superviviente**, que
combina *precision* y *recall* y obliga a balancear los dos tipos de error (no perder supervivientes
que sí sobrevivieron ni declarar supervivencia donde no la hubo). El **ROC-AUC** se reporta como
métrica complementaria, por ser independiente del umbral de decisión.

## 11. Modelo almacenado

El modelo final se almacena como un archivo `joblib`:

`fase-1/modelo.joblib`

El archivo contiene el `Pipeline` completo correspondiente al mejor estimador encontrado mediante `GridSearchCV`. Esto incluye tanto el preprocesamiento de los datos como el `RandomForestClassifier` entrenado.

Guardar el pipeline completo permite reutilizar posteriormente el modelo sobre nuevos datos sin tener que reconstruir manualmente las transformaciones realizadas durante el entrenamiento.


## 12. Puesta en marcha

### 12.1 Requisitos
* Python 3.12 o superior.
* [uv](https://docs.astral.sh/uv/) para la gestión del entorno virtual y las dependencias.

### 12.2 Instalación
```bash
uv sync
```
Este comando crea el entorno virtual (`.venv/`), instala las dependencias del proyecto
(análisis de datos y visualización) y las herramientas para ejecutar el notebook.

> **Nota:** si trabajas en otro equipo y acabas de clonar el repositorio, este es el único
> paso necesario; no hace falta crear el entorno a mano.

### 12.3 Ejecutar el notebook
Desde la raíz del proyecto:

```bash
uv run jupyter lab
```

Se abrirá el navegador; dentro de la interfaz navega hasta `fase-1/notebook.ipynb` y ejecuta
todo desde un kernel limpio con el menú **Kernel ▸ Restart Kernel and Run All Cells**. Así se
regeneran en orden todas las tablas, gráficas y salidas.

Alternativamente, se puede ejecutar el notebook desde la terminal y regenerar sus salidas:
```bash
uv run jupyter nbconvert --to notebook --execute fase-1/notebook.ipynb --inplace
```

La ejecución completa regenera además los artefactos de la fase: `fase-1/modelo.joblib`,
`fase-1/metricas_base.json` y `fase-1/metricas_modelo.json`.

**Verificación:** la última celda del notebook debe imprimir
`Verificación exitosa: el pipeline guardado se cargó y predijo correctamente.`

**Reproducibilidad:** todas las decisiones aleatorias del notebook (división de datos, validación
cruzada, bosque aleatorio y muestra de verificación) usan la constante `SEED = 42`, fijada en la
celda de configuración y propagada como `random_state` a cada estimador. Con el dataset original sin
modificar, dos ejecuciones completas producen las mismas métricas de la tabla de la sección 10.

### 12.4 Detalle técnico
Para una guía exhaustiva (requisitos, esquema del dataset, contenido celda por celda del
notebook, resultados del análisis y resolución de problemas), ver **docs/TECNICO.md**.