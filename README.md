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
│   └── raw/              # Datos originales sin modificar (train.csv, test.csv)
├── fase-1/
│   └── notebook.ipynb    # notebook ejecutable con la exploración
|   |                       preparación, entrenamiento y evaluación del modelo.
|   │
|   ├── modelo_titanic.joblib # Pipeline completo del modelo seleccionado, incluyendo 
|   |                           el preprocesamiento y Random Forest.
|   │
|   ├── metricas_base.json # Métricas obtenidas por el modelo baseline.
|   │
|   └── metricas_modelo.json # Métricas obtenidas por el modelo baseline.
|
├── pyproject.toml        # Definición del entorno y dependencias
├── uv.lock               # Bloqueo de versiones de dependencias
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

## 9. Modelo predictivo

El modelo principal utilizado es un `RandomForestClassifier` integrado en un `Pipeline` junto con el preprocesamiento de los datos.

Los hiperparámetros se seleccionan mediante `GridSearchCV` utilizando validación cruzada estratificada de 5 pliegues.

La búsqueda se realiza exclusivamente sobre el conjunto de entrenamiento (`X_train`, `y_train`), mientras que el conjunto de prueba (`X_test`, `y_test`) se reserva para la evaluación final.

## 10. Modelo almacenado

El modelo final se almacena como un archivo `joblib`:

`fase-1/modelo_titanic.joblib`

El archivo contiene el `Pipeline` completo correspondiente al mejor estimador encontrado mediante `GridSearchCV`. Esto incluye tanto el preprocesamiento de los datos como el `RandomForestClassifier` entrenado.

Guardar el pipeline completo permite reutilizar posteriormente el modelo sobre nuevos datos sin tener que reconstruir manualmente las transformaciones realizadas durante el entrenamiento.


## 11. Puesta en Marcha

### 11.1 Requisitos
* Python 3.12 o superior.
* [uv](https://docs.astral.sh/uv/) para la gestión del entorno virtual y las dependencias.

### 11.2 Instalación
```bash
uv sync
```
Este comando crea el entorno virtual (`.venv/`), instala las dependencias del proyecto
(análisis de datos y visualización) y las herramientas para ejecutar el notebook.

> **Nota:** si trabajas en otro equipo y acabas de clonar el repositorio, este es el único
> paso necesario; no hace falta crear el entorno a mano.

### 11.3 Ejecutar el notebook
Desde la raíz del proyecto:

```bash
uv run jupyter lab
```

Se abrirá el navegador; dentro de la interfaz navega hasta `fase-1/notebook.ipynb` y ejecuta
las celdas en orden (menú **Run ▸ Run All Cells**).

Alternativamente, se puede ejecutar el notebook desde la terminal y regenerar sus salidas:
```bash
uv run jupyter nbconvert --to notebook --execute fase-1/notebook.ipynb --inplace
```

### 11.4 Detalle técnico
Para una guía exhaustiva (requisitos, esquema del dataset, contenido celda por celda del
notebook, resultados del análisis y resolución de problemas), ver **docs/TECNICO.md**.