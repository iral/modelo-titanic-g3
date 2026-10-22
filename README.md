# Modelo Predictivo de Supervivencia en el Titanic

## 1. Integrantes del Equipo
* **Santiago Acevedo González**
* **Ivan Rene Acosta Lallemand**
* **Daniela Andrea Gallego Díaz**

## 2. Descripción

Proyecto de Machine Learning supervisado que predice la supervivencia de pasajeros del Titanic
(`Survived`) a partir de sus características personales y de viaje. La documentación completa de la
**Fase 1** (datos, preprocesamiento, algoritmo, resultados y modelo almacenado) está en
[`fase-1/README.md`](fase-1/README.md).

## 3. Estructura del Proyecto
```
modelo-titanic-g3/
├── data/
│   └── raw/                    # Datos originales sin modificar (train.csv, test.csv)
├── docs/
│   └── TECNICO.md              # Guía técnica: entorno, ejecución y resolución de problemas
├── fase-1/
│   ├── README.md               # Documentación de la Fase 1: método, resultados y modelo
│   ├── notebook.ipynb          # Notebook ejecutado: EDA, preprocesamiento, modelo base,
│   │                           # modelo predictivo, evaluación y verificación
│   ├── modelo.joblib           # Modelo completo (preprocesamiento + Random Forest + umbral)
│   ├── metricas_base.json      # Métricas del modelo base en la evaluación final
│   └── metricas_modelo.json    # Métricas del modelo predictivo en la evaluación final
├── pyproject.toml              # Definición del entorno y dependencias
├── uv.lock                     # Bloqueo de versiones de dependencias
└── README.md
```

## 4. Cómo ejecutar el proyecto (Fase 1)

### 4.1 Requisitos
* Python 3.12 o superior.
* [uv](https://docs.astral.sh/uv/) para la gestión del entorno virtual y las dependencias.

### 4.2 Instalación
Desde la raíz del proyecto:
```bash
uv sync
```
Crea el entorno virtual (`.venv/`), instala las dependencias (análisis de datos y visualización) y
las herramientas para ejecutar el notebook.

> **Nota:** si acabas de clonar el repositorio, este es el único paso necesario; no hace falta crear
> el entorno a mano.

### 4.3 Ejecutar el notebook
```bash
uv run jupyter lab
```
Se abrirá el navegador; navega hasta `fase-1/notebook.ipynb` y ejecuta todo desde un kernel limpio
con el menú **Kernel ▸ Restart Kernel and Run All Cells**. Así se regeneran en orden todas las
tablas, gráficas y salidas.

Alternativamente, desde la terminal:
```bash
uv run jupyter nbconvert --to notebook --execute fase-1/notebook.ipynb --inplace
```

La ejecución completa regenera además los artefactos de la fase: `fase-1/modelo.joblib`,
`fase-1/metricas_base.json` y `fase-1/metricas_modelo.json`.

**Verificación:** la última celda del notebook debe imprimir
`Verificación exitosa: el pipeline guardado se cargó y predijo correctamente.` Esa celda comprueba el
esquema de entrada, la equivalencia de predicciones entre el modelo en memoria y el cargado desde
disco, y una predicción en un proceso Python independiente.

**Reproducibilidad:** todas las decisiones aleatorias del notebook (división de datos, validación
cruzada, bosque aleatorio y muestra de verificación) usan la constante `SEED = 42`, fijada en la
celda de configuración y propagada como `random_state` a cada estimador y a cada partición. Con el
dataset original sin modificar y el entorno de `uv.lock`, dos ejecuciones completas producen las mismas
métricas publicadas en [`fase-1/README.md`](fase-1/README.md).

**Conjuntos de datos:** `train.csv` (891, etiquetado) se divide por grupos de `Ticket` en *desarrollo*
(712; entrenamiento y validación cruzada) y *evaluación final* (179; una sola vez). `test.csv` (418)
**no tiene etiquetas** y solo se usa para generar predicciones; no se calcula ninguna métrica con él.

**Modelo guardado:** `modelo.joblib` solo es válido con la versión de scikit-learn con la que se generó
(registrada en `fase-1/metricas_modelo.json`). Tras `uv sync`, vuelva a ejecutar el notebook para
regenerar los artefactos en su entorno.

## 5. Documentación

| Documento | Contenido |
|-----------|-----------|
| [`fase-1/README.md`](fase-1/README.md) | Problema, conjuntos de datos, tickets compartidos, partición por grupos, selección por F1, mejora del recall, resultados finales y modelo almacenado. |
| [`docs/TECNICO.md`](docs/TECNICO.md) | Guía exhaustiva: requisitos, esquema del dataset, ejecución, reproducibilidad y resolución de problemas. |
| `fase-1/notebook.ipynb` | Notebook completo y ejecutado, con la reflexión técnica final. |

## 6. Cambios y mejoras de la segunda entrega

Resumen de las correcciones realizadas tras la evaluación de la primera entrega. El detalle completo
(métodos, tablas y resultados) está en [`fase-1/README.md`](fase-1/README.md) y en el notebook.

| Observación de la evaluación | Cambio realizado | Dónde |
|------------------------------|------------------|-------|
| `test.csv` no tiene etiquetas y no amplía los 891 casos | Se documenta el propósito de cada conjunto (glosario): `train.csv` se divide en *desarrollo* (712) y *evaluación final* (179); `test.csv` solo genera predicciones y no se calcula ninguna métrica con él. Se añaden `assert` de estructura y se eliminan los usos ambiguos de «test» y «prueba». | Notebook, `fase-1/README.md`, `docs/TECNICO.md` |
| Selección del modelo incoherente con la métrica principal | `GridSearchCV` pasa de `scoring="accuracy"` a `refit="f1"` (F1 de la clase superviviente), registrando también accuracy, precision, recall, ROC-AUC y F1 de entrenamiento. | Notebook |
| 61 tickets compartidos entre entrenamiento y evaluación | Se verifica el hallazgo (correspondía a la partición aleatoria interna: 67 de 179 pasajeros afectados) y se añade un EDA de grupos de viaje. La partición y la validación cruzada pasan a ser **por grupos de `Ticket`** (`StratifiedGroupKFold`), con 0 tickets compartidos comprobado por `assert`. Se justifica el escenario de generalización y se documentan los vínculos familiares que `Ticket` no captura. | Notebook, `fase-1/README.md` |
| Recall bajo (se omiten supervivientes) | Se estudian mejoras solo con validación: `class_weight="balanced"` (adoptado, con regla de decisión fijada de antemano) y ajuste de umbral (evaluado con validación anidada y **no** adoptado). La evaluación final se usa una sola vez. | Notebook, `fase-1/README.md` |
| Persistencia del modelo | El modelo guardado incluye el pipeline y el umbral (`FixedThresholdClassifier`). La verificación ahora valida columnas y tipos de entrada, la equivalencia de predicciones y probabilidades sobre toda la evaluación final, y una predicción en un proceso Python independiente. | Notebook |
| Documentación desactualizada | Se actualizan los tres documentos (este README, `fase-1/README.md` y `docs/TECNICO.md`): alcance, dependencias (scikit-learn, joblib), conjuntos de datos, versión de scikit-learn y solución de problemas al cargar el modelo. | Documentación |

### Resultados tras los cambios

Los resultados anteriores (accuracy 0.8156, F1 0.7179, recall 0.6087) se calcularon sobre otra
partición aleatoria y **no son comparables** con los actuales.

| Métrica | Validación cruzada por grupos (`X_dev`) | Evaluación final (`X_final`, 179 pasajeros) |
|---------|----------------------------------------:|--------------------------------------------:|
| F1-Score | 0.7491 ± 0.0480 | 0.8201 |
| Recall | 0.7248 | 0.8261 |
| Precision | 0.7781 | 0.8143 |
| Accuracy | 0.8145 | 0.8603 |

Con solo 69 supervivientes en la evaluación final, estas cifras tienen alta varianza; la estimación
de validación cruzada (~0.75 de F1) es la más fiable del rendimiento esperado.

### Otros cambios de mantenimiento

* Reproducibilidad: el notebook registra las versiones de Python y de las librerías en
  `fase-1/metricas_modelo.json`; `modelo.joblib` solo es válido con la versión de scikit-learn con la
  que se generó, por lo que debe regenerarse ejecutando el notebook tras `uv sync`.
* Se corrigieron avisos de deprecación de seaborn y afirmaciones imprecisas en el EDA y en la
  reflexión técnica.
* Quedan pendientes los aportes y revisiones de cada integrante en GitHub (trazabilidad), que se
  registrarán en el repositorio mediante pull requests revisados.
