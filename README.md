# Modelo Predictivo de Supervivencia en el Titanic

## 1. Integrantes del Equipo
* **Santiago Acevedo González**
* **Ivan Rene Acosta Lallemand**
* **Daniela Andrea Gallego Díaz**

## 2. Descripción

Proyecto de Machine Learning supervisado que predice la supervivencia de pasajeros del Titanic
(`Survived`) a partir de sus características personales y de viaje. La **Fase 1** (análisis,
modelado y evaluación) está documentada en [`fase-1/README.md`](fase-1/README.md); la **Fase 2**
convierte ese trabajo en una aplicación de línea de comandos con versionado de modelos, pruebas y
Docker, documentada en [`fase-2/README.md`](fase-2/README.md). La **Fase 3** expone el modelo con
una API REST (FastAPI) y la **Fase 4** registra y monitorea su uso, documentadas en
[`fase-3/README.md`](fase-3/README.md) y [`fase-4/README.md`](fase-4/README.md).

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
│   ├── modelo.joblib           # Pipeline completo entrenado (preprocesamiento + Random Forest)
│   ├── metricas_base.json      # Métricas del modelo base en el conjunto de prueba
│   └── metricas_modelo.json    # Métricas del modelo predictivo en el conjunto de prueba
├── fase-2/
│   ├── README.md               # Guía de la aplicación (comandos, Docker y pruebas)
│   ├── train.py                # Entrenamiento versionado
│   ├── predict.py              # Predicción con una versión guardada
│   ├── src/                    # Lógica reutilizable (datos, pipeline, métricas, artefactos)
│   ├── tests/                  # Pruebas automatizadas con pytest
│   ├── models/                 # Versiones v1 y v2 del modelo
│   ├── data/                   # CSV de ejemplo sin la variable objetivo
│   ├── Dockerfile              # Imagen para ejecutar entrenamiento y predicción
│   └── requirements.txt        # Dependencias con versiones explícitas
├── fase-3/
│   ├── README.md               # Guía de la API: endpoints, versiones, Docker y pruebas
│   ├── api.py                  # FastAPI: GET /health, POST /predict, POST /train
│   ├── training_service.py     # Entrenamiento versionado y publicación atómica
│   ├── phase2_imports.py       # Puente hacia las funciones de la Fase 2
│   ├── client.py               # Cliente HTTP de demostración
│   ├── tests/                  # Pruebas de la API y del entrenamiento
│   ├── Dockerfile              # Imagen de la API (uvicorn en 0.0.0.0:8000)
│   └── requirements.txt        # Dependencias con versiones explícitas
├── fase-4/
│   ├── README.md               # Registro, monitoreo, política de reentrenamiento y reporte
│   ├── api.py                  # Fase 3 + registro CSV de cada POST /predict
│   ├── monitoreo.py            # Total de solicitudes y porcentaje de fallos
│   ├── tests/                  # Pruebas del registro y del monitoreo
│   ├── logs/                   # Registro persistente (muestra incluida)
│   ├── Dockerfile              # Imagen de la API con registro persistente
│   └── requirements.txt        # Dependencias con versiones explícitas
├── .dockerignore               # Exclusiones comunes de las imágenes
├── pyproject.toml              # Dependencias de las cuatro fases (uv)
├── uv.lock                     # Bloqueo de versiones de dependencias
└── README.md
```

## 4. Cómo ejecutar el proyecto (Fase 1)

### 4.1 Requisitos
* Python 3.13 (probado con 3.13.0).
* [uv](https://docs.astral.sh/uv/) para la gestión del entorno virtual y las dependencias.

### 4.2 Instalación
Desde la raíz del proyecto:
```bash
uv sync
```
Crea el entorno virtual (`.venv/`) e instala las dependencias de las dos fases, las herramientas para
ejecutar el notebook y `pytest`.

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
`Verificación exitosa: el pipeline guardado se cargó y predijo correctamente.`

**Reproducibilidad:** todas las decisiones aleatorias del notebook (división de datos, validación
cruzada, bosque aleatorio y muestra de verificación) usan la constante `SEED = 42`, fijada en la
celda de configuración y propagada como `random_state` a cada estimador. Con el dataset original sin
modificar, dos ejecuciones completas producen las mismas métricas publicadas en
[`fase-1/README.md`](fase-1/README.md).

## 5. Fase 2 (aplicación reutilizable)

La Fase 2 vive en `fase-2/` y comparte el entorno gestionado por `uv` (`pyproject.toml` en la raíz,
Python 3.13); `fase-2/requirements.txt` mantiene las mismas versiones para Docker y `pip`. Desde la
raíz:

```powershell
uv sync
```

Y desde `fase-2/`:

```powershell
uv run python train.py --target Survived --output-dir models --version v1 --seed 42
uv run python predict.py --model models\v1 --input data\ejemplo_prediccion.csv --output outputs\predicciones.csv
uv run python -m pytest
```

`train.py` usa por defecto el dataset existente `data/raw/train.csv`; no hace falta copiarlo. Incluye
entrenamiento versionado (`models/v1`, `models/v2`), predicción con validación de esquema, 9 pruebas
automatizadas y ejecución con Docker. La guía completa está en [`fase-2/README.md`](fase-2/README.md).

## 6. Fase 3 (API REST)

Desde `fase-3/`, con el modelo `v1` de la Fase 2:

```powershell
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"v1"}

curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"Pclass":3,"Age":22.0,"SibSp":1,"Parch":0,"Fare":7.25,"Sex":"male","Embarked":"S"}'
# {"prediccion":0,"version":"v1"}

curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d '{"seed":42}'
# {"mensaje":"Entrenamiento completado","version_activa":"v1","version":"v3", ...}
```

La versión activa se cambia con `MODEL_VERSION` (por ejemplo, `MODEL_VERSION=v2`) sin tocar el
código. `POST /train` crea una versión nueva y no la activa automáticamente. La guía completa está
en [`fase-3/README.md`](fase-3/README.md).

## 7. Fase 4 (monitoreo)

Desde `fase-4/`, la API registra cada `POST /predict` (incluidos los `422`) en
`logs/registro_predicciones.csv`:

```powershell
uv run uvicorn api:app --host 0.0.0.0 --port 8000
uv run python monitoreo.py
```

Salida del ensayo incluido en el repositorio:

```text
Total de solicitudes: 3
Porcentaje de fallos: 33,3%
```

La guía completa, la política de reentrenamiento y el reporte están en
[`fase-4/README.md`](fase-4/README.md).

## 8. Documentación

| Documento | Contenido |
|-----------|-----------|
| [`fase-1/README.md`](fase-1/README.md) | Problema, datos, preprocesamiento, modelo base, algoritmo definitivo, resultados finales y modelo almacenado. |
| [`fase-2/README.md`](fase-2/README.md) | Aplicación reutilizable: instalación, comandos de entrenamiento y predicción, versiones, Docker, pruebas y evidencias. |
| [`fase-3/README.md`](fase-3/README.md) | API REST: endpoints, selección de versión, Docker, cliente de prueba y pruebas automatizadas. |
| [`fase-4/README.md`](fase-4/README.md) | Registro persistente, métricas operativas, ensayo controlado, política de reentrenamiento y reporte. |
| [`docs/TECNICO.md`](docs/TECNICO.md) | Guía exhaustiva: requisitos, esquema del dataset, ejecución, reproducibilidad y resolución de problemas. |
| `fase-1/notebook.ipynb` | Notebook completo y ejecutado, con la reflexión técnica final. |

## 9. Entrega final

* Ramas integradas: `main` ← `develop` ← `feature/*`.
* Carpetas incluidas y reproducibles: `fase-1/`, `fase-2/`, `fase-3/` y `fase-4/`.
