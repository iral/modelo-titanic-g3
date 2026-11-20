# Fase 3 — API del modelo Titanic (consulta y entrenamiento por HTTP)

## 1. Descripción

API FastAPI que expone el modelo de supervivencia del Titanic (`Survived`) construido en las fases
anteriores. Permite consultar el estado, predecir con una versión guardada y **entrenar una versión
nueva** mediante HTTP. No reimplementa nada: `phase2_imports.py` reutiliza `train_version`,
`load_model` y la validación de esquema de `fase-2/`.

Tecnologías: Python 3.13, FastAPI, Uvicorn, scikit-learn, pandas, httpx (cliente), pytest, Docker, uv.

## 2. Estructura de `fase-3/`

| Archivo | Función |
|---|---|
| `api.py` | Rutas `GET /health`, `POST /predict`, `POST /train` y mapeo de errores a HTTP. |
| `training_service.py` | Validación de la solicitud de entrenamiento, asignación de versión y publicación atómica. |
| `phase2_imports.py` | Puente hacia el código de la Fase 2, incluida la función común `predict_dataframe`. |
| `client.py` | Cliente HTTP de demostración. |
| `tests/test_api.py`, `tests/test_train.py` | Pruebas de `/health`, `/predict` y `/train` + integración. |
| `Dockerfile`, `requirements.txt`, `pytest.ini` | Contenedor, dependencias fijadas y configuración de pytest. |

## 3. Instalación

Requisitos: Python 3.13 y [uv](https://docs.astral.sh/uv/). Desde la **raíz del repositorio**:

```bash
uv sync
```

Datos necesarios: `data/raw/train.csv` (entrenamiento) y `fase-2/models/v1` (modelo inicial).

## 4. Ejecución local

Desde `fase-3/`:

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

API en `http://localhost:8000` (documentación interactiva en `/docs`).

Variables de entorno:

| Variable | Defecto | Efecto |
|---|---|---|
| `MODEL_VERSION` | `v1` | Versión que usa `/predict`. |
| `MODELS_DIR` | `fase-2/models` | Carpeta de versiones (relativa a la raíz del repo o absoluta). |
| `DATA_DIR` | `data` | Directorio raíz permitido para los datos. Sin `data_path`, `/train` busca `DATA_DIR/raw/train.csv`; una ruta explícita debe permanecer dentro de `DATA_DIR`. |

## 5. Endpoints

### GET /health
Sin parámetros. `200` → `{"status": "ok", "version": "v1"}`.

### POST /predict
Cuerpo JSON con los siete campos (todos obligatorios; los nulos se envían como `null`; no se admiten
campos extra):

```json
{"Pclass": 3, "Age": 22.0, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Sex": "male", "Embarked": "S"}
```
`200` → `{"prediccion": 0, "version": "v1"}` · `422` campo ausente, tipo incorrecto, rango inválido, categoría no permitida o campo extra ·
`503` la versión configurada no existe o no se pudo cargar.

### POST /train
Cuerpo **opcional** (puede enviarse vacío; entonces usa `data/raw/train.csv`, `Survived` y la semilla
por defecto de la Fase 2):

| Campo | Tipo | Descripción |
|---|---|---|
| `data_path` | string | CSV dentro de `DATA_DIR` (relativo a la raíz del repo o absoluto). |
| `target` | string | Variable objetivo (por defecto `Survived`). |
| `seed` | entero 0–4294967295 | Semilla aleatoria. |

No se aceptan otros campos: los hiperparámetros y la selección por F1 con validación cruzada son
los de la Fase 2 y no se pueden alterar por HTTP. El único target permitido es `Survived`; sus valores
deben ser binarios (`0` y `1`), sin nulos y con ambas clases presentes.

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d '{"seed": 42}'
```
`201` →
```json
{"mensaje": "Entrenamiento completado", "version_activa": "v1", "version": "v3",
 "variable_objetivo": "Survived", "semilla": 42, "fecha_creacion": "…",
 "metricas": {"modelo": {…}, "modelo_base": {…}, "metrica_principal": "f1"}}
```

| Código | Causa |
|---|---|
| 404 | El archivo de datos no existe. |
| 422 | Parámetro inválido, ruta fuera de `DATA_DIR`, columnas ausentes o datos ilegibles. |
| 500 | Fallo de entrenamiento o al guardar; el mensaje no expone detalles internos (se registran en el log). |

## 6. Versionamiento

- Cada `POST /train` crea la siguiente versión libre (`v1`, `v2`, …) dentro de `MODELS_DIR`.
- Se entrena en una carpeta temporal `.tmp-<id>` y solo al terminar se publica con un único cambio
  de nombre: nunca hay versiones a medias, y las existentes no se tocan ni se sobrescriben.
- Cada versión contiene `modelo.joblib` (pipeline con preprocesamiento + Random Forest),
  `metrics.json` y `metadata.json` (versión, fecha, objetivo, semilla, hiperparámetros, métricas,
  datos, librerías, commit).
- **Una versión nueva no se activa sola.** Para usarla: reiniciar con `MODEL_VERSION=v3`.
  `/predict` responde siempre con la versión realmente usada.
- Si `MODEL_VERSION` no existe, `/predict` devuelve `503`; nunca se usa otra versión en silencio.

## 7. Docker

Desde la **raíz del repositorio**:

```bash
docker build -f fase-3/Dockerfile -t titanic-api .
docker run --rm -p 8000:8000 -v "$(pwd)/fase-2/models:/app/fase-2/models" titanic-api
# usar otra versión:
docker run --rm -p 8000:8000 -e MODEL_VERSION=v3 -v "$(pwd)/fase-2/models:/app/fase-2/models" titanic-api
```
El volumen `models` hace que las versiones creadas por `/train` sobrevivan al contenedor. Los datos
(`data/raw`) viajan dentro de la imagen; para usar otros, móntalos en `/app/data`. El comando de
inicio es `uvicorn api:app --host 0.0.0.0 --port 8000`. Dentro de la imagen no hay `git`, por lo que
`commit_codigo` queda como `desconocido`.

## 8. Pruebas

La suite de esta entrega contiene 58 pruebas (29 en `test_api.py` y 29 en `test_train.py`).
La suite de la Fase 2 contiene 9 pruebas. En conjunto se descubren 67 pruebas.
Desde la raíz se pueden ejecutar conjuntamente ambas fases con `uv run python -m pytest`.
Desde `fase-3/` se ejecuta solo la suite de esta fase (las de la Fase 2 se ejecutan aparte desde `fase-2/`):

```bash
uv run python -m pytest
```
- `test_api.py`: health, versión/rutas, predicción válida, nulos, campos ausentes, tipos, rangos, categorías, campo extra, 503 y modelo real.
- `test_train.py`: versión nueva y artefactos, cuerpo vacío, no sobrescritura, `MODEL_VERSION` intacto,
  targets no permitidos, targets nulos/no binarios, rutas y `DATA_DIR` personalizado, validaciones
  (422/404), fallos sin versión parcial y el ciclo entrenar → seleccionar versión → predecir. El entrenamiento es el real de
  la Fase 2 con 10 árboles por velocidad; solo los fallos se simulan.

## 9. Cliente de prueba

Con la API en marcha, desde `fase-3/`:

```bash
uv run python client.py            # health → predict → train → predict
uv run python client.py --no-train # solo health y predict
```
Usa el primer registro de `fase-2/data/ejemplo_prediccion.csv` y solo envía solicitudes HTTP. Para ver
la versión nueva en acción: reiniciar la API con `MODEL_VERSION=<nueva>` y ejecutar `client.py --no-train`.
