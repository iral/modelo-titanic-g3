# Fase 4 — Monitoreo de la API del Titanic

## 1. Descripción

La Fase 4 conserva la API de la Fase 3 y añade un **registro persistente** de cada
llamada a `POST /predict`, incluidas las rechazadas por validación. También incluye
un script que calcula dos indicadores operativos: total de solicitudes y porcentaje
de fallos (HTTP 4xx/5xx).

La API no se reimplementa: `fase-4/api.py` carga la aplicación de `fase-3/api.py`
y añade un middleware que escribe una fila por intento en un CSV.

## 2. Estructura de `fase-4/`

| Archivo | Función |
|---|---|
| `api.py` | Carga la app de Fase 3 y registra cada `POST /predict` en CSV. |
| `monitoreo.py` | Lee el registro y muestra total de solicitudes y % de fallos. |
| `tests/test_monitoreo.py` | Pruebas del registro (200 y 422) y del script. |
| `logs/registro_ejemplo.csv` | Muestra del ensayo controlado (3 solicitudes). |
| `Dockerfile`, `requirements.txt`, `pytest.ini` | Contenedor, dependencias y pytest. |

## 3. Inicio local

Desde `fase-4/`:

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

Variables de entorno:

| Variable | Defecto | Efecto |
|---|---|---|
| `MODEL_VERSION` | `v1` | Versión usada por `/predict`. |
| `MODELS_DIR` | `fase-2/models` | Carpeta de versiones (relativa a la raíz o absoluta). |
| `DATA_DIR` | `data` | Directorio permitido para los datos de `/train`. |
| `LOG_PATH` | `fase-4/logs/registro_predicciones.csv` | Archivo CSV de registro. |

El registro se abre en modo **append**: el encabezado se escribe solo al crear el
archivo y el historial se conserva entre ejecuciones.

## 4. Formato del registro

```csv
fecha_hora,estado_http,version,prediccion
2026-11-22T15:00:36+00:00,200,v1,0
2026-11-22T15:00:36+00:00,200,v1,1
2026-11-22T15:00:36+00:00,422,,
```

* Una fila por cada `POST /predict`, también cuando FastAPI responde `422`.
* En un `422` no se usó el modelo: `version` y `prediccion` quedan vacías.
* En un `200` se guardan la versión que realmente respondió y la etiqueta `0`/`1`.

## 5. Métricas operativas

```bash
uv run python monitoreo.py
uv run python monitoreo.py --log logs/registro_ejemplo.csv
```

Salida del ensayo controlado:

```text
Total de solicitudes: 3
Porcentaje de fallos: 33,3%
```

Si el archivo no existe o no tiene filas:

```text
Total de solicitudes: 0
Porcentaje de fallos: sin solicitudes
```

## 6. Ensayo controlado y Reporte

Se arrancó la API con `MODEL_VERSION=v1` y un registro vacío
(`LOG_PATH=fase-4/logs/registro_ejemplo.csv`). Se enviaron:

1. Una solicitud válida (`Pclass=3`, `Age=22`, `Sex=male`, …) → `200`, predicción `0`.
2. Otra solicitud válida (`Pclass=1`, `Sex=female`, …) → `200`, predicción `1`.
3. Una solicitud inválida sin `Age` → `422`, con el campo faltante identificado.

El resultado quedó en `logs/registro_ejemplo.csv` y el script reportó **3
solicitudes** y **33,3 % de fallos** (1 de 3). El fallo fue intencional: la prueba
de validación confirma que la API rechaza entradas incompletas y que el rechazo
también queda registrado. Los dos casos válidos fueron procesados con `v1`.

## 7. Versiones y rollback

* Versión activa del ensayo: **`v1`**.
* Segunda versión disponible: **`v2`** en `fase-2/models/`.
* `POST /train` crea la siguiente versión libre (`v3`, `v4`, …) sin sobrescribir
  las anteriores y **no la activa automáticamente**.

Para usar otra versión, se reinicia la API cambiando la configuración, sin tocar
código:

```bash
# activar v2
MODEL_VERSION=v2 uv run uvicorn api:app --host 0.0.0.0 --port 8000

# volver a v1
MODEL_VERSION=v1 uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

En Docker se usa `-e MODEL_VERSION=v2`. Si la versión configurada no existe,
`/predict` responde `503` y nunca usa otra versión en silencio.

## 8. Política de reentrenamiento

Cuando se acumulen al menos **100 predicciones nuevas con resultado real
verificado**, el integrante responsable del modelo ejecutará `POST /train` para
generar una versión candidata. La candidata y la versión activa se compararán con
la misma métrica principal de la Fase 2 (**F1-Score**) sobre el mismo conjunto de
prueba reservado; la nueva versión solo se activa si mejora el F1 de la versión
activa. Si no mejora, se conserva la versión actual y se documenta el resultado.

## 9. Docker

Desde la **raíz del repositorio**:

```bash
docker build -f fase-4/Dockerfile -t titanic-monitoreo .
docker run --rm -p 8000:8000 \
  -e MODEL_VERSION=v1 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/fase-2/models:/app/fase-2/models" \
  -v "$(pwd)/fase-4/logs:/app/fase-4/logs" \
  titanic-monitoreo
```

El volumen de `logs` es indispensable: hace que el registro sobreviva al
contenedor. El de `models` conserva las versiones creadas por `/train` y el de
`data` permite reentrenar con `data/raw/train.csv`.

## 10. Pruebas

Desde `fase-4/`:

```bash
uv run python -m pytest
```

Se ejecutan 5 pruebas: registro de una fila por intento (incluido `422`), historial
sin encabezados repetidos, conteo de fallos 4xx/5xx y los dos casos borde del
script (registro vacío y ensayo 2 válidas + 1 inválida).

## 11. Nota de compatibilidad

Al igual que la Fase 3, se usa Python 3.13 porque las versiones fijadas de
`pandas==3.0.5` y `scikit-learn==1.9.0` no publican *wheels* para Python 3.15.
Cambiar de Python obligaría a cambiar las versiones con las que se guardaron los
modelos de la Fase 2.
