# Fase 2 — Aplicación Reutilizable (Entrenamiento y Predicción)

## Integrantes

* Santiago Acevedo González
* Ivan Rene Acosta Lallemand
* Daniela Andrea Gallego Díaz

**Docente:** Andrés Parra — *Modelos y Simulación de Sistemas I*

## 1. Problema y relación con la Fase 1

Se conserva el problema de la Fase 1: predecir `Survived` (supervivencia de un pasajero del Titanic)
con datos de la competencia *Titanic - Machine Learning from Disaster* de Kaggle. La Fase 2
convierte el notebook en una aplicación ejecutable por línea de comandos: `train.py` entrena y
versiona modelos, `predict.py` predice con una versión guardada. La carpeta `fase-1/` se mantiene
intacta como entrega anterior.

**Cambios respecto a la Fase 1**

* El preprocesamiento (imputación + codificación) ya no vive en el notebook: se define una sola vez
  en `src/pipeline.py` y se guarda dentro del `Pipeline` serializado.
* La lectura, validación, partición, métricas y almacenamiento se separaron en módulos (`src/`).
* El modelo ahora se versiona: cada ejecución crea `models/<version>/` con artefacto, métricas y
  metadatos, y se rechaza un identificador repetido.
* Se añadieron validaciones de esquema, tipos y manejo explícito de errores con códigos de salida.
* El entrenamiento y la predicción corren en local y dentro de Docker, con persistencia de salidas.

## 2. Requisitos

* **Python 3.13** (probado con 3.13.0). La imagen Docker usa `python:3.13.0-slim`.
* Dependencias fijadas en `requirements.txt`:

| Paquete | Versión |
|---------|--------:|
| `pandas` | 3.0.5 |
| `numpy` | 2.5.2 |
| `scikit-learn` | 1.9.0 |
| `joblib` | 1.6.0 |
| `pytest` | 9.1.1 |

Instalación desde `fase-2/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

También se puede usar `uv`:

```powershell
uv venv .venv --python 3.13
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Todos los comandos de esta guía se ejecutan **desde la carpeta `fase-2/`** en PowerShell.

## 3. Datos

* **Fuente:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).
* **Datos etiquetados:** `data/raw/train.csv` en la raíz del repositorio (891 filas, columna
  `Survived`). No se modifican los datos originales.
* **Archivo de ejemplo para predecir:** `fase-2/data/ejemplo_prediccion.csv` (9 filas, esquema real
  sin `Survived`; incluye celdas nulas en `Age` y `Embarked`).
* **Esquema requerido para predecir:** `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `Sex`,
  `Embarked`. Las columnas adicionales (`PassengerId`, `Name`, `Ticket`, `Cabin`) se ignoran y se
  informan por consola. Las celdas nulas en las variables anteriores son válidas: el pipeline
  imputa `Age`/`Fare` con la mediana y `Sex`/`Embarked` con la categoría más frecuente.
* `Survived` no es necesaria al predecir; si viene en el CSV, se ignora como columna adicional.

## 4. Estructura

```
fase-2/
├── train.py                   # Entrenamiento y versionado
├── predict.py                 # Predicción con una versión guardada
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── pytest.ini
├── src/
│   ├── config.py              # Variables, semilla y parámetros por defecto
│   ├── data.py                # Lectura, validación y partición
│   ├── pipeline.py            # ColumnTransformer + RandomForest
│   ├── metrics.py             # Accuracy, precision, recall, F1 y ROC-AUC
│   ├── artifacts.py           # Guardado/carga de versiones
│   └── errors.py              # Errores de aplicación
├── tests/
│   └── test_aplicacion.py     # 7 pruebas con pytest
├── data/
│   └── ejemplo_prediccion.csv
├── models/
│   ├── v1/                    # modelo.joblib, metrics.json, metadata.json
│   └── v2/
└── outputs/                   # Predicciones generadas (no versionado)
```

## 5. Entrenamiento

```powershell
python train.py --data ..\data\raw\train.csv --target Survived --output-dir models --version v1 --seed 42
```

Opciones disponibles con `python train.py --help`: `--data`, `--target`, `--output-dir`,
`--version`, `--seed`, `--test-size`, `--n-estimators`, `--max-depth`, `--min-samples-leaf`,
`--author`.

El script:

1. Valida el archivo, la variable objetivo y las columnas requeridas.
2. Separa entrenamiento/prueba (20%, estratificado, `random_state=42`) **antes** de ajustar
   transformaciones.
3. Ajusta el `Pipeline` (imputación + `OneHotEncoder` + `RandomForestClassifier`) solo con
   entrenamiento y evalúa en el conjunto de prueba retenido.
4. Entrena y evalúa el `DummyClassifier` (`most_frequent`) sobre la misma partición.
5. Genera `models/<version>/modelo.joblib`, `metrics.json` y `metadata.json`.
6. Falla con código distinto de cero si la versión ya existe, sin sobrescribirla.

## 6. Predicción

```powershell
python predict.py --model models\v1\modelo.joblib --input data\ejemplo_prediccion.csv --output outputs\predicciones.csv
```

`--model` acepta la ruta del `modelo.joblib` o la carpeta de la versión (`models\v1`).

Salida: un CSV con `PassengerId` (si viene en la entrada), `prediccion` (0 = no sobrevivió,
1 = sobrevivió) y `probabilidad_1` (probabilidad opcional de supervivencia). Se conserva
estrictamente el orden de las filas de entrada. Si falta una columna obligatoria o hay un tipo
incompatible, no se escribe el archivo de salida y el proceso termina con código 1.

## 7. Versiones disponibles

| Versión | Algoritmo | Hiperparámetros | Accuracy | F1 | ROC-AUC |
|---------|-----------|-----------------|---------:|----:|--------:|
| `v1` | RandomForest | 100 árboles, `max_depth=5`, `min_samples_leaf=2` | 0.8156 | 0.7179 | 0.8356 |
| `v2` | RandomForest | 200 árboles, `max_depth=7`, `min_samples_leaf=1` | 0.8045 | 0.7059 | 0.8443 |
| Base (`most_frequent`) | DummyClassifier | — | 0.6145 | 0.0000 | 0.5000 |

Ambas versiones se generaron con `train.py` (semilla 42) y pueden cargarse indistintamente:

```powershell
python predict.py --model models\v2 --input data\ejemplo_prediccion.csv --output outputs\predicciones_v2.csv
```

Cada versión conserva sus métricas (`metrics.json`) y metadatos (`metadata.json`: fecha, variables y
tipos, hiperparámetros, semilla, tamaño de la partición, versiones de librerías, commit del código y
autor).

## 8. Docker

Construcción desde `fase-2/`:

```powershell
docker build -t modelos-fase2:1.0 .
```

Entrenamiento (los datos y los modelos se montan desde el equipo anfitrión para que persistan):

```powershell
docker run --rm -v "${PWD}\..\data\raw:/app/data:ro" -v "${PWD}\models:/app/models" modelos-fase2:1.0 python train.py --data data/train.csv --target Survived --output-dir models --version v-docker --seed 42
```

Predicción:

```powershell
docker run --rm -v "${PWD}\models:/app/models:ro" -v "${PWD}\outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_docker.csv
```

La imagen no entrena durante `docker build`, no define `ENTRYPOINT` (los comandos anteriores se
ejecutan tal cual) y `models/`, `outputs/` y los datos completos quedan fuera de la imagen mediante
`.dockerignore`. Los artefactos creados dentro del contenedor aparecen en `models/` y `outputs/` del
host.

## 9. Pruebas

Desde `fase-2/`:

```powershell
python -m pytest
```

Resultado esperado: **7 passed**. Las pruebas usan datos sintéticos y `tmp_path` (no tocan `models/`
ni requieren red):

1. `test_entrenamiento_genera_artefactos`: genera `modelo.joblib`, `metrics.json` y
   `metadata.json` con métricas del modelo y del baseline.
2. `test_prediccion_consistente_y_una_por_fila`: una predicción por fila, mismo orden de
   `PassengerId` y coincidencia con el modelo en memoria.
3. `test_valores_faltantes_no_eliminan_filas`: nulos en `Age`, `Fare` y `Embarked` se imputan sin
   perder filas.
4. `test_columna_faltante_falla_con_error_claro`: `DataValidationError` y código de salida distinto
   de cero, sin archivo de salida.
5. `test_versionamiento_no_sobrescribe`: `v1` y `v2` conviven; un identificador repetido no modifica
   los archivos existentes.
6. `test_columnas_extra_ignoradas_y_orden_irrelevante`: columnas adicionales y orden distinto no
   cambian las predicciones.
7. `test_cli_entrenamiento_genera_version`: el comando de entrenamiento funciona y el duplicado
   falla con mensaje explícito.

## 10. Resultados y limitaciones

El modelo `v1` reproduce las métricas de la Fase 1 sobre 179 pasajeros de prueba: accuracy 0.8156,
precision 0.8750, recall 0.6087, F1 0.7179 y ROC-AUC 0.8356, frente a 0.6145 de accuracy y F1 0.0
del modelo base. El F1 es la métrica principal por el desbalance de clases (38.4% de
supervivientes).

Limitaciones conocidas:

* Solo 891 registros de entrenamiento; las métricas tienen varianza alta con particiones pequeñas.
* El recall (~0.61) deja fuera a cerca de uno de cada tres supervivientes reales.
* Se descartan `Name`, `Ticket` y `Cabin` (información potencialmente útil).
* El umbral de decisión se mantiene en 0.5.
* La reproducibilidad exacta requiere el mismo entorno; pequeñas diferencias numéricas entre
  plataformas son esperables.

## 11. Evidencia de ejecución

```text
> python train.py --data ..\data\raw\train.csv --target Survived --output-dir models --version v1 --seed 42
Versión entrenada: v1
  Modelo : models\v1\modelo.joblib
  Métricas: models\v1\metrics.json
  Metadatos: models\v1\metadata.json
  RandomForest  accuracy=0.8156  f1=0.7179  roc_auc=0.8356389986824769
  Baseline      accuracy=0.6145  f1=0.0000  roc_auc=0.5

> python predict.py --model models\v1\modelo.joblib --input data\ejemplo_prediccion.csv --output outputs\predicciones.csv
Columnas ignoradas (no forman parte del esquema del modelo): PassengerId, Name, Ticket, Cabin
Modelo utilizado: models\v1\modelo.joblib
Filas procesadas: 9
Archivo generado: outputs\predicciones.csv
Supervivientes predichos: 5

> python -m pytest
7 passed in 23.75s

> docker build -t modelos-fase2:1.0 .
# imagen construida correctamente
> docker run --rm ... python train.py --data data/train.csv ... --version v-docker --seed 42
RandomForest  accuracy=0.8156  f1=0.7179  roc_auc=0.8356389986824769
> docker run --rm ... python predict.py --model models/v-docker/modelo.joblib ...
Filas procesadas: 9 / Supervivientes predichos: 5
```

## 12. Identificación de la entrega

La versión entregada se identifica con la etiqueta `entrega-2`, creada sobre `main` al integrar
esta fase:

```powershell
git tag -a entrega-2 -m "Entrega 2 - Fase 2"
git push origin entrega-2
```

## 13. Fuentes

* Kaggle — *Titanic: Machine Learning from Disaster* (datos y enunciado).
* Documentación de [scikit-learn](https://scikit-learn.org/stable/) (`Pipeline`, `ColumnTransformer`,
  `RandomForestClassifier`, métricas).
* Especificación técnica de la Fase 2 del Proyecto Integrador (2026-II).
