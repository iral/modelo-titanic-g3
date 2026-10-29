# Fase 2 — Aplicación Reutilizable (Entrenamiento y Predicción)

## Integrantes

* Santiago Acevedo González
* Ivan Rene Acosta Lallemand
* Daniela Andrea Gallego Díaz

**Docente:** Andrés Parra — *Modelos y Simulación de Sistemas I*

---

## 1. Problema y relación con la Fase 1

Se conserva el problema de la Fase 1: predecir `Survived` (supervivencia de un pasajero del Titanic) con datos de la competencia *Titanic - Machine Learning from Disaster* de Kaggle. La Fase 2 trasciende el trabajo estático de un notebook y lo convierte en una aplicación robusta, modular y ejecutable por línea de comandos: `train.py` se encarga del entrenamiento, validación cruzada y versionado estructurado de modelos, mientras que `predict.py` permite cargar cualquier versión almacenada para generar inferencias sobre datos nuevos de manera limpia y automatizada. La carpeta `fase-1/` se mantiene completamente intacta como evidencia de la entrega anterior.

**Cambios profundos y mejoras técnicas respecto a la Fase 1:**
* **Modularización de la lógica:** Todo el código fuente se desacopló y organizó en la carpeta `src/`, separando responsabilidades críticas (configuraciones, manejo y validación de datos, construcción del pipeline de scikit-learn, cálculo de métricas avanzadas, gestión de artefactos y manejo centralizado de errores).
* **Eliminación de la fuga de información (*Data Leakage*):** El preprocesamiento (imputación de valores nulos y codificación de variables categóricas) ya no vive de forma dispersa en celdas de un notebook; se define de manera centralizada en `src/pipeline.py` dentro de un objeto `Pipeline` serializable de Scikit-learn, asegurando que las reglas estadísticas (como medianas y modas) se aprendan estrictamente sobre el conjunto de entrenamiento y se apliquen de forma idéntica durante la predicción.
* **Versionamiento estricto de artefactos:** Cada ejecución exitosa de entrenamiento genera una estructura de carpetas aislada en `models/<version>/` que almacena de forma independiente el binario del modelo (`modelo.joblib`), las métricas de evaluación (`metrics.json`) y una ficha técnica completa de metadatos (`metadata.json`). El sistema rechaza automáticamente identificadores duplicados para prevenir sobreescrituras accidentales.
* **Robustez y validación defensiva:** Se implementaron validaciones estrictas de esquemas de datos, tipos y rangos esperados. Ante cualquier inconsistencia (como una columna obligatoria faltante), el sistema aborta la ejecución de forma controlada, emitiendo un mensaje de error claro y devolviendo un código de salida distinto de cero, evitando generar archivos de salida corruptos o incompletos.
* **Portabilidad y Contenerización:** La aplicación está diseñada para ejecutarse sin fricciones tanto en entornos locales (PowerShell / Bash) como dentro de un contenedor Docker aislado, garantizando la persistencia de los resultados hacia el host anfitrión mediante el uso de montajes de volúmenes.

---

## 2. Requisitos del Sistema y Entorno

* **Python 3.13** (probado y verificado con la versión 3.13.0, registrada formalmente en los metadatos de entrenamiento). La imagen oficial de Docker utiliza la variante optimizada `python:3.13.0-slim`.
* **Dependencias explícitas fijadas** en `requirements.txt`:

| Paquete | Versión Exacta | Propósito en la Aplicación |
|---------|---------------:|---------------------------|
| `pandas` | 3.0.5 | Manipulación eficiente y estructuración de DataFrames tabulares. |
| `numpy` | 2.5.2 | Operaciones numéricas y soporte vectorial subyacente. |
| `scikit-learn` | 1.9.0 | Construcción del pipeline, transformadores, modelos de ensamblaje (`RandomForestClassifier`) y métricas. |
| `joblib` | 1.6.0 | Serialización y deserialización eficiente de modelos de Machine Learning. |
| `pytest` | 9.1.1 | Framework para la ejecución de pruebas automatizadas y unitarias. |

*Nota técnica:* Las dependencias principales se encuentran declaradas en el archivo `pyproject.toml` en la raíz del repositorio y bloqueadas mediante `uv.lock`. El archivo `requirements.txt` mantiene exactamente las mismas versiones congeladas para asegurar total compatibilidad al construir la imagen Docker o al instalar mediante gestores tradicionales de Python.

### Instrucciones detalladas de instalación y configuración

#### Opción A: Gestión moderna con `uv` (Recomendada)
Si utilizas `uv` (de Astral), el flujo es extremadamente rápido y gestiona entornos virtuales de forma transparente. Ubícate en la raíz del repositorio y ejecuta:

```powershell
uv sync
```

El comando `uv sync` se encarga de instalar todas las dependencias del proyecto, incluyendo el grupo de desarrollo que habilita `pytest`. Una vez sincronizado el entorno, **todos los comandos operativos de esta guía deben ejecutarse desde la carpeta `fase-2/**` utilizando `uv run`, lo cual garantiza que se utilice el intérprete y las librerías correctas configuradas en la raíz:

```powershell
uv run python train.py --help
```

#### Opción B: Entorno virtual nativo (`venv` y `pip`)

Si prefieres utilizar el flujo estándar de Python (el cual emula exactamente el comportamiento interno de la imagen Docker), abre tu terminal en la carpeta `fase-2/` y ejecuta los siguientes comandos paso a paso:

```powershell
# 1. Crear el entorno virtual local (.venv)
python -m venv .venv

# 2. Activar el entorno virtual (en PowerShell)
.\.venv\Scripts\Activate.ps1

# 3. Actualizar pip e instalar las dependencias fijadas
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

*Nota:* Si optas por esta alternativa nativa con `venv`, puedes omitir el prefijo `uv run` en los comandos posteriores de esta guía y ejecutar directamente con `python`.

---

## 3. Fuente y Estructura de los Datos

* **Fuente oficial de origen:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).
* **Dataset de entrenamiento etiquetado:** Se encuentra ubicado en `data/raw/train.csv` en la raíz del repositorio (consta de 891 filas y contiene la columna objetivo obligatoria `Survived`). Esta es la ruta por defecto que utiliza `train.py`, por lo que **no es necesario duplicar ni mover los archivos de datos originales** dentro de la carpeta `fase-2/`.
* **Archivo de ejemplo para pruebas de predicción:** La carpeta incluye un archivo dedicado en `fase-2/data/ejemplo_prediccion.csv` (contiene 9 registros con el esquema exacto requerido, excluyendo la columna `Survived`, e incluye intencionalmente celdas con valores nulos/faltantes en `Age` y `Embarked` para probar la resiliencia del imputador).
* **Esquema de variables de entrada requerido:**
* Variables numéricas: `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`.
* Variables categóricas: `Sex`, `Embarked`.


* **Gestión de columnas adicionales o ruidosas:** Si el archivo CSV de entrada contiene columnas adicionales que no forman parte del esquema del modelo (tales como `PassengerId`, `Name`, `Ticket` o `Cabin`), el script de predicción no fallará: las detecta automáticamente, las ignora de forma segura y emite una advertencia informativa por la consola.
* **Tratamiento de valores faltantes (*Missing Values*):** Las celdas nulas en las variables de entrada son completamente válidas y toleradas. El pipeline integrado procesa automáticamente los nulos numéricos (`Age`, `Fare`) sustituyéndolos por la mediana calculada en el entrenamiento, y los nulos categóricos (`Sex`, `Embarked`) imputándolos con la categoría más frecuente (moda).
* **Variable objetivo:** `Survived` (binaria: 0 para pasajeros que no sobrevivieron, 1 para quienes sobrevivieron). No debe estar presente al momento de ejecutar predicciones; si llegara a incluirse por error en el archivo de inferencia, el script la detecta e ignora sin interrumpir el flujo.

---

## 4. Estructura del Directorio de Trabajo (`fase-2/`)

La arquitectura de archivos y módulos de la aplicación se encuentra organizada de la siguiente manera:

```text
fase-2/
├── train.py                   # Punto de entrada ejecutable para entrenamiento y versionado de modelos
├── predict.py                 # Punto de entrada ejecutable para carga de versiones y generación de inferencias
├── requirements.txt           # Declaración explícita de dependencias y versiones congeladas
├── Dockerfile                 # Instrucciones para la contenerización de la aplicación
├── .dockerignore              # Exclusión de archivos innecesarios al construir la imagen Docker
├── pytest.ini                 # Configuración del framework de pruebas automatizadas
├── src/
│   ├── __init__.py            # Inicializador del paquete interno
│   ├── config.py              # Definición centralizada de constantes, rutas, hiperparámetros y esquemas
│   ├── data.py                # Módulo de lectura, validación tipada de esquemas y partición estratificada
│   ├── pipeline.py            # Construcción unificada del ColumnTransformer, Pipeline y GridSearchCV
│   ├── metrics.py             # Funciones de cálculo para Accuracy, Precision, Recall, F1-Score y ROC-AUC
│   ├── artifacts.py           # Gestión segura de serialización (Joblib) y metadatos/métricas JSON
│   └── errors.py              # Definición de excepciones personalizadas para un manejo limpio de fallos
├── tests/
│   └── test_aplicacion.py     # Suite completa de 7 pruebas automatizadas e independientes con pytest
├── data/
│   └── ejemplo_prediccion.csv # CSV de muestra para validar la predicción sin la variable objetivo
├── models/
│   ├── v1/                    # Carpeta de artefactos de la versión 1 (modelo.joblib, metrics.json, metadata.json)
│   └── v2/                    # Carpeta de artefactos de la versión 2 (modelo.joblib, metrics.json, metadata.json)
└── outputs/                   # Directorio de destino para los archivos CSV de predicciones generadas (no versionado)
```

---

## 5. Script de Entrenamiento (`train.py`)

El script de entrenamiento ejecuta de principio a fin el flujo necesario para construir, evaluar, comparar y versionar un modelo reutilizable.

### Cómo ejecutar el entrenamiento en local

Desde la carpeta `fase-2/` y utilizando el dataset predeterminado del repositorio:

```powershell
uv run python train.py --target Survived --output-dir models --version v1 --seed 42
```

Dado que `train.py` resuelve por defecto la ruta del dataset en `data/raw/train.csv` (apuntando a la raíz del repositorio), el parámetro `--data` es opcional. Sin embargo, si deseas apuntar explícitamente a un archivo de datos personalizado, puedes incluirlo de la siguiente manera:

```powershell
uv run python train.py --data ..\data\raw\train.csv --target Survived --output-dir models --version v1 --seed 42
```

### Opciones de línea de comandos disponibles (`--help`)

Puedes consultar todos los parámetros configurables ejecutando:

```powershell
uv run python train.py --help
```

Entre los argumentos soportados en `train.py` se encuentran: `--data`, `--target`, `--output-dir`, `--version`, `--seed`, `--test-size`, `--n-estimators`, `--max-depth`, `--min-samples-leaf`, `--select-by` (f1, accuracy, recall, precision), `--cv-folds`, `--group-column` (por defecto `Ticket`), y `--author`.

### Detalle del proceso interno que ejecuta `train.py`:

1. **Validación inicial:** Lee el archivo CSV especificado, comprueba su existencia y verifica rigurosamente que la columna objetivo y las variables de entrada requeridas estén presentes.
2. **Partición estricta de datos:** Divide el conjunto de datos en entrenamiento (80%) y prueba (20%) utilizando estratificación por clases y una semilla aleatoria fija (`--seed 42`) **antes** de aplicar cualquier transformación numérica o categórica, previniendo así cualquier tipo de fuga de información.
3. **Ajuste del Pipeline:** Ajusta los imputadores y codificadores del `ColumnTransformer` junto con el estimador `RandomForestClassifier` utilizando únicamente los datos del conjunto de entrenamiento.
4. **Evaluación comparativa (Baseline):** Entrena y evalúa de manera paralela un modelo base de referencia (`DummyClassifier` con estrategia `most_frequent`) sobre exactamente la misma partición de prueba, garantizando una comparación de rendimiento objetiva y consistente con la Fase 1.
5. **Versionamiento y persistencia:** Al finalizar, crea de manera automatizada la carpeta `models/<version>/` y serializa el pipeline completo en `modelo.joblib`, acompañado de sus respectivas métricas detalladas en `metrics.json` y la ficha técnica de trazabilidad en `metadata.json`.
6. **Protección contra sobrescrituras:** Si intentas entrenar una versión especificando un identificador que ya existe en el directorio de salida (`--version v1`), el script interceptará el intento de forma segura, abortará el proceso con un error descriptivo y evitará sobrescribir el trabajo previo.

---

## 6. Script de Predicción (`predict.py`)

El script de predicción carga un modelo previamente versionado y serializado para generar inferencias sobre nuevos lotes de datos de manera rápida, eficiente y sin requerir ningún proceso de reentrenamiento.

### Cómo ejecutar la predicción en local

```powershell
uv run python predict.py --model models\v1\modelo.joblib --input data\ejemplo_prediccion.csv --output outputs\predicciones.csv
```

*Nota flexible sobre el argumento `--model*`: El script está diseñado inteligentemente para aceptar tanto la ruta directa al archivo binario del artefacto (`models\v1\modelo.joblib`) como la ruta general de la carpeta de la versión (`models\v1`), resolviendo el archivo necesario automáticamente.

### Detalle del proceso interno y manejo de salidas:

* **Validación de esquema:** Comprueba que el archivo CSV de entrada exista, sea legible y contenga todas las columnas obligatorias que el modelo espera según su entrenamiento.
* **Normalización de orden:** Reordena y filtra las columnas de entrada según el esquema estricto con el que fue entrenado el modelo, independientemente del orden desorganizado en el que vengan las columnas dentro del CSV del usuario.
* **Conservación de filas e identificadores:** Genera estrictamente una predicción por cada fila válida de entrada, preservando intacto el orden original de los registros y asociándolos con su respectivo `PassengerId` (si está presente).
* **Estructura del archivo CSV resultante:** El archivo generado en la ruta de salida (ej. `outputs\predicciones.csv`) contendrá como mínimo la columna principal `prediccion` (donde `0` significa que el pasajero no sobrevivió y `1` indica supervivencia) junto con la columna opcional `probabilidad_1` (la probabilidad estimada de supervivencia redondeada a 6 decimales).
* **Manejo de errores previsibles:** Si el archivo de entrada carece de una columna obligatoria o presenta incompatibilidades tipográficas graves, el script emitirá un mensaje de error explicativo por consola, finalizará con un código de salida distinto de cero (`exit code 1`) y **se abstendrá por completo de generar un archivo de salida parcial** que pudiera aparentar resultados válidos.

---

## 7. Versiones disponibles

La aplicación permite versionar y comparar múltiples ejecuciones de entrenamiento. A continuación se detallan las versiones generadas y sus métricas de rendimiento evaluadas sobre el conjunto de prueba retenido (179 registros):

| Versión | Algoritmo | Hiperparámetros | Accuracy | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| `v1` | RandomForest | 100 árboles, `max_depth=5`, `min_samples_leaf=2` | 0.8156 | 0.7179 | 0.8356 |
| `v2` | RandomForest | 200 árboles, `max_depth=7`, `min_samples_leaf=1` | 0.8045 | 0.7059 | 0.8443 |
| Base (`most_frequent`) | DummyClassifier | — | 0.6145 | 0.0000 | 0.5000 |

Ambas versiones se generaron con `train.py` (semilla 42) y pueden cargarse indistintamente:

```powershell
uv run python predict.py --model models\v2 --input data\ejemplo_prediccion.csv --output outputs\predicciones_v2.csv
```

Cada versión de modelo almacena de forma independiente su archivo `metrics.json` y su archivo `metadata.json`, el cual incluye trazas detalladas sobre: fecha y hora exactas de creación, tipo de problema, variable objetivo, algoritmo, variables de entrada y sus tipos de datos esperados, hiperparámetros, semilla aleatoria, tamaño de las particiones de entrenamiento y prueba, versiones exactas de las librerías de entorno, hash del commit de Git asociado y el autor del desarrollo.

---

## 8. Contenerización y Ejecución con Docker

La contenerización garantiza que el proyecto corra exactamente igual en cualquier máquina, encapsulando las dependencias de Python dentro de un entorno aislado.

### 1. Construcción de la imagen Docker

Ejecuta el siguiente comando desde la carpeta `fase-2/`:

```powershell
docker build -t modelos-fase2:1.0 .
```

### 2. Ejecución del entrenamiento dentro del contenedor

Para entrenar un modelo dentro de Docker asegurando que los datos de entrada se lean desde el host y los artefactos resultantes persistan en tu equipo local, ejecuta:

```powershell
docker run --rm -v "${PWD}\..\data\raw:/app/data:ro" -v "${PWD}\models:/app/models" modelos-fase2:1.0 python train.py --data data/train.csv --target Survived --output-dir models --version v-docker --seed 42
```

*Explicación técnica del comando:*

* `--rm`: Elimina el contenedor automáticamente una vez finaliza su ejecución para no acaparar recursos.
* `-v "${PWD}\..\data\raw:/app/data:ro"`: Monta la carpeta de datos del host como de solo lectura (`ro`) dentro del directorio `/app/data` del contenedor. Esto es necesario porque el archivo `.dockerignore` excluye deliberadamente los datos pesados de la imagen base por motivos de eficiencia.
* `-v "${PWD}\models:/app/models"`: Monta la carpeta local `models/` para que los artefactos generados por el contenedor queden guardados permanentemente en tu computador anfitrión.

### 3. Ejecución de la predicción dentro del contenedor

Una vez entrenada la versión `v-docker` (o utilizando cualquier versión existente), puedes generar predicciones montando los modelos y la carpeta de salidas:

```powershell
docker run --rm -v "${PWD}\models:/app/models:ro" -v "${PWD}\outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_docker.csv
```

*Garantías de diseño Docker:* La imagen construida **no realiza ningún entrenamiento durante el proceso de compilación (`docker build`)**, no define un `ENTRYPOINT` rígido que obstaculice la flexibilidad de los comandos (por defecto el Dockerfile ejecuta `python train.py --help` si no se especifica otro comando), y mantiene la carpeta de modelos, salidas y datos completamente fuera de la imagen mediante un archivo `.dockerignore` riguroso, logrando una arquitectura limpia, ligera y profesional.

---

## 9. Pruebas automatizadas (`pytest`)

El proyecto cuenta con una suite completa de pruebas automatizadas escritas en `tests/test_aplicacion.py` que validan rigurosamente el comportamiento de la aplicación utilizando datos sintéticos generados al vuelo y directorios temporales (`tmp_path`), garantizando que las pruebas corran de manera totalmente independiente, rápida y sin requerir conexiones de red ni modificar los modelos reales de entrega.   

### Cómo ejecutar las pruebas desde `fase-2/`:

```powershell
uv run python -m pytest
```

### Detalle de las 7 pruebas automatizadas implementadas:

Resultado esperado: **7 passed**. Las pruebas usan datos sintéticos y `tmp_path` (no tocan `models/` ni requieren red):

1. `test_entrenamiento_genera_artefactos`: Valida que al ejecutar el proceso de entrenamiento con un conjunto pequeño de datos sintéticos, se generen y verifiquen correctamente los tres archivos indispensables: `modelo.joblib`, `metrics.json` y `metadata.json`, comprobando la estructura de las métricas tanto del modelo como del baseline.
2. `test_prediccion_consistente_y_una_por_fila`: Comprueba que el artefacto guardado sea capaz de procesar un archivo de entrada y devuelva exactamente una predicción por cada fila de entrada, conservando de forma estricta el orden original de los `PassengerId` y coincidiendo con los resultados del modelo cargado en memoria.   
3. `test_valores_faltantes_no_eliminan_filas`: Verifica que una entrada válida que contenga valores nulos en variables permitidas (como `Age`, `Fare` o `Embarked`) sea procesada correctamente por el pipeline de imputación sin descartar ni eliminar filas silenciosamente del resultado final.
4. `test_columna_faltante_falla_con_error_claro`: Prueba que la ausencia deliberada de una columna obligatoria en el archivo de entrada dispare de forma controlada la excepción personalizada `DataValidationError`, finalizando el proceso por la línea de comandos con un código de salida distinto de cero y evitando la creación de un archivo CSV de salida corrupto.  
5. `test_versionamiento_no_sobrescribe`: Comprueba que múltiples versiones (ej. v1 y v2) puedan convivir pacíficamente en el directorio de modelos, y que intentar registrar un identificador de versión ya existente lance una excepción por `VersionAlreadyExistsError` impidiendo la sobrescritura silenciosa.  
6. `test_columnas_extra_ignoradas_y_orden_irrelevante`: Valida que la presencia de columnas adicionales no esperadas en el archivo de entrada sea ignorada sin generar fallos, y que alterar el orden de las columnas en el CSV no altere en absoluto las predicciones finales obtenidas.  
7. `test_cli_entrenamiento_genera_version`: Ejecuta directamente el script ejecutable `train.py` a través de una llamada CLI simulada, comprobando la correcta creación de los archivos de versión y el manejo de errores ante duplicados por la consola.  

---

## 10. Resultados y limitaciones

El modelo principal (`v1`) demuestra un rendimiento sólido y consistente, superando ampliamente al modelo base (`DummyClassifier`, el cual alcanza un accuracy de 0.6145 y un F1-Score de 0.0 debido al sesgo de la clase mayoritaria de no supervivientes). Con un Accuracy de 0.8156 y un F1-Score de 0.7179, el modelo logra un balance adecuado entre precisión y sensibilidad en un escenario de clases desbalanceadas (donde aproximadamente el 38.4% de los pasajeros del conjunto de referencia corresponden a supervivientes). El AUC de la curva ROC de 0.8356 confirma una excelente capacidad discriminativa general del clasificador

### Limitaciones Conocidas del Sistema  

* Tamaño reducido del conjunto de datos: El dataset original de entrenamiento cuenta con apenas 891 registros, lo que provoca que las métricas de evaluación presenten cierta sensibilidad y varianza ante particiones de prueba reducidas.  
* Margen de mejora en Recall: Si bien la precisión es alta (0.8750), el indicador de recall se sitúa en torno a 0.6087, lo que implica que el modelo aún deja sin identificar (falsos negativos) a cerca de uno de cada tres supervivientes reales.  
* Exclusión de variables textuales ricas: En esta iteración modular basada en atributos tabulares estructurados, se prescindió del procesamiento avanzado de campos textuales complejos como `Name`, `Ticket` y `Cabin`, los cuales podrían aportar señales predictivas adicionales en futuras fases.  
* Umbral de decisión estático: El punto de corte para la clasificación binaria se mantiene por defecto en el umbral estándar de 0.5, sin haber aplicado una optimización de umbral orientada específicamente a maximizar el F1-Score.  
* Sensibilidad de entorno: Aunque se garantiza la reproducibilidad algorítmica mediante la fijación de semillas (`seed=42`), pueden presentarse mínimas variaciones numéricas en los decimales de métricas específicas al ejecutar los scripts en arquitecturas de hardware o sistemas operativos distintos.

---

## 11. Evidencia de ejecución

A continuación se presenta un registro real de la ejecución de los comandos principales del proyecto para verificar su correcto funcionamiento:

```text
> uv run python train.py --target Survived --output-dir models --version v1 --seed 42
Versión entrenada: v1
  Modelo : models\v1\modelo.joblib
  Métricas: models\v1\metrics.json
  Metadatos: models\v1\metadata.json
  RandomForest  accuracy=0.8156  f1=0.7179  roc_auc=0.8356389986824769
  Baseline      accuracy=0.6145  f1=0.0000  roc_auc=0.5

> uv run python predict.py --model models\v1 --input data\ejemplo_prediccion.csv --output outputs\predicciones_v1.csv
Columnas ignoradas (no forman parte del esquema del modelo): PassengerId, Name, Ticket, Cabin
Modelo utilizado: models\v1\modelo.joblib
Filas procesadas: 9
Archivo generado: outputs\predicciones_v1.csv
Supervivientes predichos: 5

> uv run python -m pytest
7 passed in 22.43s

> docker build -t modelos-fase2:1.0 .
# imagen construida correctamente
> docker run --rm ... python train.py --data data/train.csv ... --version v-docker --seed 42
RandomForest  accuracy=0.8156  f1=0.7179  roc_auc=0.8356389986824769
> docker run --rm ... python predict.py --model models/v-docker/modelo.joblib ...
Filas procesadas: 9
Supervivientes predichos: 5
```

---

## 12. Identificación de la entrega

Para efectos de auditoría, trazabilidad y evaluación académica, la versión definitiva correspondiente a esta Fase 2 se encuentra debidamente etiquetada en el sistema de control de versiones Git sobre la rama principal `main`:

```powershell
git tag -a entrega-2 -m "Entrega 2 - Fase 2"
git push origin entrega-2
```

---

## 13. Fuentes

* Kaggle — *Titanic: Machine Learning from Disaster* (Conjunto de datos y especificaciones del problema).
* Documentación de [scikit-learn](https://scikit-learn.org/stable/) (`Pipeline`, `ColumnTransformer`, `RandomForestClassifier`, `DummyClassifier`, métricas de evaluación).
* Documentación de [Docker](https://docs.docker.com/) (Contenerización de aplicaciones científicas, gestión de volúmenes y persistencia de datos).
* Especificación técnica de la Fase 2 del Proyecto Integrador (2026-II).
