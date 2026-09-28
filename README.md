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
│   ├── modelo.joblib           # Pipeline completo entrenado (preprocesamiento + Random Forest)
│   ├── metricas_base.json      # Métricas del modelo base en el conjunto de prueba
│   └── metricas_modelo.json    # Métricas del modelo predictivo en el conjunto de prueba
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
`Verificación exitosa: el pipeline guardado se cargó y predijo correctamente.`

**Reproducibilidad:** todas las decisiones aleatorias del notebook (división de datos, validación
cruzada, bosque aleatorio y muestra de verificación) usan la constante `SEED = 42`, fijada en la
celda de configuración y propagada como `random_state` a cada estimador. Con el dataset original sin
modificar, dos ejecuciones completas producen las mismas métricas publicadas en
[`fase-1/README.md`](fase-1/README.md).

## 5. Documentación

| Documento | Contenido |
|-----------|-----------|
| [`fase-1/README.md`](fase-1/README.md) | Problema, datos, preprocesamiento, modelo base, algoritmo definitivo, resultados finales y modelo almacenado. |
| [`docs/TECNICO.md`](docs/TECNICO.md) | Guía exhaustiva: requisitos, esquema del dataset, ejecución, reproducibilidad y resolución de problemas. |
| `fase-1/notebook.ipynb` | Notebook completo y ejecutado, con la reflexión técnica final. |
