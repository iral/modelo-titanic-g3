# Modelo Predictivo de Supervivencia en el Titanic

## 1. Integrantes del Equipo
* **Santiago Acevedo González**
* **[Nombre Integrante 2]**
* **Ivan Rene Acosta Lallemand**

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
│   └── notebook.ipynb    # Análisis exploratorio de datos (EDA)
├── pyproject.toml        # Definición del entorno y dependencias
├── uv.lock               # Bloqueo de versiones de dependencias
└── README.md
```

## 6. Puesta en Marcha

### 6.1 Requisitos
* Python 3.12 o superior.
* [uv](https://docs.astral.sh/uv/) para la gestión del entorno virtual y las dependencias.

### 6.2 Instalación
```bash
uv sync
```
Este comando crea el entorno virtual (`.venv/`), instala las dependencias del proyecto
(análisis de datos y visualización) y las herramientas para ejecutar el notebook.

> **Nota:** si trabajas en otro equipo y acabas de clonar el repositorio, este es el único
> paso necesario; no hace falta crear el entorno a mano.

### 6.3 Ejecutar el notebook
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

### 6.4 Detalle técnico
Para una guía exhaustiva (requisitos, esquema del dataset, contenido celda por celda del
notebook, resultados del análisis y resolución de problemas), ver **docs/TECNICO.md**.
