# Guía Técnica del Proyecto

Documento técnico detallado para la instalación, ejecución y reproducción del análisis
exploratorio del proyecto **Modelo Predictivo de Supervivencia en el Titanic**.

---

## 1. Alcance

Esta guía cubre:

- Estructura del repositorio.
- Requisitos de software y configuración del entorno.
- Instalación paso a paso y gestión de dependencias.
- Formas de ejecutar el notebook de la Fase 1 (EDA).
- Reproducibilidad y restablecimiento del entorno.
- Solución de problemas frecuentes.

---

## 2. Estructura del Repositorio

```
modelo-titanic-g3/
├── .gitignore                # Archivos y directorios excluidos del control de versiones
├── .venv/                    # Entorno virtual (generado automáticamente, no se versiona)
├── data/
│   └── raw/                  # Datos originales sin ninguna transformación
│       ├── train.csv         # Conjunto de entrenamiento (891 registros, con Survived)
│       └── test.csv          # Conjunto de prueba (418 registros, sin Survived)
├── docs/
│   └── TECNICO.md            # Este documento
├── fase-1/
│   └── notebook.ipynb        # Análisis exploratorio de datos (EDA) ejecutado y documentado
├── pyproject.toml            # Metadatos del proyecto y declaración de dependencias
├── README.md                 # Resumen del proyecto y guía rápida de puesta en marcha
└── uv.lock                   # Versiones exactas de todas las dependencias (bloqueo)
```

### Convenciones sobre los datos

| Ubicación | Contenido | Regla |
|-----------|-----------|-------|
| `data/raw/` | Datos tal como se descargan de la fuente | **Nunca se modifican ni se sobrescriben** |
| `data/processed/` (a futuro) | Datos transformados por el pipeline | Generados por scripts; se ignoran en git |

---

## 3. Requisitos de Software

| Requisito | Versión mínima | Notas |
|-----------|---------------|-------|
| Sistema operativo | Cualquiera | Las instrucciones usan sintaxis de terminal estándar |
| Python | 3.12 | La versión la gestiona el propio entorno virtual |
| uv | 0.5.x o superior | Gestor de entornos virtuales y dependencias |

> **¿Por qué un entorno virtual?** Aísla las dependencias del proyecto del Python del sistema,
> evita conflictos entre proyectos y garantiza que todos los integrantes usen las mismas
> versiones. Con `uv` el entorno se crea y se instala automáticamente en un solo paso.

---

## 4. Instalación Paso a Paso

### 4.1 Clonar el repositorio (solo la primera vez)

```bash
git clone <url-del-repositorio> modelo-titanic-g3
cd modelo-titanic-g3
```

### 4.2 Crear el entorno e instalar dependencias

```bash
uv sync
```

Este comando hace dos cosas:

1. Crea el directorio `.venv/` con una versión de Python acorde a `pyproject.toml`.
2. Instala todas las dependencias declaradas como **dependencias regulares** y las del
   grupo **dev** (herramientas para trabajar con notebooks).

### 4.3 Verificar la instalación

```bash
uv run python -c "import pandas, numpy, matplotlib, seaborn; print('OK')"
```

Si todo está correcto, la terminal muestra `OK`.

### 4.4 Dependencias instaladas

Puedes consultarlas en `pyproject.toml` o ejecutando:

```bash
uv pip list
```

Compendio de lo que se instala:

| Paquete | Propósito |
|---------|-----------|
| `pandas` | Lectura, limpieza y análisis de datos tabulares |
| `numpy` | Operaciones numéricas y arrays |
| `matplotlib` | Generación de gráficas base |
| `seaborn` | Gráficas estadísticas de alto nivel (sobre matplotlib) |
| `jupyter` | Ejecución del notebook (Jupyter Lab / Notebook) |
| `ipykernel` | Kernel de Python que usa Jupyter para ejecutar el código |

---

## 5. Ejecución del Notebook

### 5.1 Opción A - Interfaz gráfica (Jupyter Lab)

```bash
uv run jupyter lab
```

1. Se abre el navegador con el panel de Jupyter Lab.
2. Navega hasta `fase-1/` y abre `notebook.ipynb`.
3. Ejecuta todas las celdas en orden: menú **Run ▸ Run All Cells**.
4. Es posible reiniciar el kernel desde **Kernel ▸ Restart Kernel and Run All Cells**.

### 5.2 Opción B - Ejecución desde la terminal

Regenera las salidas del notebook y las guarda en el mismo archivo:

```bash
uv run jupyter nbconvert --to notebook --execute fase-1/notebook.ipynb --inplace
```

Opciones útiles:

| Opción | Efecto |
|--------|--------|
| `--to notebook` | Mantiene el formato `.ipynb` |
| `--execute` | Ejecuta todas las celdas de forma secuencial |
| `--inplace` | Sobrescribe el mismo archivo con los resultados |
| `--ExecutePreprocessor.timeout=120` | Límite de tiempo por celda (por defecto 30 s) |

### 5.3 Nota sobre el kernel

El notebook usa el kernel **Python 3 (ipykernel)**. Al ejecutarlo con `uv run`, Jupyter levanta
el kernel del entorno virtual del proyecto automáticamente, por lo que no hace falta registrarlo
a mano.

---

## 6. Reproducibilidad Y Restablecimiento

### 6.1 Regenerar todos los resultados

Desde la raíz del proyecto:

```bash
uv run jupyter nbconvert --to notebook --execute fase-1/notebook.ipynb --inplace
```

El notebook queda sobrescrito con las tablas y gráficas recién generadas. Si algo sale mal,
revisar la sección 7.

### 6.2 Borrar el entorno y volver a crearlo

```bash
uv sync --reinstall
```

Para eliminar por completo el entorno y regenerarlo limpio:

```bash
rm -rf .venv        # (Windows PowerShell): Remove-Item -Recurse -Force .venv
uv sync
```

### 6.3 Actualizar dependencias

```bash
uv lock --upgrade   # recalcula las versiones en uv.lock
uv sync             # instala las nuevas versiones
```

---

## 7. Solución de Problemas Frecuentes

### 7.1 `uv` no se reconoce como comando

- Instalar uv y verificar que el ejecutable quede en el `PATH` del sistema.
- En Windows, si se instaló a través de otro gestor, comprobar la carpeta `Scripts` del Python.

### 7.2 Error de versión de Python

- El proyecto requiere Python 3.12 o superior; `pyproject.toml` declara `requires-python = ">=3.12"`.
- `uv` descarga la versión adecuada automáticamente si no está disponible en el sistema.

### 7.3 El notebook no encuentra los datos

- El notebook lee desde `../data/raw/` (relativo a `fase-1/`). Ejecutar el notebook **desde el
  proyecto** sin reubicarlo de carpeta. No mover `train.csv` ni `test.csv` fuera de `data/raw/`.

### 7.4 Las gráficas no se muestran

- Ejecutar la celda de imports (el primero de todo) antes que las de gráficas, y ejecutar las celdas en orden.
- Si se usa la terminal, el notebook se abre correctamente si se ejecuta con `uv run jupyter lab`.

### 7.5 Kernel sin el entorno del proyecto

- Asegurarse de iniciar Jupyter con `uv run jupyter lab`; de lo contrario el kernel puede usar
  un Python distinto al del proyecto y fallar los `import`.

---

## 8. Referencias

- Competencia de Kaggle: https://www.kaggle.com/competitions/titanic
- Sistema de gestión de dependencias `uv`: https://docs.astral.sh/uv/
- Documentación de pandas: https://pandas.pydata.org/docs/
- Documentación de seaborn: https://seaborn.pydata.org/