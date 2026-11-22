"""Métricas operativas de la Fase 4 a partir del registro de /predict.

Uso:
    uv run python monitoreo.py
    uv run python monitoreo.py --log logs/registro_predicciones.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = BASE_DIR / "logs" / "registro_predicciones.csv"


def leer_registro(path) -> list[dict]:
    """Lee el CSV; un archivo ausente o vacío se trata como registro vacío."""
    file_path = Path(path)
    if not file_path.is_file() or file_path.stat().st_size == 0:
        return []
    with file_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def calcular_resumen(rows: list[dict]) -> dict:
    """Cuenta solicitudes y fallos (HTTP 400-599)."""
    total = len(rows)
    fallos = 0
    for row in rows:
        try:
            estado = int(row.get("estado_http", ""))
        except (TypeError, ValueError):
            continue
        if 400 <= estado <= 599:
            fallos += 1
    porcentaje = None if total == 0 else fallos / total * 100
    return {"total": total, "fallos": fallos, "porcentaje": porcentaje}


def formatear_porcentaje(porcentaje) -> str:
    """Formato español con coma decimal y un decimal."""
    if porcentaje is None:
        return "sin solicitudes"
    return f"{porcentaje:.1f}%".replace(".", ",")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Métricas operativas de la API.")
    parser.add_argument(
        "--log",
        default=str(DEFAULT_LOG_PATH),
        help="Ruta del CSV de registro de solicitudes.",
    )
    args = parser.parse_args(argv)

    resumen = calcular_resumen(leer_registro(args.log))
    print(f"Total de solicitudes: {resumen['total']}")
    print(f"Porcentaje de fallos: {formatear_porcentaje(resumen['porcentaje'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
