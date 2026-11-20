"""Cliente HTTP de demostración de la API del Titanic (Fase 3).

Uso (con la API ya iniciada):
    python client.py                      # health -> predict -> train -> predict
    python client.py --no-train           # solo health y predict
    python client.py --base-url http://localhost:8000

Usa un registro real de fase-2/data/ejemplo_prediccion.csv (sin la variable
objetivo) y solo realiza solicitudes HTTP: nunca toca los artefactos.
"""

import argparse
import sys
from pathlib import Path

import httpx
import pandas as pd

FEATURE_COLUMNS = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
SAMPLE_CSV = (
    Path(__file__).resolve().parents[1] / "fase-2" / "data" / "ejemplo_prediccion.csv"
)


def load_sample_record(csv_path: Path = SAMPLE_CSV, row: int = 0) -> dict:
    """Primer registro real del CSV de ejemplo; los nulos se envían como null."""
    dataframe = pd.read_csv(csv_path)
    record = {}
    for column in FEATURE_COLUMNS:
        value = dataframe.loc[row, column]
        if pd.isna(value):
            record[column] = None
        else:
            record[column] = value.item() if hasattr(value, "item") else value
    return record


def show(title: str, response: httpx.Response) -> None:
    print(f"\n== {title}")
    print(f"HTTP {response.status_code}")
    print(response.json())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--no-train", action="store_true", help="Omite POST /train.")
    parser.add_argument("--seed", type=int, default=None, help="Semilla para /train.")
    args = parser.parse_args(argv)

    record = load_sample_record()
    print("Registro enviado a /predict:", record)

    try:
        with httpx.Client(base_url=args.base_url, timeout=300.0) as client:
            show("GET /health", client.get("/health"))
            show("POST /predict", client.post("/predict", json=record))

            if not args.no_train:
                body = {"seed": args.seed} if args.seed is not None else None
                response = client.post("/train", json=body)
                show("POST /train", response)
                if response.status_code == 201:
                    new_version = response.json()["version"]
                    show("POST /predict (sin cambiar la configuración)",
                         client.post("/predict", json=record))
                    print(
                        f"\nLa versión {new_version} NO está activa. Para usarla, "
                        f"reinicia la API con MODEL_VERSION={new_version} y vuelve "
                        "a ejecutar este cliente con --no-train."
                    )
    except httpx.ConnectError:
        print(f"No se pudo conectar con {args.base_url}. ¿Está la API iniciada?",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
