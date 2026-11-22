"""Pruebas del registro de la Fase 4 y del script de monitoreo."""

import csv

import pytest
from fastapi.testclient import TestClient

import api as api_module
from monitoreo import calcular_resumen, formatear_porcentaje, main

VALID = {
    "Pclass": 3,
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Sex": "male",
    "Embarked": "S",
}


@pytest.fixture()
def log_path(tmp_path, monkeypatch):
    path = tmp_path / "registro.csv"
    monkeypatch.setenv("LOG_PATH", str(path))
    return path


def test_registra_una_fila_por_intento_incluido_el_422(log_path):
    client = TestClient(api_module.app)

    assert client.post("/predict", json=VALID).status_code == 200
    invalido = {key: value for key, value in VALID.items() if key != "Age"}
    assert client.post("/predict", json=invalido).status_code == 422

    with log_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["estado_http"] == "200"
    assert rows[0]["version"] == "v1"
    assert rows[0]["prediccion"] in {"0", "1"}
    assert rows[0]["fecha_hora"]

    assert rows[1]["estado_http"] == "422"
    assert rows[1]["version"] == ""
    assert rows[1]["prediccion"] == ""


def test_el_historial_se_agrega_sin_repetir_encabezado(log_path):
    client = TestClient(api_module.app)
    client.post("/predict", json=VALID)
    client.post("/predict", json=VALID)

    lines = log_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 3  # encabezado + dos solicitudes
    assert lines[0] == "fecha_hora,estado_http,version,prediccion"


def test_resumen_cuenta_fallos_4xx_y_5xx():
    rows = [
        {"estado_http": "200"},
        {"estado_http": "422"},
        {"estado_http": "503"},
        {"estado_http": "201"},
    ]

    resumen = calcular_resumen(rows)

    assert resumen == {"total": 4, "fallos": 2, "porcentaje": 50.0}


def test_monitoreo_vacio_muestra_sin_solicitudes(tmp_path, capsys):
    path = tmp_path / "vacio.csv"

    assert main(["--log", str(path)]) == 0

    output = capsys.readouterr().out
    assert "Total de solicitudes: 0" in output
    assert "Porcentaje de fallos: sin solicitudes" in output


def test_monitoreo_del_ensayo_2_validas_y_1_invalida(tmp_path, capsys):
    path = tmp_path / "ensayo.csv"
    path.write_text(
        "fecha_hora,estado_http,version,prediccion\n"
        "2026-11-22T10:00:00+00:00,200,v1,0\n"
        "2026-11-22T10:01:00+00:00,200,v1,1\n"
        "2026-11-22T10:02:00+00:00,422,,\n",
        encoding="utf-8",
    )

    assert main(["--log", str(path)]) == 0

    output = capsys.readouterr().out
    assert "Total de solicitudes: 3" in output
    assert "Porcentaje de fallos: 33,3%" in output
    assert formatear_porcentaje(None) == "sin solicitudes"
