"""
Pruebas de `evaluacion.py` (no necesitan GPU ni modelo).

Correr:  python -m pytest -q tests/
"""

import json
import os
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import evaluacion as E  # noqa: E402

CONFIG_OK = {"rol": 5, "estrategia": 2, "formato": 0, "estilo": 6, "cierre": 3, "temperatura": 0.0}


def _escribir(tmp_path, nombre, contenido):
    p = tmp_path / nombre
    p.write_text(contenido, encoding="utf-8")
    return str(p)


# ── entrega.json ────────────────────────────────────────────────────


def test_cargar_entrega_valida(tmp_path):
    ruta = _escribir(tmp_path, "entrega.json",
                     json.dumps({"grupo": "G01", "config": CONFIG_OK, "semana": 4}))
    grupo, semana, config = E.cargar_entrega(ruta)
    assert (grupo, semana) == ("G01", 4)
    assert list(config) == [*E.RANURAS, "temperatura"]


@pytest.mark.parametrize("cambio", [
    {"rol": 8}, {"rol": -1}, {"rol": "5"}, {"rol": True}, {"temperatura": 0.5},
])
def test_config_invalida(cambio):
    with pytest.raises(ValueError):
        E.validar_config({**CONFIG_OK, **cambio})


def test_config_sin_ranura():
    c = dict(CONFIG_OK)
    del c["cierre"]
    with pytest.raises(ValueError):
        E.validar_config(c)


def test_entrega_del_repositorio_es_valida():
    """Si el repo ya trae entrega.json, tiene que pasar la validación."""
    ruta = os.path.join(RAIZ, "entrega.json")
    if not os.path.exists(ruta):
        pytest.skip("todavía no hay entrega.json en el repositorio")
    E.cargar_entrega(ruta)


# ── datos ───────────────────────────────────────────────────────────


def test_formato_del_curso_filtra_descartadas(tmp_path):
    datos = [
        {"id": 1, "prompt": "a", "ids": ["x:y"], "kwargs": [{}], "split": "validacion"},
        {"id": 2, "prompt": "b", "ids": ["x:y"], "kwargs": [{}], "split": "descartada"},
    ]
    inst, aviso = E.leer_instancias(_escribir(tmp_path, "d.json", json.dumps(datos)))
    assert [x["id"] for x in inst] == [1]
    assert "descartada" in aviso


def test_formato_ifbench_jsonl_quita_none(tmp_path):
    lineas = [
        {"key": "7", "prompt": "p", "instruction_id_list": ["count:a"],
         "kwargs": [{"N": 3, "word": None}]},
        {"key": "8", "prompt": "q", "instruction_id_list": ["count:b", "count:c"],
         "kwargs": [{"x": None}, {}]},
    ]
    ruta = _escribir(tmp_path, "d.jsonl", "\n".join(json.dumps(l) for l in lineas))
    inst, _ = E.leer_instancias(ruta)
    assert inst[0]["id"] == "7" and inst[0]["ids"] == ["count:a"]
    assert inst[0]["kwargs"] == [{"N": 3}]
    assert inst[1]["kwargs"] == [{}, {}]
    assert inst[0]["restriccion"] == ""


def test_ids_repetidos_se_renombran(tmp_path):
    datos = [{"id": 1, "prompt": "a", "ids": ["x:y"], "kwargs": [{}]}] * 2
    inst, aviso = E.leer_instancias(_escribir(tmp_path, "d.json", json.dumps(datos)))
    assert len({x["id"] for x in inst}) == 2 and "repetidos" in aviso


def test_split_test_tiene_prioridad(tmp_path):
    datos = [
        {"id": 1, "prompt": "a", "ids": ["x:y"], "kwargs": [{}], "split": "busqueda"},
        {"id": 2, "prompt": "b", "ids": ["x:y"], "kwargs": [{}], "split": "test"},
    ]
    inst, _ = E.leer_instancias(_escribir(tmp_path, "d.json", json.dumps(datos)))
    assert [x["id"] for x in inst] == [2]


def test_datos_visibles_del_repo_se_leen():
    inst, _ = E.leer_instancias(os.path.join(RAIZ, "datos_visibles.json"))
    assert len(inst) == 187  # 100 búsqueda + 87 validación


# ── verificadores ───────────────────────────────────────────────────


class _Termina:
    """Verificador falso: la respuesta debe terminar con `fin`."""

    def __init__(self, iid):
        self.fin = "."

    def build_description(self, fin=".", prompt=None):
        self.fin = fin

    def get_instruction_args(self):
        return {"fin": self.fin}

    def check_following(self, texto):
        return texto[-1] == self.fin  # IndexError con texto vacío, como algunos reales


REG = {"fin:termina": _Termina}


def test_revisar_verificadores_detecta_familia_desconocida():
    inst = [
        {"id": 1, "prompt": "p", "ids": ["fin:termina"], "kwargs": [{"fin": "!"}]},
        {"id": 2, "prompt": "p", "ids": ["nueva:familia"], "kwargs": [{}]},
        {"id": 3, "prompt": "p", "ids": ["fin:termina"], "kwargs": [{"parametro_raro": 1}]},
    ]
    validas, problemas = E.revisar_verificadores(inst, [REG])
    assert [x["id"] for x in validas] == [1]
    assert {p[0] for p in problemas} == {2, 3}


def test_verificar_todo_o_nada_y_robusto():
    x = {"id": 1, "prompt": "p", "ids": ["fin:termina", "fin:termina"],
         "kwargs": [{"fin": "!"}, {"fin": "!"}]}
    clase_de = REG.__getitem__
    assert E.verificar(x, "hola!", clase_de) == (1, None)
    assert E.verificar(x, "hola.", clase_de) == (0, "fin:termina")
    assert E.verificar(x, "", clase_de) == (0, "fin:termina")      # vacía: no revienta
    assert E.verificar(x, "   ", clase_de) == (0, "fin:termina")


def test_resumen():
    class R:
        precision, n = 0.5, 2
        trazas = [{"id": 2, "violo": "fin:termina", "salida": "hola."}]
    inst = [{"id": 1, "ids": ["fin:termina"]}, {"id": 2, "ids": ["fin:termina"]}]
    s = E.resumen(R, inst)
    assert s["aciertos"] == 1 and s["familias_que_mas_fallan"] == [("fin:termina", 1)]
    assert [f["acierto"] for f in s["por_instancia"]] == [1, 0]
