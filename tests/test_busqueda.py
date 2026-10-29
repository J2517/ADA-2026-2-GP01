"""
Pruebas de `busqueda.py` y `registro.py` con un oráculo SIMULADO (no necesita GPU ni modelo).

El simulador sigue el modelo de compatibilidad del informe: cada instancia
admite ciertas opciones por ranura y se acierta solo si TODAS las ranuras
elegidas son admisibles (todo-o-nada). Una ranura ausente no estorba, salvo
en `cierre`: algunas instancias solo se aciertan con el recordatorio de
restricciones (como pasa con el oráculo real).

Correr:  python -m pytest -q tests/
"""

import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import busqueda as B  # noqa: E402
import registro as R  # noqa: E402

K = 8


class _Resultado:
    def __init__(self, precision, trazas, n):
        self.precision, self.trazas, self.n = precision, trazas, n


class OraculoSimulado:
    """Misma interfaz que `oraculo.Oraculo`: evaluar(config, instancias, semilla), cache."""

    def __init__(self, n=60, semilla=7, prob_admisible=0.75):
        rng = random.Random(semilla)
        self.limpia = {r: rng.randrange(K) for r in B.RANURAS}
        self.instancias = []
        for i in range(n):
            adm = {}
            for r in B.RANURAS:
                adm[r] = {o for o in range(K) if o == self.limpia[r] or rng.random() < prob_admisible}
            self.instancias.append({"id": i, "adm": adm, "pide_cierre": rng.random() < 0.5})
        self.cache = {}

    def exito(self, config, x):
        for r in B.RANURAS:
            if r in config and config[r] not in x["adm"][r]:
                return 0, f"familia_{r}"
        if x["pide_cierre"] and config.get("cierre") != self.limpia["cierre"]:
            return 0, "familia_recordatorio"
        return 1, None

    def evaluar(self, config, instancias, semilla=1):
        trazas, ok = [], 0
        for x in instancias:
            clave = (R._clave(config, semilla), x["id"])
            if clave not in self.cache:
                self.cache[clave] = self.exito(config, x)
            r, violo = self.cache[clave]
            ok += r
            if not r:
                trazas.append({"id": x["id"], "violo": violo, "salida": "(simulada)"})
        return _Resultado(ok / len(instancias), trazas, len(instancias))


def _precision_directa(o, cfg):
    return sum(o.exito(cfg, x)[0] for x in o.instancias) / len(o.instancias)


def _pipeline(o):
    reg = R.Registro(o, o.instancias, verbose=False)
    base, tabla = B.cribado(reg, K)
    cand, desc = B.podar(tabla, conservar=3)
    ini = B.inicio_desde_cribado(tabla)
    cfg, p, _, _ = B.ascenso(reg, ini, cand)
    cfg, p, _, conv = B.ascenso(reg, cfg, B.todas_las_opciones(K), fase="3-verificacion")
    return reg, tabla, cand, desc, cfg, p, conv


def test_registro_cuenta_generaciones_y_no_duplica():
    o = OraculoSimulado(n=20)
    reg = R.Registro(o, o.instancias, verbose=False)
    cfg = {"rol": 1, "temperatura": 0.0}
    reg.evaluar(cfg, "x")
    assert reg.filas[-1]["generaciones"] == 20
    reg.evaluar(cfg, "x")  # repetida: no genera ni se anota
    assert len(reg.filas) == 1
    reg.evaluar(cfg, "x", semilla=2)  # otra semilla: es otra consulta
    assert len(reg.filas) == 2 and reg.generaciones == 40


def test_cribado_cuesta_1_mas_m_por_k():
    o = OraculoSimulado(n=10)
    reg = R.Registro(o, o.instancias, verbose=False)
    _, tabla = B.cribado(reg, K)
    assert len(tabla) == 5 * K
    assert len(reg.filas) == 1 + 5 * K


def test_poda_nunca_vacia_una_ranura():
    o = OraculoSimulado(n=30)
    reg = R.Registro(o, o.instancias, verbose=False)
    _, tabla = B.cribado(reg, K)
    cand, desc = B.podar(tabla, conservar=3)
    assert all(len(cand[r]) == 3 for r in B.RANURAS)
    assert len(desc) == 5 * (K - 3)


def test_resultado_es_optimo_local_1opt():
    """Ninguna config que cambie UNA sola ranura supera a la elegida."""
    o = OraculoSimulado(n=60, semilla=3)
    *_, cfg, p, conv = _pipeline(o)
    assert conv
    for r in B.RANURAS:
        for v in range(K):
            assert _precision_directa(o, {**cfg, r: v}) <= p + 1e-12


def test_encuentra_el_optimo_global_en_modelo_separable():
    """En este simulador las ranuras no interactúan: el óptimo local es global.
    Se comprueba contra fuerza bruta sobre las 8^5 = 32768 configuraciones."""
    o = OraculoSimulado(n=40, semilla=11)
    *_, cfg, p, _ = _pipeline(o)
    optimo = max(
        _precision_directa(o, dict(zip(B.RANURAS, combo), temperatura=0.0))
        for combo in itertools.product(range(K), repeat=5)
    )
    assert abs(p - optimo) < 1e-12


def test_config_final_tiene_formato_de_entrega():
    o = OraculoSimulado(n=30)
    *_, cfg, _, _ = _pipeline(o)
    assert list(cfg) == [*B.RANURAS, "temperatura"]
    assert all(isinstance(cfg[r], int) and 0 <= cfg[r] < K for r in B.RANURAS)
    assert cfg["temperatura"] in (0.0, 0.3, 0.7)


def test_csv_se_escribe(tmp_path):
    o = OraculoSimulado(n=10)
    reg = R.Registro(o, o.instancias, verbose=False)
    B.cribado(reg, K)
    ruta = reg.a_csv(tmp_path / "registro.csv")
    lineas = open(ruta, encoding="utf-8").read().strip().splitlines()
    assert len(lineas) == 1 + 1 + 5 * K
