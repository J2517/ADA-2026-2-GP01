"""
Bitácora de la búsqueda. NO decide nada: envuelve al oráculo, anota cada
configuración evaluada (precisión, costo, tiempo, familias violadas) y la
exporta a CSV. Es la evidencia que pide la guía de entrega: tabla de
configuraciones probadas con su precisión y trazas de fallos.

También define el formato común de una configuración (`RANURAS`,
`config_completa`), que usan `busqueda.py` y el notebook.

No importa `torch` ni `oraculo`: recibe el oráculo ya armado.
"""

import csv
import json
import time
from collections import Counter

RANURAS = ["rol", "estrategia", "formato", "estilo", "cierre"]


def config_completa(parcial, temperatura=0.0):
    """Dict con las 5 ranuras en orden fijo + temperatura (formato de entrega)."""
    c = {r: int(parcial[r]) for r in RANURAS if r in parcial}
    c["temperatura"] = float(parcial.get("temperatura", temperatura))
    return c


def _clave(config, semilla):
    return json.dumps(config, sort_keys=True) + f"|{semilla}"


#  Registro de evaluaciones


class Registro:
    """Envuelve al oráculo y anota cada configuración evaluada.

    Por cada consulta guarda: fase, config, precisión, aciertos, fallos,
    cuántas generaciones nuevas costó (las que no estaban en caché),
    segundos y las familias de restricción que fallaron. Una misma config
    con la misma semilla no se anota dos veces.
    """

    def __init__(self, oraculo, instancias, semilla=1, verbose=True):
        self.oraculo = oraculo
        self.instancias = instancias
        self.semilla = semilla
        self.verbose = verbose
        self.filas = []
        self._resultados = {}

    # -- consulta --------------------------------------------------------

    def evaluar(self, config, fase, nota="", semilla=None):
        semilla = self.semilla if semilla is None else semilla
        k = _clave(config, semilla)
        if k in self._resultados:
            return self._resultados[k]

        antes = len(self.oraculo.cache)
        t0 = time.perf_counter()
        r = self.oraculo.evaluar(config, self.instancias, semilla=semilla)
        segundos = time.perf_counter() - t0
        # generaciones = consultas que NO estaban en caché
        generaciones = len(self.oraculo.cache) - antes

        r.fallidas = {t["id"]: t["violo"] for t in r.trazas}
        r.aciertos = {x["id"] for x in self.instancias} - set(r.fallidas)
        self._resultados[k] = r

        fila = {
            "n": len(self.filas) + 1,
            "fase": fase,
            "config": dict(config),
            "semilla": semilla,
            "precision": r.precision,
            "aciertos": len(r.aciertos),
            "fallos": len(r.trazas),
            "consultas": r.n,
            "generaciones": generaciones,
            "segundos": round(segundos, 2),
            "familias": Counter(t["violo"] for t in r.trazas),
            "nota": nota,
        }
        self.filas.append(fila)
        if self.verbose:
            print(
                f"[{fila['n']:3d}] {fase:<12} {_corta(config):<34} "
                f"precisión {r.precision:6.1%}   gen nuevas {generaciones:4d}   "
                f"{segundos:6.1f}s  {nota}"
            )
        return r

    def resultado(self, config, semilla=None):
        return self._resultados[_clave(config, self.semilla if semilla is None else semilla)]

    # -- resúmenes -------------------------------------------------------

    @property
    def historial(self):
        return [f["precision"] for f in self.filas]

    @property
    def consultas(self):
        """Costo lógico: instancias consultadas (lo que cuenta el presupuesto B)."""
        return sum(f["consultas"] for f in self.filas)

    @property
    def generaciones(self):
        return sum(f["generaciones"] for f in self.filas)

    def mejor(self, fases=None):
        filas = [f for f in self.filas if fases is None or f["fase"] in fases]
        f = max(filas, key=lambda f: f["precision"])
        return f["precision"], f["config"]

    def a_csv(self, ruta):
        """Tabla completa de evaluaciones (evidencia para el informe)."""
        campos = ["n", "fase", *RANURAS, "temperatura", "semilla", "precision",
                  "aciertos", "fallos", "consultas", "generaciones", "segundos",
                  "familias_mas_violadas", "nota"]
        with open(ruta, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            for f in self.filas:
                fila = {c: f[c] for c in ("n", "fase", "semilla", "aciertos", "fallos",
                                          "consultas", "generaciones", "segundos", "nota")}
                fila.update({r: f["config"].get(r, "") for r in RANURAS})
                fila["temperatura"] = f["config"].get("temperatura", "")
                fila["precision"] = round(f["precision"], 4)
                fila["familias_mas_violadas"] = "; ".join(
                    f"{fam} x{v}" for fam, v in f["familias"].most_common(3))
                w.writerow(fila)
        return ruta


def _corta(config):
    partes = [f"{r[:3]}={config[r]}" for r in RANURAS if r in config]
    partes.append(f"T={config.get('temperatura', 0.0)}")
    return " ".join(partes) if len(partes) > 1 else "(sin instrucciones) T=" + str(config.get("temperatura", 0.0))
