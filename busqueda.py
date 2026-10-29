"""
Solución base de la primera entrega. El método tiene cinco fases:

    Fase 0  Línea base: el prompt sin ninguna instrucción.
    Fase 1  Cribado por ranura guiado por trazas: cada opción se prueba sola,
            se mide qué instancias rompe frente a la base y qué familia de
            restricción falló. Con eso se podan las opciones dañinas.
    Fase 2  Ascenso por coordenadas sobre las opciones que sobrevivieron.
    Fase 3  Verificación del vecindario completo: una pasada más con las 8
            opciones por ranura. Si nada mejora, la config es un óptimo local
            (ningún cambio de una sola ranura la mejora). Eso NO es una
            garantía de óptimo global: el problema es NP-difícil.
    Fase 4  Temperatura: se prueba 0.3 y 0.7 con varias semillas.

Aquí solo está el ALGORITMO. Las evaluaciones pasan por un `Registro`
(ver `registro.py`), que lleva la bitácora y el CSV.

No importa `torch` ni `oraculo`: recibe el registro ya armado, así que se
puede probar en cualquier máquina con un oráculo simulado
(ver `tests/test_busqueda.py`).
"""

import random
from collections import Counter

from registro import RANURAS, config_completa


#  Fase 0 y Fase 1 · línea base y cribado guiado por trazas


def cribado(reg, k_por_ranura=8, temperatura=0.0):
    """Prueba cada opción de cada ranura SOLA (las demás ranuras vacías).

    El oráculo permite configs parciales: una ranura ausente no pega texto.
    Comparando contra la base sin instrucciones, cada opción muestra su
    efecto aislado:

      roturas  = instancias que la base acertaba y con esta opción fallan,
                 con la familia que falló (sale de las trazas);
      arreglos = instancias que la base fallaba y con esta opción aciertan.

    Costo: 1 + m·k evaluaciones (41 con m=5, k=8).
    Devuelve (resultado_base, tabla) con una fila por (ranura, opción).
    """
    base_cfg = {"temperatura": temperatura}
    base = reg.evaluar(base_cfg, "0-base", "sin instrucciones")
    tabla = []
    for ranura in RANURAS:
        for o in range(k_por_ranura):
            cfg = {ranura: o, "temperatura": temperatura}
            r = reg.evaluar(cfg, "1-cribado")
            rotas = [i for i in base.aciertos if i in r.fallidas]
            arregladas = [i for i in r.aciertos if i in base.fallidas]
            tabla.append({
                "ranura": ranura,
                "opcion": o,
                "precision": r.precision,
                "delta": r.precision - base.precision,
                "roturas": len(rotas),
                "arreglos": len(arregladas),
                "familias_rotas": Counter(r.fallidas[i] for i in rotas),
                "ids_rotas": rotas,
            })
    return base, tabla


def podar(tabla, conservar=3):
    """Decide qué opciones sobreviven en cada ranura.

    Criterio: se ordenan por precisión (desc) y, en empate, por menos
    roturas; se conservan las `conservar` primeras. Las demás se descartan
    y se deja escrito por qué (sus familias rotas), para el informe.
    Nunca deja una ranura vacía.
    """
    candidatos, descartes = {}, []
    for ranura in RANURAS:
        filas = [f for f in tabla if f["ranura"] == ranura]
        filas.sort(key=lambda f: (-f["precision"], f["roturas"], f["opcion"]))
        candidatos[ranura] = [f["opcion"] for f in filas[:conservar]]
        for f in filas[conservar:]:
            fams = ", ".join(f"{a} x{b}" for a, b in f["familias_rotas"].most_common(3))
            descartes.append({
                "ranura": ranura, "opcion": f["opcion"],
                "precision": f["precision"], "roturas": f["roturas"],
                "motivo": fams or "precisión menor que las conservadas",
            })
    return candidatos, descartes


def inicio_desde_cribado(tabla, temperatura=0.0):
    """Punto de partida del ascenso: la mejor opción de cada ranura sola."""
    c = {}
    for ranura in RANURAS:
        filas = [f for f in tabla if f["ranura"] == ranura]
        c[ranura] = max(filas, key=lambda f: (f["precision"], -f["roturas"], -f["opcion"]))["opcion"]
    return config_completa(c, temperatura)


#  Fase 2 y Fase 3 · ascenso por coordenadas


def ascenso(reg, inicio, candidatos, fase="2-ascenso", max_pasadas=3):
    """Ascenso por coordenadas con la mejor mejora dentro de cada ranura.

    Recorre las ranuras en orden. En cada una prueba los `candidatos[ranura]`
    con las otras 4 fijas y se queda con el mejor solo si mejora
    ESTRICTAMENTE la precisión actual (así nunca cicla). Para cuando una
    pasada completa no mejora nada o se llega a `max_pasadas`.

    Costo por pasada: a lo sumo Σ(|candidatos[s]| - 1) evaluaciones nuevas.
    Devuelve (config, precision, pasadas_hechas, convergio).
    """
    actual = config_completa(inicio)
    p_actual = reg.evaluar(actual, fase, "inicio").precision
    for pasada in range(1, max_pasadas + 1):
        mejoro = False
        for ranura in RANURAS:
            mejor_cfg, mejor_p = actual, p_actual
            for o in candidatos[ranura]:
                if o == actual[ranura]:
                    continue
                cfg = {**actual, ranura: o}
                p = reg.evaluar(cfg, fase, f"pasada {pasada}, {ranura}").precision
                if p > mejor_p:
                    mejor_cfg, mejor_p = cfg, p
            if mejor_p > p_actual:
                if reg.verbose:
                    print(f"      ↑ {ranura}: {actual[ranura]} → {mejor_cfg[ranura]}  "
                          f"({p_actual:.1%} → {mejor_p:.1%})")
                actual, p_actual, mejoro = mejor_cfg, mejor_p, True
        if not mejoro:
            return actual, p_actual, pasada, True
    return actual, p_actual, max_pasadas, False


def todas_las_opciones(k_por_ranura=8):
    return {r: list(range(k_por_ranura)) for r in RANURAS}


#  Fase 4 · temperatura


def probar_temperaturas(reg, config, temperaturas=(0.3, 0.7), semillas=(1, 2, 3)):
    """Con τ > 0 la salida es aleatoria: se mide con varias semillas.

    Regla: se cambia a τ > 0 solo si su precisión PROMEDIO supera
    estrictamente a la de τ = 0 (en empate gana 0.0, que es determinista).
    Devuelve (config_elegida, resumen) con media y desviación por τ.
    """
    base = config_completa(config, 0.0)
    resumen = {0.0: {"precisiones": [reg.evaluar(base, "4-temperatura").precision]}}
    for t in temperaturas:
        cfg = {**base, "temperatura": float(t)}
        ps = [reg.evaluar(cfg, "4-temperatura", f"semilla {s}", semilla=s).precision
              for s in semillas]
        resumen[float(t)] = {"precisiones": ps}
    for t, d in resumen.items():
        ps = d["precisiones"]
        media = sum(ps) / len(ps)
        d["media"] = media
        d["desv"] = (sum((p - media) ** 2 for p in ps) / len(ps)) ** 0.5
    elegida = max(resumen, key=lambda t: (resumen[t]["media"], t == 0.0))
    return {**base, "temperatura": elegida}, resumen


#  Comparación opcional · búsqueda aleatoria con el mismo número de evaluaciones


def busqueda_aleatoria(reg, configs, evaluaciones, semilla_rng=0):
    rng = random.Random(semilla_rng)
    mejor = (-1.0, None)
    for c in rng.sample(configs, evaluaciones):
        p = reg.evaluar(c, "aleatoria").precision
        if p > mejor[0]:
            mejor = (p, c)
    return mejor
