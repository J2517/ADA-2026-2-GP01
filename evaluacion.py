"""
evaluacion.py — Evaluación de la configuración entregada · Grupo G01
Proyecto Oráculo · Análisis y Diseño de Algoritmos · Universidad de Caldas · 2026-2

Funciones que usa `evaluacion_G01.ipynb` para que la profesora evalúe
`entrega.json` sobre sus datos privados sin editar nada:

    cargar_entrega      lee y valida entrega.json
    leer_instancias     lee JSON o JSONL y normaliza los dos formatos conocidos
    revisar_verificadores  detecta, antes de gastar GPU, familias sin verificador
    resumen             precisión, familias que más fallan y tabla por instancia

No importa `torch` ni `oraculo`: se puede probar sin GPU (tests/test_evaluacion.py).
"""

import json
from collections import Counter

RANURAS = ["rol", "estrategia", "formato", "estilo", "cierre"]
TEMPERATURAS = (0.0, 0.3, 0.7)
OPCIONES_POR_RANURA = 8


# ─────────────────────────────────────────────────────────────────────
#  entrega.json
# ─────────────────────────────────────────────────────────────────────


def validar_config(config, k=OPCIONES_POR_RANURA):
    """Lanza ValueError con un mensaje claro si la config no es válida."""
    faltan = [r for r in RANURAS + ["temperatura"] if r not in config]
    if faltan:
        raise ValueError(f"a la config le faltan las llaves {faltan}")
    for r in RANURAS:
        v = config[r]
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < k:
            raise ValueError(f"{r}={v!r}: debe ser un entero entre 0 y {k - 1}")
    if float(config["temperatura"]) not in TEMPERATURAS:
        raise ValueError(f"temperatura={config['temperatura']!r}: debe ser una de {TEMPERATURAS}")
    return True


def cargar_entrega(ruta="entrega.json"):
    """Lee entrega.json y devuelve (grupo, semana, config) ya validados.

    La config se devuelve con las ranuras en orden fijo y la temperatura
    como float, que es lo que espera el oráculo.
    """
    with open(ruta, encoding="utf-8") as fh:
        d = json.load(fh)
    for llave in ("grupo", "config", "semana"):
        if llave not in d:
            raise ValueError(f"entrega.json no trae la llave {llave!r}")
    validar_config(d["config"])
    config = {r: int(d["config"][r]) for r in RANURAS}
    config["temperatura"] = float(d["config"]["temperatura"])
    return d["grupo"], d["semana"], config


# ─────────────────────────────────────────────────────────────────────
#  Datos de prueba
# ─────────────────────────────────────────────────────────────────────


def _leer_crudo(ruta):
    with open(ruta, encoding="utf-8") as fh:
        texto = fh.read().strip()
    if not texto:
        return []
    if texto[0] == "[":
        return json.loads(texto)
    if texto[0] == "{":
        try:  # un solo objeto JSON (p. ej. {"instancias": [...]})
            d = json.loads(texto)
            for llave in ("instancias", "datos", "data", "test"):
                if isinstance(d.get(llave), list):
                    return d[llave]
            return [d]
        except json.JSONDecodeError:
            pass  # JSONL: un objeto por línea
    return [json.loads(l) for l in texto.splitlines() if l.strip()]


def normalizar(x, posicion):
    """Lleva una instancia al formato que usa el oráculo.

    Formatos aceptados:
      - el del curso (datos_visibles.json): prompt, ids, kwargs, id, restriccion
      - el de IFEval / IFBench: prompt, instruction_id_list, kwargs, key

    En los kwargs se quitan los valores None (así los guarda IFBench y así
    los filtra su propio evaluador antes de construir el verificador).
    """
    if "prompt" not in x:
        raise ValueError(f"la instancia #{posicion} no trae 'prompt'")
    ids = x.get("ids", x.get("instruction_id_list"))
    if ids is None:
        raise ValueError(f"la instancia #{posicion} no trae 'ids' ni 'instruction_id_list'")
    kwargs = x.get("kwargs") or [{} for _ in ids]
    if len(kwargs) != len(ids):
        raise ValueError(f"la instancia #{posicion}: {len(ids)} restricciones y {len(kwargs)} kwargs")
    iid = x.get("id", x.get("key", posicion))
    return {
        **x,
        "id": iid,
        "prompt": x["prompt"],
        "ids": list(ids),
        "kwargs": [{k: v for k, v in (kw or {}).items() if v is not None} for kw in kwargs],
        "restriccion": x.get("restriccion", ""),
    }


def leer_instancias(ruta):
    """Lee el archivo de prueba (JSON o JSONL) y devuelve (instancias, aviso).

    Si el archivo trae la marca `split` del curso, se descartan las
    `descartada` (la misma curaduría que `ayudas.dividir`). Si hay
    instancias marcadas `test`/`prueba`, se usan solo esas.
    """
    crudo = _leer_crudo(ruta)
    inst = [normalizar(x, i) for i, x in enumerate(crudo)]
    aviso = ""
    splits = Counter(x.get("split") for x in inst if "split" in x)
    if splits:
        marcadas = [x for x in inst if x.get("split") in ("test", "prueba")]
        if marcadas:
            inst, aviso = marcadas, f"se usan las {len(marcadas)} instancias marcadas test/prueba"
        else:
            antes = len(inst)
            inst = [x for x in inst if x.get("split") != "descartada"]
            if len(inst) < antes:
                aviso = f"se omitieron {antes - len(inst)} instancias marcadas 'descartada'"
    vistos = Counter(x["id"] for x in inst)
    repetidos = [k for k, v in vistos.items() if v > 1]
    if repetidos:  # el caché del oráculo usa el id: no puede repetirse
        for i, x in enumerate(inst):
            x["id"] = f"{x['id']}#{i}"
        aviso += (" · " if aviso else "") + f"{len(repetidos)} ids repetidos: se renombraron"
    return inst, aviso


# ─────────────────────────────────────────────────────────────────────
#  Verificadores
# ─────────────────────────────────────────────────────────────────────


def revisar_verificadores(instancias, registros):
    """Comprueba, SIN generar texto, que cada restricción tiene verificador
    y que sus kwargs son aceptados.

    `registros` es la misma lista de diccionarios {familia: clase} que usa el
    oráculo, en el mismo orden. Devuelve (validas, problemas): las instancias
    que se pueden calificar y una lista de (id, familia, motivo) de las que no.
    """
    validas, problemas = [], []
    for x in instancias:
        motivo = None
        for iid, kw in zip(x["ids"], x["kwargs"]):
            clase = next((reg[iid] for reg in registros if iid in reg), None)
            if clase is None:
                motivo = (iid, "ningún registro de verificadores conoce esta familia")
                break
            try:
                c = clase(iid)
                c.build_description(**kw)
                args = c.get_instruction_args()
                if args and "prompt" in args:
                    c.build_description(prompt=x["prompt"])
            except Exception as e:  # noqa: BLE001 — se reporta tal cual
                motivo = (iid, f"{type(e).__name__}: {e}")
                break
        if motivo:
            problemas.append((x["id"], *motivo))
        else:
            validas.append(x)
    return validas, problemas


def verificar(instancia, respuesta, clase_de):
    """Misma regla que `Oraculo._verificar` (todo-o-nada, `violo` = primera
    familia que falla), con dos protecciones del evaluador oficial de IFBench:

      - una respuesta vacía falla sin llamar al verificador (algunos, como
        `last_word:last_word_answer` o `sentence:increment`, lanzan
        IndexError con texto vacío);
      - si un verificador lanza una excepción, esa restricción cuenta como
        no cumplida en vez de detener toda la evaluación.
    """
    ids = instancia["ids"]
    if not respuesta or not respuesta.strip():
        return 0, ids[0] if ids else None
    for iid, kw in zip(ids, instancia["kwargs"]):
        try:
            c = clase_de(iid)(iid)
            c.build_description(**kw)
            args = c.get_instruction_args()
            if args and "prompt" in args:
                c.build_description(prompt=instancia["prompt"])
            if not c.check_following(respuesta):
                return 0, iid
        except Exception:  # noqa: BLE001
            return 0, iid
    return 1, None


def blindar(oraculo):
    """Reemplaza el verificador del oráculo por `verificar` (ver arriba).
    No cambia el modelo, el prompt ni la generación: solo el manejo de errores."""
    oraculo._verificar = lambda instancia, respuesta: verificar(instancia, respuesta, oraculo._clase)
    return oraculo


# ─────────────────────────────────────────────────────────────────────
#  Resumen del resultado
# ─────────────────────────────────────────────────────────────────────


def resumen(resultado, instancias, top=10):
    """Dict con precisión, conteos, familias que más fallan y fila por instancia."""
    fallidas = {t["id"]: t for t in resultado.trazas}
    filas = []
    for x in instancias:
        t = fallidas.get(x["id"])
        filas.append({
            "id": x["id"],
            "restricciones": "; ".join(x["ids"]),
            "acierto": 0 if t else 1,
            "violo": t["violo"] if t else "",
            "salida_inicio": (t["salida"][:200] if t else ""),
        })
    return {
        "precision": resultado.precision,
        "aciertos": resultado.n - len(resultado.trazas),
        "n": resultado.n,
        "familias_que_mas_fallan": Counter(t["violo"] for t in resultado.trazas).most_common(top),
        "por_instancia": filas,
    }
