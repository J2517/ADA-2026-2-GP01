"""
ayudas.py — funciones que el estudiante sí puede leer.

    from ayudas import cargar_modelo, cargar_datos, dividir, ver_prompt, curva, entrega

    modelo = cargar_modelo("qwen17b")
    datos = cargar_datos()
    busqueda, validacion = dividir(datos)

El oráculo (`oraculo.py`) se consulta, no se abre. Acá está lo que sí hace falta
para armarlo: bajar el modelo, leer el JSON visible y partir búsqueda/validación.
"""

import json

from oraculo import armar


MODELOS = {
    "qwen17b": "Qwen/Qwen3-1.7B",
    "qwen17b4bit": "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    # El repo por defecto de Ministral 3 es FP8: la T4 (sm_75) no lo soporta.
    "ministral3b": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    "llama3b": "unsloth/Llama-3.2-3B-Instruct",
    "qwen8b": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "mistral7b": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
}


class Modelo:
    """Red + tokenizador + nombre, para armar el oráculo en una línea.

    El oráculo no importa `transformers`: recibe este envoltorio. `nombre` entra
    en la clave de caché, así que cambiar de checkpoint no reusa respuestas
    de otro modelo.
    """

    def __init__(self, red, tok, nombre):
        self.red, self.tok, self.nombre = red, tok, nombre


def _parchear_compute_dtype(cfg):
    """La T4 es sm_75: sin bf16 nativo. Solo se toca el dtype de cómputo
    para no perder llm_int8_skip_modules del checkpoint dynamic-quant."""
    qc = getattr(cfg, "quantization_config", None)
    if not qc:
        return False
    if isinstance(qc, dict):
        qc["bnb_4bit_compute_dtype"] = "float16"
    else:
        import torch

        qc.bnb_4bit_compute_dtype = torch.float16
    return True


def _resolver_modelo(modelo: str) -> str:
    """Alias de `MODELOS` → id de Hugging Face, o el id si ya trae barra.

    Un id `org/nombre` se deja pasar para cargar checkpoints que no están en
    la tabla sin tocar este archivo. Un nombre suelto que no es alias falla
    en claro, con la lista de alias válidos.
    """
    if modelo in MODELOS:
        return MODELOS[modelo]
    if "/" in modelo:
        return modelo
    alias = ", ".join(MODELOS)
    raise ValueError(
        f"{modelo!r} no es un alias. Use {alias} "
        "o un id público de Hugging Face (org/nombre)."
    )


def _cargar_red(nombre, kwargs):
    """Ministral 3 llega como `Mistral3ForConditionalGeneration` (lleva torre
    de visión) y no está en el mapeo de AutoModelForCausalLM, aunque el oráculo
    solo le mande texto. El ValueError salta antes de bajar los pesos."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    try:
        return AutoModelForCausalLM.from_pretrained(nombre, **kwargs)
    except ValueError:
        return AutoModelForImageTextToText.from_pretrained(nombre, **kwargs)


def cargar_modelo(modelo="qwen17b"):
    """Carga un alias (ver `MODELOS`) o un id público de Hugging Face.

    `token=False`: nunca pide login. Todos los checkpoints de la tabla son
    públicos. Exige GPU: en Colab hay que elegir T4 antes, si no el mensaje
    indica el menú. Devuelve un `Modelo` listo para `Oraculo(...)`.

    Parameters
    ----------
    modelo:
        Alias (`qwen17b`, `llama3b`, …) o id `org/nombre`.
    """
    import torch
    from transformers import AutoConfig, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No hay GPU. En Colab: Entorno de ejecución ▸ Cambiar tipo de "
            "entorno de ejecución ▸ T4 GPU. Luego reinicia y corre desde la celda 1."
        )

    nombre = _resolver_modelo(modelo)
    print(f"cargando {modelo} → {nombre}")
    tok = AutoTokenizer.from_pretrained(nombre, token=False)
    cfg = AutoConfig.from_pretrained(nombre, token=False)
    kwargs = {"device_map": {"": "cuda:0"}, "config": cfg, "token": False}
    if not _parchear_compute_dtype(cfg):
        kwargs["dtype"] = torch.float16
    red = _cargar_red(nombre, kwargs)
    red.eval()
    print(f"{nombre}  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return Modelo(red, tok, nombre)


def _promedio_restricciones(filas):
    """Restricciones por instancia. El verificador es todo-o-nada: una media
    más alta implica instancias más difíciles de acertar de un solo golpe."""
    return sum(len(x["ids"]) for x in filas) / len(filas) if filas else 0.0


def cargar_datos(ruta="datos_visibles.json"):
    """Lee el JSON visible (búsqueda y validación todavía mezcladas).

    No filtra el split: eso lo hace `dividir`. Imprime cuántas instancias hay
    y cuántas restricciones traen en promedio, para calibrar el tamaño de muestra
    antes de gastar generaciones.

    Parameters
    ----------
    ruta:
        Por defecto `datos_visibles.json` en el directorio de trabajo de Colab.
    """
    datos = json.load(open(ruta))
    print(
        f"{len(datos)} instancias, "
        f"{_promedio_restricciones(datos):.1f} restricciones por instancia"
    )
    return datos


def dividir(datos):
    """(busqueda, validacion) — particiones disjuntas y **curadas** del pool.

    El split viene marcado en los datos. Las dos mitades comparten distribución
    (mismo promedio de restricciones por instancia), así que validar mide si la
    config aguanta instancias nuevas y no un lote más difícil.

    El archivo trae además instancias marcadas `descartada`, que ninguna de las
    dos particiones usa. Se sacaron porque rompían la medición, no por difíciles:

    - piden más texto del que caben en `max_new_tokens` (imposibles de acertar),
    - ninguna opción del catálogo puede romperlas (puntos que nadie puede perder),
    - traen una sola restricción (el modelo las acierta siempre y nada las toca),
    - traen cuatro o cinco (el modelo casi nunca las acierta, no informan).

    Sin esa curaduría el puntaje casi no dependía de la configuración elegida,
    que es justo lo que el ejercicio tiene que medir.
    """
    busqueda = [x for x in datos if x["split"] == "busqueda"]
    validacion = [x for x in datos if x["split"] == "validacion"]
    descartadas = sum(1 for x in datos if x["split"] == "descartada")
    print(f"búsqueda: {len(busqueda)} inst, {_promedio_restricciones(busqueda):.1f} restr/inst")
    print(f"validación: {len(validacion)} inst, {_promedio_restricciones(validacion):.1f} restr/inst")
    if descartadas:
        print(f"({descartadas} instancias descartadas: no medían nada — ver `dividir`)")
    return busqueda, validacion


def ver_prompt(config, instancia):
    """El texto exacto que el oráculo le mandaría al modelo.

    No genera tokens: solo sustituye los índices de la config por el texto del
    catálogo y lo pega al prompt de la instancia. Sirve para inspeccionar una
    config sin gastar una consulta.
    """
    return armar(config, instancia)


def curva(historial):
    """historial: lista de precisiones. Dibuja la mejor hasta cada consulta.

    El eje Y no es 'esta config' sino 'el mejor hasta ahora': explorar configs
    peores no baja la curva, y se ve si el presupuesto extra sigue encontrando
    mejoras o ya se estancó.
    """
    import matplotlib.pyplot as plt

    mejor = []
    m = 0.0
    for p in historial:
        m = max(m, p)
        mejor.append(m)
    plt.plot(mejor)
    plt.xlabel("consultas")
    plt.ylabel("mejor precisión")
    plt.show()


def entrega(grupo, config, semana, ruta="entrega.json"):
    """Escribe el JSON de entrega: grupo, config ganadora y semana.

    La nota no sale de `datos_visibles.json`: el profesor corre esta config
    sobre el test privado. Imprime el JSON y lo deja en disco para descargarlo.

    Parameters
    ----------
    grupo:
        Identificador del grupo, p. ej. ``"G07"``.
    config:
        Dict con índices de ranura y ``temperatura``.
    semana:
        Número de semana del curso.
    ruta:
        Archivo de salida. El notebook lo descarga con ``files.download``.
    """
    d = {"grupo": grupo, "config": config, "semana": semana}
    json.dump(d, open(ruta, "w"), indent=1)
    print(json.dumps(d, indent=1))
    return d
