# Proyecto Oráculo

Curso de Análisis y Diseño de Algoritmos. El trabajo es encontrar una buena configuración de prompt consultando un oráculo. Busquen y validen cuantas veces quieran: el caché guarda lo ya generado y las trazas de fallos vienen con cada resultado.

| Notebook | Colab |
|---|---|
| Proyecto | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanielMelo404/Proyecto-Oraculo-AyD/blob/main/proyecto_oraculo.ipynb) |

## Cómo se juega

1. Abran el notebook del curso en Colab.
2. Corran las celdas 1–3. En la 1, **reinicien el entorno** cuando lo pida.
3. Escriban su heurística **solo en la celda 4**.
4. Exporten `entrega.json` en la celda 5.

```
r = oraculo.evaluar(config, instancias, semilla)
r_val = oraculo.validar(config, n)

r.precision        # 0.55
r.trazas           # [{id, violo, salida}, ...]
```

Hay **32 768** configuraciones (`espacio()`, temperatura 0.0): un índice de 0 a 7 por ranura entre `rol`, `estrategia`, `formato`, `estilo` y `cierre`, más la temperatura. Cada ranura es un consejo de prompting; algunos combinan mal con las restricciones que mide el verificador, y las trazas dicen cuál falló. El lote de búsqueda del notebook son las 100 instancias de `busqueda`. Medir con cuántas instancias buscar es parte del problema.

**Es un problema de optimización de verdad.** Cada ranura ataca una parte distinta de la respuesta —el prefijo, el largo, la estructura, los caracteres, el sufijo— y los efectos se componen. Una configuración al azar se queda **debajo del 5%**; el techo está cerca del **24%**. Sortear no alcanza: hay **una sola opción limpia entre ocho** en cada ranura, así que acertar cuatro de cinco ranuras al azar pasa 1 vez cada 900. Para pasar de ahí hay que usar la estructura — leer las trazas, ver qué familia rompe cada opción, y buscar con eso (backtracking con poda, ascenso por coordenadas, haz, recocido). Reparar una ranura a la vez sube de a escalones; ninguna opción buena está en el mismo índice en todas las ranuras.

Las particiones vienen **curadas**. `datos_visibles.json` marca como `descartada` a la instancia que pide más texto del que cabe en `max_new_tokens`, a la que ninguna opción puede romper, y a las de una o de cuatro-cinco restricciones. Las primeras son imposibles de acertar; las segundas son puntos que nadie puede perder, y con un tercio del lote así el puntaje casi no dependía de la configuración elegida. Búsqueda y validación quedan con la misma dificultad (2.5 restricciones por instancia), para que validar mida generalización y no un lote más duro.

## Modelos

`cargar_modelo` acepta un alias o un id de Hugging Face:

| Alias | Checkpoint | Notas |
|---|---|---|
| `qwen17b` | `Qwen/Qwen3-1.7B` | el de por defecto; el más rápido |
| `qwen17b4bit` | `unsloth/Qwen3-1.7B-unsloth-bnb-4bit` | el mismo 1.7B en 4-bit |
| `ministral3b` | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | 3.8B en fp16 (~7.7 GB), T4 |
| `llama3b` | `unsloth/Llama-3.2-3B-Instruct` | 3.2B en fp16 (~6.4 GB), T4 |
| `qwen8b` | `unsloth/Qwen3-8B-unsloth-bnb-4bit` | 4-bit, T4 |
| `mistral7b` | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` | 4-bit, T4 |

```
modelo = cargar_modelo("qwen8b")
```

Los 7-8B son lentos en T4: tengan paciencia con las primeras corridas.

## Búsqueda y validación

`datos_visibles.json` tiene 450 instancias, de las que se usan **100 de búsqueda y 87 de validación** (el resto va marcado `descartada`, ver `dividir`). Son el mismo split que usan [GEPA](https://arxiv.org/abs/2507.19457) y NPO [1] sobre IF-RLVR Train, de donde salen estos datos. No son las filas exactas de los papers —no publican los índices—, sí la fuente, los tamaños y la disyunción.

```
datos = cargar_datos()
busqueda, validacion = dividir(datos)
oraculo = Oraculo(modelo, busqueda, validacion)
```

Las dos particiones salen del mismo pool y comparten distribución: validar mide si la config aguanta **instancias** nuevas, no restricciones de un tipo nuevo. Ese otro salto se mide al calificar, con familias que no están en este archivo.

Busquen solo sobre `busqueda`. `oraculo.validar` mide la config elegida con una muestra fija de `n` instancias (las 300 tardan) e imprime qué restricciones se cayeron más.

```
r_val = oraculo.validar(mejor[1], n=30)
```

### Ojo con el todo-o-nada

Una instancia trae **hasta 5 restricciones** (2.3 en promedio) y se puntúa todo-o-nada: basta que falle una para que valga 0. En las instancias más duras el modelo acierta poco, así que **con pocas instancias muchas configuraciones van a marcar `0.0` y parecer iguales**. Si su búsqueda no logra distinguir nada, suban el número de instancias antes de cambiar de heurística — decidir cuántas medir es parte del problema.

## Archivos

| Archivo | Qué es |
|---|---|
| `proyecto_oraculo.ipynb` | El notebook del curso |
| `oraculo.py` | La caja negra. Se consulta, no se abre |
| `ayudas.py` | Cargar modelo, datos, dividir, ver un prompt, curva, entrega |
| `datos_visibles.json` | 450 instancias: 100 de búsqueda, 87 de validación, 263 descartadas |

## Entrega

Un `entrega.json` con grupo, configuración y semana:

```json
{
 "grupo": "G07",
 "config": {"rol": 2, "estrategia": 7, "formato": 4, "estilo": 1, "cierre": 6, "temperatura": 0.0},
 "semana": 3
}
```

## Referencias

[1] Yuan Chang y Xiaoqi Chen, *Naive Prompt Optimization: Rethinking the Need for Complex Prompt Search*, arXiv:2608.27266, 2026. https://arxiv.org/abs/2608.27266
