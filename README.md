# Proyecto Oráculo · Grupo G01

**Análisis y Diseño de Algoritmos (ADAG) · Ingeniería de Sistemas · Universidad de Caldas · 2026-2**

| | |
|---|---|
| Integrante | Jackeline Rivera Aguilera |
| Grupo / subproyecto | G01 · Proyecto Oráculo (búsqueda de la configuración de prompt) |
| Entrega | Primera entrega (semanas 1–4): formalización, NP-dureza y solución base |
| Versión entregada | rama `main` |
| Repositorio base | Fork de [DanielMelo404/Proyecto-Oraculo-AyD](https://github.com/DanielMelo404/Proyecto-Oraculo-AyD). Sus instrucciones originales están en [`docs/README_original_curso.md`](docs/README_original_curso.md) |

## El problema en una línea

Elegir un índice (0–7) para cada una de las cinco ranuras del prompt (`rol`, `estrategia`, `formato`, `estilo`, `cierre`) y una temperatura, de modo que el modelo Qwen3-1.7B cumpla el mayor número posible de restricciones verificables. Son 8⁵ = 32 768 configuraciones y cada evaluación cuesta unos 3 minutos de GPU. La solución base encuentra una buena configuración sin probarlas todas.

## Resultado

| | |
|---|---|
| Configuración ([`entrega.json`](entrega.json)) | `rol 6 · estrategia 6 · formato 0 · estilo 7 · cierre 3 · temperatura 0.0` |
| Precisión en búsqueda (100 instancias) | **21 %**. Sin instrucciones: 16 % |
| Precisión en validación (`oraculo.validar`, 87 instancias) | **18,4 %** |
| Garantía | Óptimo local 1-opt: ninguna configuración que cambie una sola ranura la supera (35 vecinos evaluados). **No** es un óptimo global demostrado |
| Costo | 103 evaluaciones = 10 300 consultas, frente a 3 276 800 de la fuerza bruta (0,31 %) |

La nota no sale de estos números: la profesora evalúa `entrega.json` sobre un conjunto de prueba privado con [`evaluacion_G01.ipynb`](evaluacion_G01.ipynb).

## Notebooks

| Notebook | Para qué | Colab |
|---|---|---|
| [`busqueda_G01_resultado.ipynb`](busqueda_G01_resultado.ipynb) | **Evidencia de la búsqueda**: ejecución completa con salidas, trazas, interpretación, curva y validación | Solo lectura |
| [`busqueda_G01.ipynb`](busqueda_G01.ipynb) | El mismo notebook, sin salidas, para volver a ejecutar la búsqueda | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/J2517/ADA-2026-2-GP01/blob/main/busqueda_G01.ipynb) |
| [`evaluacion_G01.ipynb`](evaluacion_G01.ipynb) | **Evaluación** de `entrega.json` sobre datos privados, sin editar nada | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/J2517/ADA-2026-2-GP01/blob/main/evaluacion_G01.ipynb) |
| [`proyecto_oraculo.ipynb`](proyecto_oraculo.ipynb) | Notebook original del curso (sin modificar), como referencia | — |

## Estructura del repositorio

```
ADA-2026-2-GP01/
├── busqueda.py                    # Algoritmo de búsqueda (fases 0–4)
├── registro.py                    # Bitácora de evaluaciones y exportación a CSV
├── evaluacion.py                  # Carga/validación de entrega.json, lectura de datos, verificadores
├── busqueda_G01.ipynb             # Notebook de búsqueda (limpio)
├── busqueda_G01_resultado.ipynb   # Notebook de búsqueda ejecutado (evidencia)
├── evaluacion_G01.ipynb           # Notebook de evaluación para la profesora
├── entrega.json                   # Configuración ganadora
├── registro_busqueda.csv          # Las 103 evaluaciones (configuración, precisión, costo, familias violadas)
├── cribado.csv                    # Resultado de la fase 1 por ranura y opción
├── oraculo.py · ayudas.py         # Del curso, sin modificar
├── datos_visibles.json            # Del curso: 100 búsqueda + 87 validación (+ 263 descartadas)
├── proyecto_oraculo.ipynb         # Notebook original del curso
├── requirements.txt               # Dependencias para Colab (GPU)
├── requirements-dev.txt           # Dependencias para las pruebas locales
├── tests/                         # Pruebas con un oráculo simulado (sin GPU)
└── docs/
    ├── MANUAL_TECNICO.md          # Manual técnico de la solución base
    └── README_original_curso.md   # Instrucciones originales del repositorio del curso
```

## Cómo ejecutar

**Requisitos:** Google Colab con GPU T4 y Python 3 del entorno de Colab. Cada notebook instala sus dependencias en la primera celda.

### Búsqueda (opcional: el resultado ya está en el repositorio)
1. Abrir `busqueda_G01.ipynb` en Colab y elegir **Entorno de ejecución ▸ Cambiar tipo ▸ T4 GPU**.
2. Ejecutar la celda 1 y luego **Entorno de ejecución ▸ Reiniciar sesión**.
3. Desde la celda 2: **Entorno de ejecución ▸ Ejecutar las celdas posteriores**. Pide acceso a Google Drive para guardar el caché.
4. Desde cero tarda unas 5–6 horas de GPU (≈ 3–3,7 min por evaluación). Si la sesión se corta, al volver a ejecutar lo ya generado sale del caché de Drive en segundos.

### Evaluación
1. Abrir `evaluacion_G01.ipynb` en Colab con **T4 GPU**.
2. **Ejecutar todo.** La primera celda instala las dependencias y reinicia la sesión sola.
3. **Ejecutar todo** otra vez.
4. Cuando se pida, subir el archivo de prueba (JSON o JSONL). Si se cancela, corre en modo demostración con la partición de validación.

### Pruebas (local, sin GPU, ~1 s)
```bash
pip install -r requirements-dev.txt
python -m pytest -q tests/
```
Resultado esperado: `23 passed`.

## Documentación

- **Manual técnico:** [`docs/MANUAL_TECNICO.md`](docs/MANUAL_TECNICO.md). Entorno, arquitectura, algoritmo, formatos de entrada/salida, uso, pruebas y limitaciones.
- **Informe académico** (formalización, NP-dureza por reducción desde 3-SAT, proceso y resultados): se entrega aparte.

## Créditos

- `oraculo.py`, `ayudas.py`, `datos_visibles.json` y `proyecto_oraculo.ipynb` son del repositorio del curso ([DanielMelo404/Proyecto-Oraculo-AyD](https://github.com/DanielMelo404/Proyecto-Oraculo-AyD)) y no se modificaron.
- Verificadores de restricciones: [allenai/open-instruct](https://github.com/allenai/open-instruct) (IFEvalG) e [allenai/IFBench](https://github.com/allenai/IFBench).
- Modelo: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B).
