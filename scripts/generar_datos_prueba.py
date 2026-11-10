"""
Genera un dataset PEQUEÑO de prueba (prueba de humo) para el pipeline de destilación.

- Define ejercicios tipo Senecode con tests y restricciones.
- Para cada ejercicio crea 3 variantes de código: correcta, con error en caso borde
  y que viola una restricción.
- Ejecuta los tests de cada variante en un subproceso con límite de tiempo.
- Revisa las restricciones con el módulo ast.
- Arma la respuesta esperada (JSON) a partir de esos resultados, con plantillas.
- Separa por EJERCICIO: los de eval nunca aparecen en train.

Las respuestas son de plantilla: sirven para probar el PROCESO, no para un modelo final.
El modelo final se entrena con respuestas del modelo profesor (Gemma 4 31B).

NOTA: el subproceso con timeout NO es un sandbox de seguridad. Para código real de
estudiantes, ejecutar dentro de un contenedor sin red (Docker/Apptainer).

Uso:  python generar_datos_prueba.py --salida data
"""

import argparse
import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

SYSTEM_PROMPT = (
    "Eres un evaluador de código para estudiantes principiantes de Python.\n"
    "Recibes el enunciado, el código del estudiante, los resultados de los tests y un análisis "
    "estático. Responde SOLO con un JSON con esta forma:\n"
    '{"correccion": true/false, "casos_borde": true/false, "restricciones": true/false, '
    '"evidencia": "qué viste que justifica cada valor", '
    '"feedback": "2-3 frases en español para el estudiante, sin dar la solución"}'
)

RESTRICCION_TXT = "No usar for, while, listas ni el operador **."

# Cada test: (argumentos, esperado, es_caso_borde, descripción)
EJERCICIOS = [
    {
        "nombre": "potenciador",
        "enunciado": "Escribe una función potenciador(base, exponente) que calcule base elevado a "
                     "exponente (exponente entero >= 0).",
        "tests": [((2, 3), 8, False, "base=2, exponente=3"),
                  ((5, 1), 5, False, "base=5, exponente=1"),
                  ((7, 0), 1, True, "exponente=0")],
        "correcta": """
            def potenciador(base, exponente):
                if exponente == 0:
                    return 1
                return base * potenciador(base, exponente - 1)
        """,
        "borde": """
            def potenciador(base, exponente):
                if exponente == 1:
                    return base
                return base * potenciador(base, exponente - 1)
        """,
        "viola": """
            def potenciador(base, exponente):
                resultado = 1
                for i in range(exponente):
                    resultado = resultado * base
                return resultado
        """,
    },
    {
        "nombre": "factorial",
        "enunciado": "Escribe una función factorial(n) que calcule n! para un entero n >= 0.",
        "tests": [((5,), 120, False, "n=5"),
                  ((3,), 6, False, "n=3"),
                  ((0,), 1, True, "n=0")],
        "correcta": """
            def factorial(n):
                if n == 0:
                    return 1
                return n * factorial(n - 1)
        """,
        "borde": """
            def factorial(n):
                if n == 1:
                    return 1
                return n * factorial(n - 1)
        """,
        "viola": """
            def factorial(n):
                total = 1
                while n > 1:
                    total = total * n
                    n = n - 1
                return total
        """,
    },
    {
        "nombre": "suma_digitos",
        "enunciado": "Escribe una función suma_digitos(n) que retorne la suma de los dígitos de un "
                     "entero n >= 0.",
        "tests": [((123,), 6, False, "n=123"),
                  ((9045,), 18, False, "n=9045"),
                  ((0,), 0, True, "n=0")],
        "correcta": """
            def suma_digitos(n):
                if n < 10:
                    return n
                return n % 10 + suma_digitos(n // 10)
        """,
        "borde": """
            def suma_digitos(n):
                if n == 0:
                    return 0
                if n < 10:
                    return n
                return n % 10 + suma_digitos(n / 10)
        """,
        "viola": """
            def suma_digitos(n):
                digitos = [int(c) for c in str(n)]
                return sum(digitos)
        """,
    },
    {
        "nombre": "invertir",
        "enunciado": "Escribe una función invertir(texto) que retorne el texto al revés.",
        "tests": [(("hola",), "aloh", False, "texto='hola'"),
                  (("ab",), "ba", False, "texto='ab'"),
                  (("",), "", True, "texto vacío")],
        "correcta": """
            def invertir(texto):
                if texto == "":
                    return ""
                return invertir(texto[1:]) + texto[0]
        """,
        "borde": """
            def invertir(texto):
                if len(texto) == 1:
                    return texto
                return invertir(texto[1:]) + texto[0]
        """,
        "viola": """
            def invertir(texto):
                resultado = ""
                for letra in texto:
                    resultado = letra + resultado
                return resultado
        """,
    },
    {
        "nombre": "fibonacci",
        "enunciado": "Escribe una función fibonacci(n) que retorne el n-ésimo número de Fibonacci, "
                     "con fibonacci(0)=0 y fibonacci(1)=1.",
        "tests": [((6,), 8, False, "n=6"),
                  ((2,), 1, False, "n=2"),
                  ((0,), 0, True, "n=0")],
        "correcta": """
            def fibonacci(n):
                if n < 2:
                    return n
                return fibonacci(n - 1) + fibonacci(n - 2)
        """,
        "borde": """
            def fibonacci(n):
                if n == 1:
                    return 1
                if n == 2:
                    return 1
                return fibonacci(n - 1) + fibonacci(n - 2)
        """,
        "viola": """
            def fibonacci(n):
                serie = [0, 1]
                for i in range(2, n + 1):
                    serie.append(serie[-1] + serie[-2])
                return serie[n]
        """,
    },
]

EJERCICIOS_EVAL = {"fibonacci"}  # ejercicios completos reservados para evaluación


def ejecutar_tests(codigo, nombre_funcion, tests, timeout=5):
    """Ejecuta los tests en un subproceso. Devuelve una lista de resultados por test."""
    casos = [{"args": list(a), "esperado": e} for a, e, _, _ in tests]
    arnes = textwrap.dedent(f"""
        import json, sys
        sys.setrecursionlimit(2000)
        {{codigo}}
        resultados = []
        for caso in json.loads(sys.argv[1]):
            try:
                obtenido = {nombre_funcion}(*caso["args"])
                resultados.append({{"ok": obtenido == caso["esperado"], "obtenido": repr(obtenido)}})
            except RecursionError:
                resultados.append({{"ok": False, "obtenido": "RecursionError (recursión infinita)"}})
            except Exception as ex:
                resultados.append({{"ok": False, "obtenido": type(ex).__name__ + ": " + str(ex)}})
        print(json.dumps(resultados))
    """).replace("{codigo}", textwrap.dedent(codigo))
    try:
        salida = subprocess.run([sys.executable, "-c", arnes, json.dumps(casos)],
                                capture_output=True, text=True, timeout=timeout)
        return json.loads(salida.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return [{"ok": False, "obtenido": "Timeout"} for _ in tests]


def revisar_restricciones(codigo):
    """Busca construcciones prohibidas con ast. Devuelve lista de (construcción, línea)."""
    violaciones = []
    for nodo in ast.walk(ast.parse(textwrap.dedent(codigo).strip())):
        if isinstance(nodo, (ast.For, ast.comprehension)):
            violaciones.append(("for", getattr(nodo, "lineno", getattr(nodo.target, "lineno", "?"))))
        elif isinstance(nodo, ast.While):
            violaciones.append(("while", nodo.lineno))
        elif isinstance(nodo, (ast.List, ast.ListComp)):
            violaciones.append(("listas", nodo.lineno))
        elif isinstance(nodo, ast.BinOp) and isinstance(nodo.op, ast.Pow):
            violaciones.append(("el operador **", nodo.lineno))
    vistas, unicas = set(), []
    for v in violaciones:
        if v[0] not in vistas:
            vistas.add(v[0])
            unicas.append(v)
    return unicas


def construir_ejemplo(ej, variante, codigo):
    resultados = ejecutar_tests(codigo, ej["nombre"], ej["tests"])
    violaciones = revisar_restricciones(codigo)

    lineas_tests = []
    for (args, esperado, borde, desc), r in zip(ej["tests"], resultados):
        estado = "PASA" if r["ok"] else f"FALLA (esperado {esperado!r}, obtenido {r['obtenido']})"
        lineas_tests.append(f"- {desc}{' [caso borde]' if borde else ''}: {estado}")
    analisis = ("Sin construcciones prohibidas." if not violaciones else
                "; ".join(f"usa {c} (línea {l})" for c, l in violaciones))

    usuario = (f"Enunciado: {ej['enunciado']}\nRestricción: {RESTRICCION_TXT}\n\n"
               f"Código del estudiante:\n{textwrap.dedent(codigo).strip()}\n\n"
               f"Resultados de tests:\n" + "\n".join(lineas_tests) +
               f"\n\nAnálisis estático: {analisis}")

    normales_ok = all(r["ok"] for t, r in zip(ej["tests"], resultados) if not t[2])
    borde_ok = all(r["ok"] for t, r in zip(ej["tests"], resultados) if t[2])
    correccion = normales_ok and borde_ok
    restricciones = not violaciones

    evidencia = []
    evidencia.append("Todos los tests pasan." if correccion else
                     "Fallan: " + ", ".join(t[3] for t, r in zip(ej["tests"], resultados) if not r["ok"]) + ".")
    evidencia.append("Respeta las restricciones." if restricciones else
                     "Viola la restricción: " + ", ".join(c for c, _ in violaciones) + ".")

    if not borde_ok:
        caso = next(t[3] for t, r in zip(ej["tests"], resultados) if t[2] and not r["ok"])
        feedback = (f"Tu función funciona en los casos generales, pero falla cuando {caso}. "
                    "Revisa si tu caso base cubre ese valor y qué pasa en la primera llamada.")
    elif not correccion:
        feedback = ("Tu función no devuelve el resultado esperado en algunos casos. "
                    "Prueba a mano con un ejemplo pequeño y revisa cada paso de la recursión.")
    elif not restricciones:
        feedback = (f"Tu solución da el resultado correcto, pero usa {violaciones[0][0]}, que el "
                    "enunciado prohíbe. Intenta expresar la misma idea con recursión.")
    else:
        feedback = ("¡Bien hecho! Tu solución es correcta, cubre los casos borde y respeta las "
                    "restricciones. Fíjate en cómo tu caso base detiene la recursión.")

    respuesta = {"correccion": correccion, "casos_borde": borde_ok, "restricciones": restricciones,
                 "evidencia": " ".join(evidencia), "feedback": feedback}

    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": usuario},
                         {"role": "assistant", "content": json.dumps(respuesta, ensure_ascii=False)}],
            "meta": {"ejercicio": ej["nombre"], "variante": variante}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--salida", default="data")
    args = parser.parse_args()
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    train, eval_ = [], []
    for ej in EJERCICIOS:
        for variante in ("correcta", "borde", "viola"):
            ejemplo = construir_ejemplo(ej, variante, ej[variante])
            (eval_ if ej["nombre"] in EJERCICIOS_EVAL else train).append(ejemplo)

    for nombre, filas in (("train.jsonl", train), ("eval.jsonl", eval_)):
        with open(salida / nombre, "w", encoding="utf-8") as f:
            for fila in filas:
                # "meta" se quita: el script de entrenamiento solo espera "messages"
                f.write(json.dumps({"messages": fila["messages"]}, ensure_ascii=False) + "\n")

    print(f"train: {len(train)} ejemplos | eval: {len(eval_)} ejemplos -> {salida}/")
    for fila in train + eval_:
        r = json.loads(fila["messages"][2]["content"])
        print(f"  {fila['meta']['ejercicio']:13s} {fila['meta']['variante']:9s} "
              f"correccion={r['correccion']!s:5s} borde={r['casos_borde']!s:5s} "
              f"restricciones={r['restricciones']}")


if __name__ == "__main__":
    main()
