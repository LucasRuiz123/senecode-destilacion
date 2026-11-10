r"""
Benchmark de oportunismo graduado: un modelo pequeño (p. ej. Qwen3.5 4B) en UNA GPU.

Compara dos formas de atender muchas evaluaciones de Senecode a la vez:

  batching    -> 1 instancia del servidor, con P peticiones simultáneas (--parallel P)
  instancias  -> K instancias del servidor (cada una con S peticiones simultáneas)
                 K = "escalones" del oportunismo graduado: 1, 2, 3, 4 instancias

Para cada configuración mide:
  - throughput (ejercicios/min y tokens/s totales)
  - latencia por ejercicio (media y p95)
  - VRAM usada (antes, después de cargar y pico durante la carga de trabajo)
  - % de respuestas que son JSON válido

Uso en Windows con llama.cpp (llama-server.exe), sin WSL:

  uv run --python 3.12 --with matplotlib scripts\benchmark_oportunismo.py ^
      --servidor C:\Users\<usuario>\senecode\llama\llama-server.exe ^
      --modelo modelos\<archivo>.gguf --modo ambos --peticiones 64

Uso en WSL con llamafile, dentro de ~/senecode-destilacion con el .venv activado:

  python scripts/benchmark_oportunismo.py \
      --servidor ~/llamafiles/llamafile \
      --modelo modelos/Qwen3.5-4B-Q4_K_M.gguf \
      --modo ambos --paralelos 1,2,4,8,16 --instancias 1,2,3,4 \
      --por-instancia 4 --peticiones 64

Requisitos: solo librería estándar de Python. Opcional: matplotlib para la gráfica.
Los prompts salen de data/train.jsonl y data/eval.jsonl (generar_datos_prueba.py).
"""

import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

PUERTO_BASE = 8100


# ---------------------------------------------------------------- utilidades
def leer_prompts(archivos):
    """Toma system+user de cada ejemplo (sin la respuesta del assistant)."""
    prompts = []
    for archivo in archivos:
        if not Path(archivo).exists():
            continue
        with open(archivo, encoding="utf-8") as f:
            for linea in f:
                if linea.strip():
                    msgs = json.loads(linea)["messages"]
                    prompts.append([m for m in msgs if m["role"] != "assistant"])
    if not prompts:
        raise SystemExit("No encontré prompts. Corre antes generar_datos_prueba.py.")
    return prompts


def vram_usada_mib():
    """VRAM total usada en la GPU 0 (MiB) según nvidia-smi. None si no está disponible."""
    try:
        salida = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return int(salida.stdout.strip().splitlines()[0])
    except Exception:
        return None


class MuestreadorVRAM:
    """Mide la VRAM cada 0,5 s en un hilo aparte y guarda el pico."""

    def __init__(self):
        self.pico = None
        self._parar = threading.Event()
        self._hilo = threading.Thread(target=self._bucle, daemon=True)

    def _bucle(self):
        while not self._parar.is_set():
            v = vram_usada_mib()
            if v is not None:
                self.pico = v if self.pico is None else max(self.pico, v)
            time.sleep(0.5)

    def __enter__(self):
        self._hilo.start()
        return self

    def __exit__(self, *_):
        self._parar.set()
        self._hilo.join()


def http_json(url, cuerpo=None, timeout=600):
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    req = urllib.request.Request(url, data=datos,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def es_json_valido(texto):
    t = texto.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):] if "{" in t else t
    try:
        json.loads(t[t.find("{"): t.rfind("}") + 1])
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- servidores
class Servidor:
    def __init__(self, args, puerto, paralelo, carpeta_logs):
        self.puerto = puerto
        self.url = f"http://127.0.0.1:{puerto}"
        es_llamafile = "llamafile" in Path(args.servidor).name.lower()
        # En WSL, llamafile (formato APE) se lanza con "sh" para evitar el error
        # "Exec format error" cuando la integración con Windows está desactivada.
        lanzador = ["sh", args.servidor] if es_llamafile and os.name == "posix" else [args.servidor]
        cmd = lanzador + (["--server"] if es_llamafile else []) + ["-m", args.modelo,
               "--host", "127.0.0.1", "--port", str(puerto),
               "-ngl", "999",
               "-c", str(args.ctx_por_peticion * paralelo),
               "--parallel", str(paralelo)]
        cmd += shlex.split(args.args_extra)
        self.log = open(carpeta_logs / f"servidor_{puerto}.log", "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)

    def esperar_listo(self, limite=600):
        inicio = time.time()
        while time.time() - inicio < limite:
            if self.proc.poll() is not None:
                raise RuntimeError(f"El servidor del puerto {self.puerto} se cerró. "
                                   f"Revisa {self.log.name}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return time.time() - inicio
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(1)
        raise TimeoutError(f"El servidor del puerto {self.puerto} no respondió a tiempo")

    def apagar(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


def una_peticion(url, mensajes, args):
    cuerpo = {
        "model": "alumno",
        "messages": mensajes,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        # Qwen3.5: sin bloque de razonamiento (si el servidor no lo soporta, lo ignora)
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    try:
        r = http_json(url + "/v1/chat/completions", cuerpo)
        latencia = time.time() - t0
        texto = r["choices"][0]["message"]["content"] or ""
        uso = r.get("usage", {})
        return {"ok": True, "latencia_s": latencia,
                "tokens_salida": uso.get("completion_tokens", 0),
                "tokens_entrada": uso.get("prompt_tokens", 0),
                "json_valido": es_json_valido(texto), "error": ""}
    except Exception as ex:
        return {"ok": False, "latencia_s": time.time() - t0, "tokens_salida": 0,
                "tokens_entrada": 0, "json_valido": False, "error": str(ex)[:200]}


def correr_configuracion(nombre, servidores, concurrencia, prompts, args):
    """Envía args.peticiones evaluaciones repartidas en los servidores (round-robin)."""
    # Calentamiento: una petición por servidor (no se mide)
    for s in servidores:
        una_peticion(s.url, prompts[0], args)

    tareas = [(servidores[i % len(servidores)].url, prompts[i % len(prompts)])
              for i in range(args.peticiones)]
    with MuestreadorVRAM() as m, ThreadPoolExecutor(max_workers=concurrencia) as ex:
        t0 = time.time()
        resultados = list(ex.map(lambda t: una_peticion(t[0], t[1], args), tareas))
        duracion = time.time() - t0

    ok = [r for r in resultados if r["ok"]]
    lat = sorted(r["latencia_s"] for r in ok) or [0]
    tokens = sum(r["tokens_salida"] for r in ok)
    resumen = {
        "configuracion": nombre,
        "instancias": len(servidores),
        "concurrencia_total": concurrencia,
        "peticiones": args.peticiones,
        "exitosas": len(ok),
        "duracion_s": round(duracion, 1),
        "ejercicios_por_min": round(len(ok) / duracion * 60, 1),
        "tokens_por_s_total": round(tokens / duracion, 1),
        "latencia_media_s": round(statistics.mean(lat), 2),
        "latencia_p95_s": round(lat[int(0.95 * (len(lat) - 1))], 2),
        "json_valido_pct": round(100 * sum(r["json_valido"] for r in ok) / max(len(ok), 1), 1),
        "vram_pico_mib": m.pico,
    }
    for r in resultados:
        r["configuracion"] = nombre
    return resumen, resultados


def ejecutar(nombre, paralelos_por_servidor, args, prompts, carpeta):
    """Levanta los servidores, mide VRAM de carga, corre la carga y apaga."""
    vram_antes = vram_usada_mib()
    servidores = []
    try:
        carga_s = []
        for i, par in enumerate(paralelos_por_servidor):
            s = Servidor(args, PUERTO_BASE + i, par, carpeta)
            servidores.append(s)
            carga_s.append(s.esperar_listo())
        vram_cargado = vram_usada_mib()
        concurrencia = sum(paralelos_por_servidor)
        print(f"  [{nombre}] {len(servidores)} instancia(s) listas en "
              f"{sum(carga_s):.0f} s; enviando {args.peticiones} evaluaciones "
              f"con concurrencia {concurrencia}...")
        resumen, filas = correr_configuracion(nombre, servidores, concurrencia, prompts, args)
        resumen["vram_antes_mib"] = vram_antes
        resumen["vram_cargado_mib"] = vram_cargado
        if vram_antes is not None and vram_cargado is not None:
            resumen["vram_modelo_mib"] = vram_cargado - vram_antes
        return resumen, filas
    finally:
        for s in servidores:
            s.apagar()
        time.sleep(3)  # dar tiempo a que la GPU libere la memoria


# ---------------------------------------------------------------- principal
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--servidor", required=True, help="ejecutable de llamafile o llama-server")
    p.add_argument("--modelo", required=True, help="archivo .gguf del modelo")
    p.add_argument("--modo", choices=["batching", "instancias", "ambos"], default="ambos")
    p.add_argument("--paralelos", default="1,2,4,8,16",
                   help="batching: peticiones simultáneas en 1 instancia")
    p.add_argument("--instancias", default="1,2,3,4",
                   help="instancias: escalones a probar")
    p.add_argument("--por-instancia", type=int, default=1,
                   help="instancias: peticiones simultáneas dentro de cada instancia")
    p.add_argument("--peticiones", type=int, default=64,
                   help="evaluaciones por configuración")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--ctx-por-peticion", type=int, default=4096)
    p.add_argument("--args-extra", default=None,
                   help='argumentos extra para el servidor. Por defecto "--gpu nvidia" '
                        'con llamafile y nada con llama-server')
    p.add_argument("--datos", default="data/train.jsonl,data/eval.jsonl")
    p.add_argument("--salida", default="resultados")
    args = p.parse_args()
    if args.args_extra is None:
        args.args_extra = "--gpu nvidia" if "llamafile" in Path(args.servidor).name.lower() else ""

    prompts = leer_prompts(args.datos.split(","))
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta = Path(args.salida) / f"bench_{marca}"
    carpeta.mkdir(parents=True, exist_ok=True)
    print(f"{len(prompts)} prompts distintos. Resultados en {carpeta}/")

    planes = []
    if args.modo in ("batching", "ambos"):
        for par in map(int, args.paralelos.split(",")):
            planes.append((f"batching_x{par}", [par]))
    if args.modo in ("instancias", "ambos"):
        for k in map(int, args.instancias.split(",")):
            planes.append((f"instancias_{k}x{args.por_instancia}", [args.por_instancia] * k))

    resumenes, todas = [], []
    for nombre, paralelos in planes:
        try:
            resumen, filas = ejecutar(nombre, paralelos, args, prompts, carpeta)
        except Exception as ex:
            print(f"  [{nombre}] FALLÓ: {ex}")
            continue
        resumenes.append(resumen)
        todas.extend(filas)
        print(f"  [{nombre}] {resumen['ejercicios_por_min']} ejercicios/min | "
              f"{resumen['tokens_por_s_total']} tok/s | latencia media "
              f"{resumen['latencia_media_s']} s | VRAM pico {resumen['vram_pico_mib']} MiB | "
              f"JSON válido {resumen['json_valido_pct']}%")

    if not resumenes:
        raise SystemExit("Ninguna configuración terminó. Revisa los logs de los servidores.")

    with open(carpeta / "resumen.csv", "w", newline="", encoding="utf-8") as f:
        campos = list(dict.fromkeys(k for r in resumenes for k in r))
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(resumenes)
    with open(carpeta / "peticiones.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(todas[0].keys()))
        w.writeheader()
        w.writerows(todas)

    print("\nResumen")
    print(f"{'configuración':22s} {'ej/min':>8s} {'tok/s':>8s} {'lat media':>10s} "
          f"{'lat p95':>8s} {'VRAM pico':>10s}")
    for r in resumenes:
        print(f"{r['configuracion']:22s} {r['ejercicios_por_min']:>8} "
              f"{r['tokens_por_s_total']:>8} {r['latencia_media_s']:>10} "
              f"{r['latencia_p95_s']:>8} {str(r['vram_pico_mib']):>10}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
        for prefijo, color in (("batching", "#534AB7"), ("instancias", "#0F6E56")):
            rs = [r for r in resumenes if r["configuracion"].startswith(prefijo)]
            if not rs:
                continue
            x = [r["concurrencia_total"] for r in rs]
            ejes[0].plot(x, [r["ejercicios_por_min"] for r in rs], "o-", color=color, label=prefijo)
            ejes[1].plot(x, [r["vram_pico_mib"] or 0 for r in rs], "o-", color=color, label=prefijo)
        ejes[0].set(xlabel="evaluaciones simultáneas", ylabel="ejercicios por minuto",
                    title="Throughput")
        ejes[1].set(xlabel="evaluaciones simultáneas", ylabel="VRAM pico (MiB)",
                    title="Memoria de GPU")
        for e in ejes:
            e.grid(alpha=0.3)
            e.legend()
        fig.tight_layout()
        fig.savefig(carpeta / "grafica.png", dpi=150)
        print(f"\nGráfica: {carpeta / 'grafica.png'}")
    except ImportError:
        print("\n(Instala matplotlib para generar la gráfica: uv pip install matplotlib)")


if __name__ == "__main__":
    main()
