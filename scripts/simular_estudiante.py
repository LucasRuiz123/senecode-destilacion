"""
Simula a un estudiante usando la GPU, para probar el oportunismo graduado.

Ocupa una cantidad fija de VRAM y hace cálculo en la GPU una fracción del tiempo
(ciclo de trabajo). Cada segundo registra cuántas operaciones logró hacer: si CupIA
le quita GPU, ese número baja. Así se mide el impacto sobre el "usuario local".

Perfiles sugeridos:
  ligero  (navegador, video):   --vram-gb 2  --carga 0.2
  medio   (IDE + algo gráfico):  --vram-gb 6  --carga 0.5
  pesado  (render, juego, CUDA): --vram-gb 12 --carga 1.0

Uso:
  python scripts/simular_estudiante.py --perfil medio --duracion 180 --salida resultados/estudiante_medio_k0.csv
"""

import argparse
import csv
import time

import torch

PERFILES = {
    "ninguno": (0, 0.0),
    "ligero": (2, 0.2),
    "medio": (6, 0.5),
    "pesado": (12, 1.0),
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--perfil", choices=PERFILES, default=None,
                   help="atajo para --vram-gb y --carga")
    p.add_argument("--vram-gb", type=float, default=2, help="VRAM que ocupa el estudiante")
    p.add_argument("--carga", type=float, default=0.5,
                   help="fracción del tiempo calculando en la GPU (0 a 1)")
    p.add_argument("--duracion", type=float, default=120, help="segundos")
    p.add_argument("--tam", type=int, default=4096, help="tamaño de las matrices")
    p.add_argument("--salida", default=None, help="CSV con operaciones por segundo")
    args = p.parse_args()
    if args.perfil:
        args.vram_gb, args.carga = PERFILES[args.perfil]

    if not torch.cuda.is_available():
        raise SystemExit("No hay GPU CUDA visible para PyTorch.")
    dev = "cuda"

    # 1) Ocupar VRAM (como un programa que tiene memoria reservada)
    reserva = None
    if args.vram_gb > 0:
        reserva = torch.empty(int(args.vram_gb * 1024 ** 3), dtype=torch.uint8, device=dev)
        reserva.fill_(1)

    # 2) Matrices para el cálculo
    a = torch.randn(args.tam, args.tam, device=dev, dtype=torch.float16)
    b = torch.randn(args.tam, args.tam, device=dev, dtype=torch.float16)
    torch.cuda.synchronize()

    print(f"Estudiante simulado: {args.vram_gb} GB de VRAM, carga {args.carga:.0%}, "
          f"{args.duracion:.0f} s. Ctrl+C para detener.")

    ventana = 0.1  # cada 100 ms: calcula carga*100 ms y descansa el resto
    filas = []
    inicio = time.time()
    inicio_intervalo, ops_intervalo = inicio, 0
    try:
        while time.time() - inicio < args.duracion:
            t0 = time.time()
            while args.carga > 0 and time.time() - t0 < args.carga * ventana:
                torch.matmul(a, b)
                torch.cuda.synchronize()
                ops_intervalo += 1
            resto = ventana - (time.time() - t0)
            if resto > 0:
                time.sleep(resto)

            ahora = time.time()
            if ahora - inicio_intervalo >= 1.0:
                ops_s = ops_intervalo / (ahora - inicio_intervalo)
                filas.append({"t_s": round(ahora - inicio, 1),
                              "epoch": round(ahora, 1),
                              "ops_por_s": round(ops_s, 2)})
                print(f"  t={ahora - inicio:6.1f} s  {ops_s:8.1f} ops/s", flush=True)
                inicio_intervalo, ops_intervalo = ahora, 0
    except KeyboardInterrupt:
        pass

    if filas:
        media = sum(f["ops_por_s"] for f in filas) / len(filas)
        print(f"\nPromedio: {media:.1f} ops/s en {len(filas)} s medidos")
    if args.salida and filas:
        with open(args.salida, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
        print(f"Guardado en {args.salida}")
    del reserva


if __name__ == "__main__":
    main()
