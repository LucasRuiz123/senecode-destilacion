# senecode-destilacion

Pruebas de CupIA para evaluar ejercicios de Senecode con modelos pequeños en los PCs de la sala
(WSL + RTX 4090): destilación profesor → alumno y oportunismo graduado.

## Arrancar en un PC nuevo

```bash
sudo apt-get update && sudo apt-get install -y gh git
gh auth login                      # GitHub.com → HTTPS → Login with a web browser
gh repo clone <tu-usuario>/senecode-destilacion ~/senecode-destilacion
cd ~/senecode-destilacion
bash setup_sala.sh                 # ~20-30 min la primera vez
```

## En cada sesión

```bash
cd ~/senecode-destilacion
source sesion.sh                   # activa el entorno, arregla llamafile, monta Descargas, define $MODELO
```

## Scripts

| Script | Qué hace |
| --- | --- |
| `scripts/generar_datos_prueba.py` | Crea un dataset de prueba (15 ejemplos): ejecuta tests y revisa restricciones con `ast` |
| `scripts/benchmark_oportunismo.py` | Mide throughput, latencia y VRAM con batching o con varias instancias |
| `scripts/simular_estudiante.py` | Simula a un estudiante usando la GPU (VRAM + cálculo) para medir convivencia |
| `scripts/destilar_senecode.py` | Entrena el alumno con LoRA (Unsloth) y exporta a GGUF |

## Experimentos

```bash
# A: curva de batching (1 instancia)
python scripts/benchmark_oportunismo.py --servidor ./llamafile --modelo $MODELO \
  --modo batching --paralelos 1,2,4,8,16 --peticiones 96

# B': solo copias, sin batching (K instancias x 1)
python scripts/benchmark_oportunismo.py --servidor ./llamafile --modelo $MODELO \
  --modo instancias --instancias 1,2,3,4,5 --por-instancia 1 --peticiones 64
```

Los resultados quedan en `resultados/bench_<fecha>/` (`resumen.csv`, `peticiones.csv`, `grafica.png`).
Súbelos al repositorio con:

```bash
git add resultados && git commit -m "Resultados <descripción>" && git push
```

## Resultados hasta ahora (6 nov 2026, ISC401, Qwen3.5 4B Q4_K_M, llamafile 0.10.6)

- **Batching (1 instancia):** de 68 a 188 ejercicios/min al pasar de 1 a 16 simultáneas; punto de equilibrio en 4
  (140 ej/min, 1,7 s por evaluación). VRAM de 3,9 a 6,6 GB.
- **Varias instancias × 4:** no suman capacidad (~140-155 ej/min con 1 a 4 copias); cada copia cuesta ~3,8 GB.
- **Conclusión:** el oportunismo graduado se hace con una sola instancia y concurrencia graduada desde el master,
  no con varias copias.

## Reglas

- Nada de datos reales de estudiantes en el repositorio (`data/` está en `.gitignore`).
- Modelos, entorno y binarios no se suben: se reinstalan con `setup_sala.sh`.
