#!/usr/bin/env bash
# Instala todo el entorno de CupIA/Senecode en un PC de la sala (WSL + RTX 4090).
# Se puede correr varias veces: lo que ya está instalado se salta.
#
# Uso (dentro de la carpeta del repositorio):
#   bash setup_sala.sh
#
# Tarda ~20-30 min la primera vez (casi todo es descarga). Pide la contraseña de Linux para sudo.

set -euo pipefail
cd "$(dirname "$0")"

MODELO_REPO="unsloth/Qwen3.5-4B-GGUF"
MODELO_PATRON="*Q4_K_M*"

paso() { echo; echo "==> $*"; }

paso "1/8 Verificando la GPU"
if ! nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; then
  echo "ERROR: nvidia-smi no funciona en WSL. Revisa el driver de NVIDIA en Windows." >&2
  exit 1
fi

paso "2/8 Paquetes del sistema"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential git cmake curl unzip python3-venv tmux gh

paso "3/8 uv (gestor de Python)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck disable=SC1091
source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
uv --version

paso "4/8 Entorno de Python 3.12 + Unsloth + PyTorch"
if [ ! -d .venv ]; then
  uv venv --python 3.12 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if ! python -c "import unsloth" >/dev/null 2>&1; then
  uv pip install unsloth matplotlib --torch-backend=auto
fi
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch no ve la GPU'; print('GPU para PyTorch:', torch.cuda.get_device_name(0))"

paso "5/8 Carpetas"
mkdir -p data modelos resultados salida checkpoints

paso "6/8 Datos de prueba"
if [ ! -f data/train.jsonl ]; then
  python scripts/generar_datos_prueba.py --salida data
else
  echo "Ya existen (data/train.jsonl)."
fi

paso "7/8 Modelo $MODELO_REPO ($MODELO_PATRON)"
if ! ls modelos/*.gguf >/dev/null 2>&1; then
  hf download "$MODELO_REPO" --include "$MODELO_PATRON" --local-dir modelos
fi
ls -lh modelos/*.gguf

paso "8/8 llamafile"
if [ ! -x llamafile ]; then
  URL=$(curl -s https://api.github.com/repos/mozilla-ai/llamafile/releases/latest \
        | grep -oE 'https://[^"]*/llamafile-[0-9.]+"' | tr -d '"' | head -1)
  if [ -z "$URL" ]; then
    echo "No encontré el ejecutable de llamafile en GitHub. Descárgalo a mano (llamafile-0.10.x) y ponlo aquí como ./llamafile" >&2
    exit 1
  fi
  curl -L -o llamafile "$URL"
  chmod +x llamafile
fi

echo
echo "Listo. En cada sesión nueva de WSL corre:   source sesion.sh"
