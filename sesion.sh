# Prepara una sesión de trabajo en WSL. Hay que cargarlo con "source", no con "bash":
#   source sesion.sh
#
# - Activa el entorno de Python
# - Permite ejecutar llamafile en WSL (desactiva la integración con Windows hasta reiniciar WSL)
# - Monta las Descargas de red en /mnt/descargas
# - Define la variable MODELO con el .gguf de modelos/

# Cambia esta ruta si tu carpeta de Descargas de red es otra:
DESCARGAS_RED='\\CODD.sis.virtual.uniandes.edu.co\Estudiantes\Profiles\l.ruiz11\Downloads'

cd "$(dirname "${BASH_SOURCE[0]}")" || return
source "$HOME/.local/bin/env" 2>/dev/null
source .venv/bin/activate

# llamafile: evita "APE is running on WIN32 inside WSL"
for f in /proc/sys/fs/binfmt_misc/WSLInterop*; do
  [ -e "$f" ] && sudo sh -c "echo -1 > '$f'"
done

# Descargas de red
if ! mountpoint -q /mnt/descargas; then
  sudo mkdir -p /mnt/descargas
  sudo mount -t drvfs "$DESCARGAS_RED" /mnt/descargas \
    || echo "Aviso: no se pudieron montar las Descargas de red (no es grave)."
fi

MODELO=$(ls modelos/*.gguf 2>/dev/null | head -1)
export MODELO
echo "Sesión lista. MODELO=$MODELO"
echo "Recuerda: mientras tanto no puedes abrir programas de Windows desde WSL (se restablece al reiniciar WSL)."
