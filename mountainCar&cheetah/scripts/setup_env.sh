#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="python3"
INSTALL_DEV=0
TINYSIM_PATH="${TINYSIM_PATH:-}"
SKIP_TINYSIM=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Create a virtual environment and install MountainCar dependencies.

Options:
  --python <bin>         Python executable to use (default: python3)
  --venv <dir>           Virtualenv directory (default: .venv under mountaincar)
  --dev                  Install development dependencies (pytest)
  --tinysim-path <path>  Install TinySim from local path (preferred when available)
  --skip-tinysim         Skip TinySim installation attempt
  -h, --help             Show this help

Examples:
  ./scripts/setup_env.sh
  ./scripts/setup_env.sh --dev
  ./scripts/setup_env.sh --python python3.11 --tinysim-path ../TinySim
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --dev)
      INSTALL_DEV=1
      shift
      ;;
    --tinysim-path)
      TINYSIM_PATH="$2"
      shift 2
      ;;
    --skip-tinysim)
      SKIP_TINYSIM=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

echo "[setup] Root: ${ROOT_DIR}"
echo "[setup] Python: ${PYTHON_BIN}"
echo "[setup] Venv: ${VENV_DIR}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
pip install -r "$ROOT_DIR/requirements.txt"

if [[ "$INSTALL_DEV" -eq 1 ]]; then
  pip install -r "$ROOT_DIR/requirements-dev.txt"
fi

if [[ "$SKIP_TINYSIM" -eq 1 ]]; then
  echo "[setup] Skipping TinySim installation (--skip-tinysim)."
else
  if [[ -n "$TINYSIM_PATH" ]]; then
    if [[ -d "$TINYSIM_PATH" ]]; then
      echo "[setup] Installing TinySim from local path: $TINYSIM_PATH"
      pip install -e "$TINYSIM_PATH"
    else
      echo "[setup] TinySim path not found: $TINYSIM_PATH" >&2
      exit 1
    fi
  else
    echo "[setup] Attempting 'pip install tinysim'..."
    if ! pip install tinysim; then
      cat <<MSG
[setup] Could not install 'tinysim' from package index.
        Re-run with --tinysim-path <local TinySim repo path>, for example:
        ./scripts/setup_env.sh --tinysim-path ../TinySim
MSG
      exit 1
    fi
  fi
fi

echo "[setup] Done. Activate with:"
echo "        source '$VENV_DIR/bin/activate'"
