#!/bin/bash
# Install py-spinnaker2 and Brian2 into .venv.
# py-spinnaker2 is not on PyPI. Its setup script uses a bash-4 array subscript,
# which macOS /bin/bash 3.2 rejects, so this pins the snn-core checkout first
# and then installs the Python package. No SpiNNaker2 board is required.

set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

PY=""
if command -v python3.12 >/dev/null 2>&1; then
  PY="python3.12"
elif command -v uv >/dev/null 2>&1; then
  uv python install 3.12
  PY="$(uv python find 3.12)"
elif [ -x "$HOME/.local/bin/uv" ]; then
  "$HOME/.local/bin/uv" python install 3.12
  PY="$("$HOME/.local/bin/uv" python find 3.12)"
else
  echo "Python 3.12 is required (py-spinnaker2 needs >= 3.10)." >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
fi

py_install() {
  if .venv/bin/python -m pip --version >/dev/null 2>&1; then
    .venv/bin/python -m pip install "$@"
  elif command -v uv >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python "$@"
  elif [ -x "$HOME/.local/bin/uv" ]; then
    "$HOME/.local/bin/uv" pip install --python .venv/bin/python "$@"
  else
    echo "pip is not available in .venv and uv is not installed." >&2
    exit 1
  fi
}

if .venv/bin/python -m pip --version >/dev/null 2>&1; then
  .venv/bin/python -m pip install -U pip
fi
py_install -r requirements.txt

SRC=".tools/py-spinnaker2"
if [ ! -d "$SRC/.git" ]; then
  mkdir -p .tools
  git clone --depth 1 https://gitlab.com/spinnaker2/py-spinnaker2.git "$SRC"
fi

LIBS_COMMIT="$(sed -n 's/^LIBS_COMMIT="\([0-9a-f]*\)".*/\1/p' "$SRC/setup_libs.sh")"
LIBS="$SRC/src/spinnaker2/libs"
if [ ! -d "$LIBS/.git" ]; then
  rm -rf "$LIBS"
  git clone https://gitlab.com/spinnaker2/snn-core.git "$LIBS"
fi
git -C "$LIBS" fetch origin "$LIBS_COMMIT"
git -C "$LIBS" checkout --detach "$LIBS_COMMIT"

py_install "$SRC"
echo "Installed py-spinnaker2 and Brian2 into .venv"
