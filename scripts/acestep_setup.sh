#!/usr/bin/env bash
# Install ACE-Step 1.5 and start its local API server, for GEN_BACKEND=local.
#
# Why this lives in a separate virtualenv:
#   ACE-Step 1.5 pins transformers>=4.51,<4.58. CLAP (this project) needs transformers>=5.
#   They cannot coexist. So ACE-Step is installed alongside and spoken to over its own
#   localhost REST API. That is still fully offline, and it keeps the model resident
#   between requests — loading it costs far more than generating with it.
#
# Not needed for the default demo: GEN_BACKEND=elevenlabs generates in seconds over the network.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${ACESTEP_DIR:-$ROOT/.acestep}"
PORT="${ACESTEP_PORT:-8001}"

command -v uv >/dev/null 2>&1 || { echo "error: uv is required (brew install uv)" >&2; exit 1; }

if [ ! -d "$DIR" ]; then
  echo "cloning ACE-Step 1.5 into $DIR"
  git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5 "$DIR"
fi

cd "$DIR"
echo "installing dependencies (this pulls ~2GB of wheels)"
# PyPI intermittently 502s on large wheels; retry rather than fail the whole setup.
for i in 1 2 3 4 5; do
  if uv sync; then break; fi
  echo "  attempt $i failed, retrying in 15s..."
  sleep 15
done

cat <<EOF

ACE-Step installed at $DIR

Start the API server (downloads ~9GB of weights on first run):

  cd $DIR && ./start_api_server_macos.sh          # Apple Silicon (MLX)
  cd $DIR && ./start_api_server.sh                # NVIDIA

Then point this project at it:

  GEN_BACKEND=local ACESTEP_URL=http://localhost:$PORT uv run vt generate <profile>

On a 16GB Mac, force the smaller language model — the auto-selected 1.7B swaps badly
alongside a browser and screen share:

  export ACESTEP_LM_MODEL=acestep-5Hz-lm-0.6B

EOF
