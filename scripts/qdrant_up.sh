#!/usr/bin/env bash
# Start Qdrant locally. Uses whichever of podman or docker is installed.
#
# Pinned to an exact version on purpose: `:latest` would change search behaviour under a
# saved demo, and snapshot/format compatibility is version-scoped. A live demo should
# never be surprised by an upgrade.
set -euo pipefail

QDRANT_VERSION="${QDRANT_VERSION:-v1.19.0}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
CONTAINER="${CONTAINER_NAME:-vector-taste-qdrant}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE="$ROOT/qdrant_storage"

if command -v podman >/dev/null 2>&1; then
  RUNTIME=podman
elif command -v docker >/dev/null 2>&1; then
  RUNTIME=docker
else
  echo "error: need podman or docker on PATH." >&2
  echo "  macOS: brew install podman && podman machine init && podman machine start" >&2
  exit 1
fi

# podman on macOS runs a VM that may exist but be stopped.
if [ "$RUNTIME" = podman ] && ! podman info >/dev/null 2>&1; then
  echo "podman machine is not running. Start it with:" >&2
  echo "  podman machine init   # first time only" >&2
  echo "  podman machine start" >&2
  exit 1
fi

if $RUNTIME ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "$CONTAINER already running on port $QDRANT_PORT ($RUNTIME)"
  exit 0
fi

if $RUNTIME ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "starting existing container $CONTAINER"
  $RUNTIME start "$CONTAINER" >/dev/null
else
  mkdir -p "$STORAGE"
  echo "creating $CONTAINER from qdrant/qdrant:$QDRANT_VERSION on port $QDRANT_PORT ($RUNTIME)"
  $RUNTIME run -d \
    --name "$CONTAINER" \
    -p "$QDRANT_PORT:6333" \
    -p "$((QDRANT_PORT + 1)):6334" \
    -v "$STORAGE:/qdrant/storage:z" \
    "docker.io/qdrant/qdrant:$QDRANT_VERSION" >/dev/null
fi

printf 'waiting for qdrant'
for _ in $(seq 1 40); do
  if curl -sf "http://localhost:$QDRANT_PORT/readyz" >/dev/null 2>&1; then
    echo
    curl -s "http://localhost:$QDRANT_PORT/" | sed 's/^/  /'
    echo
    if [ "$QDRANT_PORT" != "6333" ]; then
      echo "note: not on the default port. Put this in your .env:"
      echo "  QDRANT_URL=http://localhost:$QDRANT_PORT"
    fi
    exit 0
  fi
  printf '.'
  sleep 1
done

echo >&2
echo "error: qdrant did not become ready. Logs:" >&2
$RUNTIME logs --tail 20 "$CONTAINER" >&2
exit 1
