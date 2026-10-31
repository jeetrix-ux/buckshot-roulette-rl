#!/usr/bin/env bash
# Run any engine command (validate / tournament) on Linux via Docker, from Windows Git Bash.
#   train/engine.sh python -m engine.validate my_bot/ --backend docker
# The repo is mounted at Docker Desktop's VM path for C:, so sandbox `-v` mounts resolve on the host.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd -W 2>/dev/null || pwd)"
DRIVE="${REPO%%:*}"
VM_PATH="/run/desktop/mnt/host/${DRIVE,,}${REPO#*:}"
docker image inspect cses-engine >/dev/null 2>&1 || docker build -q -t cses-engine -f "$REPO/train/Dockerfile.engine" "$REPO/train"
MSYS_NO_PATHCONV=1 docker run --rm -i -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$REPO:$VM_PATH" -w "$VM_PATH" cses-engine "$@"
