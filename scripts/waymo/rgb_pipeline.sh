#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${ADSG_PYTHON:-/root/miniconda3/envs/dggt/bin/python}"
exec "$PYTHON" -m tools.rgb_only.pipeline "$@"
