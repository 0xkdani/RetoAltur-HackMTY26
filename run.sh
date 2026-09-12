#!/usr/bin/env bash
# Levanta el servicio. PORT=8000 por defecto.
set -euo pipefail
cd "$(dirname "$0")"

PY="./.venv/Scripts/python.exe"
[ -x "$PY" ] || PY="./.venv/bin/python"
[ -x "$PY" ] || PY="python"

echo "Servicio en http://0.0.0.0:${PORT:-8000}  (docs en /docs)"
exec "$PY" -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
