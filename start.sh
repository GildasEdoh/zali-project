#!/usr/bin/env sh
: "${PORT:=8000}"
# Use gunicorn + uvicorn worker for production
set -e
cd /app
exec gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT --workers 2