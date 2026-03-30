#!/usr/bin/env sh
: "${PORT:=8000}"
# Use gunicorn + uvicorn worker for production
set -e
cd /app
exec uvicorn main:app --host 0.0.0.0 --port $PORT