#!/usr/bin/env sh
: "${PORT:=8000}"
# Use gunicorn + uvicorn worker for production
exec gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT --workers 2