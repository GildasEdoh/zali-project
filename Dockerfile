FROM python:3.11-slim

# Install system deps for pillow and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# ...
WORKDIR /app

# Copie le requirements.txt depuis app/
COPY app/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Copie tout le code de app/
COPY app/ /app/

RUN mkdir -p /app/uploads

ENV PORT=8000

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8000
ENTRYPOINT ["/start.sh"]