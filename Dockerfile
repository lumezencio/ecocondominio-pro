# Dockerfile para EcoCondomínio Pro
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dependências do sistema (fonts, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY ecocondominio_pro.py /app/ecocondominio_pro.py

EXPOSE 8501
CMD ["streamlit","run","ecocondominio_pro.py","--server.port=8501","--server.address=0.0.0.0"]
