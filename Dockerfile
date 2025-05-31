# Dockerfile para Hugging Face Spaces
FROM python:3.10-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Instalar dependencias del sistema (TODAS las que necesitas)
RUN apt-get update && apt-get install -y \
    libzbar0 \
    zbar-tools \
    libzbar-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements.txt COMPLETO
COPY farmacia_interface/requirements.txt .

# Instalar dependencias Python (TODAS, sin restricciones)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar todo el código
COPY farmacia_interface/ .

# Crear directorios necesarios
RUN mkdir -p runs/detect/yolov10_train7/weights/ && \
    mkdir -p data/temp && \
    mkdir -p data/uploads

# Exponer puerto de Hugging Face
EXPOSE 7860

# Comando de inicio para Hugging Face Spaces
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.maxUploadSize=200"]
