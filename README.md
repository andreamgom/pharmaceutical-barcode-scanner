---
title: Pharmaceutical Barcode Scanner
emoji: 💊
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: farmcode_detector/streamlit_app.py
python_version: 3.11
pinned: false
---

# Pharmaceutical Barcode Scanner

Sistema de detección automática de códigos de barras en cupones farmacéuticos españoles desarrollado con Streamlit, YOLO v10 y OpenCV.

## 🚀 Características

- **Detección automática** con YOLO v10 entrenado específicamente para códigos farmacéuticos
- **Algoritmo de gradientes** como método alternativo de detección
- **Procesamiento de imágenes** con OpenCV y scikit-image
- **Decodificación de códigos** con pyzbar/ZBar
- **Validación CIMA** para códigos farmacéuticos españoles
- **Interfaz web moderna** con Streamlit

## 📁 Estructura del Proyecto

pharmaceutical-barcode-scanner/
├── README.md
├── farmcode_detector/
│ ├── streamlit_app.py # Aplicación principal
│ ├── requirements.txt # Dependencias Python
│ ├── packages.txt # Dependencias del sistema
│ ├── components/ # Componentes de la interfaz
│ ├── core/ # Lógica de detección
│ └── utils/ # Utilidades
└── runs/ # Modelos YOLO entrenados


## 🌐 Uso en Hugging Face Spaces

Esta aplicación está optimizada para ejecutarse en Hugging Face Spaces:

1. **Crea un nuevo Space** en [huggingface.co/new-space](https://huggingface.co/new-space)
2. **Configura**: SDK = Streamlit, Python = 3.11
3. **Conecta** este repositorio o sube los archivos
4. **El despliegue es automático** (8-12 minutos)

## 📱 Cómo Usar la Aplicación

1. **Sube imágenes** de cupones farmacéuticos (JPG, PNG, hasta 10MB)
2. **Selecciona el método** de detección (YOLO recomendado)
3. **Visualiza los resultados** con códigos detectados automáticamente
4. **Valida códigos** con la base de datos CIMA (opcional)
5. **Exporta resultados** en formato CSV

## 🔧 Tecnologías Utilizadas

- **Python 3.11** - Lenguaje principal
- **Streamlit** - Framework web interactivo
- **YOLO v10** - Detección de objetos en tiempo real
- **OpenCV** - Procesamiento de imágenes
- **pyzbar** - Decodificación de códigos de barras
- **PyTorch** - Framework de deep learning
- **scikit-image** - Algoritmos de imagen

## 📊 Métodos de Detección

| Método | Velocidad | Precisión | Disponibilidad |
|--------|-----------|-----------|----------------|
| YOLO v10 | 2-4s | Alta | 24/7 |
| Gradientes | 1-2s | Media | 24/7 |

## 🏥 Validación CIMA

La aplicación puede validar códigos farmacéuticos españoles usando la API oficial de CIMA:
- Verifica existencia del medicamento
- Obtiene información del laboratorio
- Comprueba estado de autorización
- Detecta problemas de suministro

## 📄 Licencia

MIT License © 2024 Andrea Martín Gómez

Desarrollado como Trabajo de Fin de Grado en la Universidad de Las Palmas de Gran Canaria.
