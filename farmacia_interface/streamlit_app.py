# streamlit_app.py - VERSIÓN COMPATIBLE CON STREAMLIT CLOUD
import streamlit as st

# ⚠️ IMPORTANTE: set_page_config DEBE ser lo PRIMERO
st.set_page_config(
    page_title="Sistema de Detección de Códigos Farmacéuticos",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DESPUÉS de set_page_config, importar el resto
import pandas as pd
from pathlib import Path
import time
from PIL import Image
import cv2
import numpy as np

# Imports locales CON manejo de errores ROBUSTO
IMPORTS_OK = True
HYBRID_AVAILABLE = False
CIMA_AVAILABLE = False

try:
    from core.hybrid_detector import HybridDetector
    HYBRID_AVAILABLE = True
except ImportError as e:
    st.warning(f"Detector híbrido no disponible: {e}")
    HYBRID_AVAILABLE = False

try:
    from core.cima_validator import CIMAValidator
    CIMA_AVAILABLE = True
except ImportError as e:
    st.warning(f"Validador CIMA no disponible: {e}")
    CIMA_AVAILABLE = False

# Si ninguno está disponible, mostrar modo demo
if not HYBRID_AVAILABLE and not CIMA_AVAILABLE:
    IMPORTS_OK = False

# CSS básico integrado
def load_custom_css():
    st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
    }
    .header-title {
        text-align: center;
        color: #2E7D32;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .demo-mode {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    load_custom_css()
    
    st.markdown('<h1 class="header-title">Sistema de Detección de Códigos</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #555;">Análisis Inteligente de Códigos Farmacéuticos</p>', unsafe_allow_html=True)
    
    if not IMPORTS_OK:
        show_demo_mode()
        return
    
    # Sidebar simplificado
    config = create_simple_sidebar()
    
    # Área principal
    tab1, tab2, tab3 = st.tabs(["Subir Imagen", "Resultados", "Validación CIMA"])
    
    with tab1:
        upload_section(config)
    
    with tab2:
        if 'detection_results' in st.session_state:
            results_section()
        else:
            st.info("Sube una imagen primero para ver los resultados")
    
    with tab3:
        if 'detection_results' in st.session_state:
            validation_section()
        else:
            st.info("Procesa una imagen primero para validar códigos")

def show_demo_mode():
    """Muestra modo demo cuando los módulos no están disponibles"""
    
    st.markdown("""
    <div class="demo-mode">
    <h3>🚧 Modo Demo - Dependencias Limitadas</h3>
    <p>Algunas funcionalidades no están disponibles en Streamlit Cloud debido a limitaciones de dependencias.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📋 Funcionalidades del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Disponible en producción local:
        - 🤖 Detección híbrida YOLO + Gradientes
        - 📊 Análisis de grid 6×4 / 7×4
        - 🔍 Decodificación múltiple (pyzbar + zxingcpp)
        - ✅ Validación CIMA individual
        - 📝 Tabla editable de códigos
        - 📥 Exportación CSV
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ Limitado en Streamlit Cloud:
        - 🔧 Dependencias de sistema (zbar, zxingcpp)
        - 🤖 Modelos YOLO pesados
        - 🔍 Procesamiento intensivo
        - 📊 Funciones avanzadas de CV
        """)
    
    st.subheader("🎯 Arquitectura del Sistema")
    
    st.markdown("""
    ```
    Sistema de Detección Farmacéutica
    ├── 🤖 Detector Híbrido
    │   ├── YOLO v10 (detección de objetos)
    │   └── Gradient Detector (ordenamiento)
    ├── 🔍 Decodificación Robusta
    │   ├── pyzbar (códigos EAN-13)
    │   └── zxingcpp (respaldo)
    ├── ✅ Validación CIMA
    │   └── API oficial española
    └── 📊 Interfaz Streamlit
        ├── Tabla editable
        └── Exportación CSV
    ```
    """)
    
    st.subheader("📸 Capturas del Sistema")
    
    # Mostrar imágenes de ejemplo si están disponibles
    st.info("Para ver el sistema completo funcionando, ejecuta localmente con todas las dependencias instaladas.")
    
    # Información de contacto
    st.subheader("📞 Información del Proyecto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Proyecto:** Trabajo de Fin de Grado  
        **Universidad:** Universidad de Las Palmas de Gran Canaria  
        **Área:** Detección de códigos farmacéuticos  
        **Tecnologías:** YOLO, OpenCV, Streamlit, FastAPI
        """)
    
    with col2:
        st.markdown("""
        **Características:**
        - Detección automática de headers
        - Grid inteligente 6×4 / 7×4
        - Validación con base de datos oficial
        - Interfaz web profesional
        """)

def create_simple_sidebar():
    """Sidebar simplificado"""
    st.sidebar.title("Configuración")
    st.sidebar.info("Modo demo - Funcionalidad limitada en Streamlit Cloud")
    
    return {
        'yolo_model': "yolov8n.pt",
        'use_gradient': True,
        'validate_with_cima': True,
        'debug_mode': False
    }

def upload_section(config):
    """Sección de subida para modo demo"""
    st.header("Subir Imagen de Cupones")
    
    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=['jpg', 'jpeg', 'png'],
        help="En modo demo - funcionalidad limitada"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)
        
        st.warning("⚠️ Modo demo: El procesamiento completo requiere instalación local con todas las dependencias.")
        
        if st.button("Ver Información del Sistema", type="primary"):
            st.info("Para procesamiento completo, ejecuta el sistema localmente con requirements.txt completo.")

def results_section():
    """Sección de resultados para modo demo"""
    st.header("Resultados de Detección")
    st.info("Esta sección mostraría los resultados del análisis en el sistema completo.")

def validation_section():
    """Sección de validación para modo demo"""
    st.header("Validación Individual de Códigos")
    st.info("Esta sección permitiría validar códigos con la base de datos CIMA en el sistema completo.")

if __name__ == "__main__":
    main()
