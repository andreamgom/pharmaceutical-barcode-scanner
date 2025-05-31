# components/sidebar.py
import streamlit as st
from pathlib import Path

def create_sidebar():
    """Sidebar simplificada sin opciones confusas"""
    
    st.sidebar.title("Configuración")
    
    # Solo información, no opciones
    st.sidebar.header("Sistema")
    st.sidebar.info("Detector híbrido activado")
    
    # Modelo YOLO fijo
    yolo_model = "../runs/detect/yolov10_train7/weights/best.pt"
    
    # Verificar si existe
    model_path = Path(yolo_model)
    if model_path.exists():
        st.sidebar.success("Modelo YOLO: Disponible")
    else:
        st.sidebar.error("Modelo YOLO: No encontrado")
        yolo_model = "yolov8n.pt"  # Fallback
    
    # Configuración de validación
    st.sidebar.header("Validación CIMA")
    
    validate_with_cima = st.sidebar.checkbox(
        "Validar códigos detectados",
        value=True,
        help="Valida códigos con la base de datos oficial CIMA"
    )
    
    # Estado del sistema
    with st.sidebar.expander("Estado del Sistema"):
        st.write("Detector: Híbrido")
        st.write("Gradientes: Habilitado")
        st.write(f"Modelo: {'Disponible' if model_path.exists() else 'Por defecto'}")
    
    # Retornar configuración fija
    return {
        'detector_type': 'Híbrido',
        'yolo_model': yolo_model,
        'use_gradient': True,
        'validate_with_cima': validate_with_cima,
        'debug_mode': False  # Siempre False
    }
