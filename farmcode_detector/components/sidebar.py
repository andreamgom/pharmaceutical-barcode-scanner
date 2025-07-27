# components/sidebar.py

import streamlit as st
from pathlib import Path

def create_sidebar():
    """Sidebar simplificada que busca el modelo en varias rutas."""
    st.sidebar.title("Configuración")

    # Solo información, no opciones
    st.sidebar.header("Sistema")
    st.sidebar.info("Detector híbrido activado")

    # 1. Definir la lista de posibles rutas relativas
    possible_paths = [
        "runs/detect/yolov10_train7/weights/best.pt",        # Ideal para Streamlit Cloud (desde la raíz)
        "../runs/detect/yolov10_train7/weights/best.pt",       # Si el script se ejecuta desde una carpeta
        "../../runs/detect/yolov10_train7/weights/best.pt",    # Si está aún más anidado
        "../../../runs/detect/yolov10_train7/weights/best.pt"
    ]

    # 2. Buscar el modelo iterando sobre la lista
    found_path = None
    for path_str in possible_paths:
        if Path(path_str).exists():
            found_path = path_str  # Guardamos la primera ruta que funcione
            break                  # Dejamos de buscar

    # 3. Decidir qué modelo usar y mostrar el estado
    if found_path:
        st.sidebar.success("Modelo YOLO: Disponible")
        yolo_model_to_use = found_path
    else:
        st.sidebar.error("Modelo YOLO: No encontrado")
        yolo_model_to_use = "yolov8.pt"  # Modelo de respaldo que se descarga solo
    
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
        # El mensaje de estado depende de si se encontró el modelo o no
        st.write(f"Modelo: {'Disponible' if found_path else 'Por defecto'}")

    # Retornar configuración fija
    return {
        'detector_type': 'Híbrido',
        'yolo_model': yolo_model_to_use,  # Retornamos la ruta encontrada o la de por defecto
        'use_gradient': True,
        'validate_with_cima': validate_with_cima,
        'debug_mode': True
    }

