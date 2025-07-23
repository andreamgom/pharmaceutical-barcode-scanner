# streamlit_app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import time
from PIL import Image
import cv2
import numpy as np
import os

# Configuración específica para OpenCV en entornos headless
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Import de OpenCV con manejo de errores robusto
try:
    import cv2
    # Verificar que OpenCV puede funcionar básicamente
    test_array = np.zeros((100, 100, 3), dtype=np.uint8)
    _ = cv2.cvtColor(test_array, cv2.COLOR_BGR2RGB)
    CV2_AVAILABLE = True
    st.success("✅ OpenCV funcionando correctamente")
except Exception as e:
    CV2_AVAILABLE = False
    st.error(f"❌ Error con OpenCV: {e}")
    
    # Mensaje informativo para usuarios
    st.warning("""
    🔧 **Estado de la Aplicación**: Configurando dependencias del sistema
    
    La aplicación está en proceso de instalación de librerías gráficas necesarias.
    Esto puede tomar unos minutos adicionales.
    
    **¿Qué está pasando?**
    - Las dependencias de Python ✅ están instaladas
    - Las librerías del sistema ⏳ se están configurando
    - OpenCV requiere acceso a librerías gráficas específicas
    
    **Próximos pasos:**
    1. Recarga la página en 2-3 minutos
    2. Si el problema persiste, contacta al desarrollador
    """)
    
    # Mostrar información técnica en un expander
    with st.expander("🔍 Información técnica del error"):
        st.code(f"""
Error específico: {str(e)}
Dependencias instaladas: PyTorch ✅, Streamlit ✅, NumPy ✅
Problema: Acceso a libGL.so.1 para OpenCV
Estado: Configurando entorno del sistema...
        """)
    
    st.stop()

st.set_page_config(
    page_title="Sistema de Detección de Códigos Farmacéuticos",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

IMPORTS_OK = True
try:
    from components.sidebar import create_sidebar
    from components.results_display import ResultsDisplay
    from components.grid_editor import GridEditor
    from components.session_manager import SessionManager
    from core.orchestrator import Orchestrator
    from utils.file_utils import save_uploaded_file, cleanup_temp_files
    from components.cima_validator import CIMAValidator
except ImportError as e:
    st.error(f"Error importando módulos: {e}")
    IMPORTS_OK = False

def load_custom_css():
    """Aplica estilos CSS personalizados para la interfaz"""
    st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
    }
    
    .stButton > button:hover {
        background-color: #1B5E20;
    }
    
    .header-title {
        text-align: center;
        color: #2E7D32;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .header-subtitle {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .main .block-container {
        padding-bottom: 4.5rem;
        max-width: 95%;
    }
    
    .stImage > img {
        max-height: 70vh;
        width: 100%;
        object-fit: contain;
    }
    
    .floating-table {
        position: sticky;
        top: 80px;
        z-index: 1000;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        padding: 0.8rem;
        max-height: 85vh;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        font-size: 0.9rem;
    }
    
    /* 🆕 ESTILOS PARA COLORES DE DETECCIÓN */
    .detected-cell {
        background-color: #E8F5E8 !important;
        border: 2px solid #4CAF50 !important;
        color: #2E7D32 !important;
        font-weight: bold;
    }
    
    .not-detected-cell {
        background-color: #FFEBEE !important;
        border: 2px solid #F44336 !important;
        color: #C62828 !important;
    }
    
    .grid-header-with {
        background-color: #E3F2FD;
        color: #1976D2;
        font-weight: bold;
        text-align: center;
        padding: 8px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    .grid-header-without {
        background-color: #FFF3E0;
        color: #F57C00;
        font-weight: bold;
        text-align: center;
        padding: 8px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    .css-1d391kg {
        padding: 1rem 2rem;
    }
    
    @media (max-width: 768px) {
        .main .block-container {
            max-width: 100%;
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def add_file_management_controls(uploaded_files, session_manager):
    """Control mínimo y claro para gestión de archivos"""
    if uploaded_files:
        # Simple línea divisoria
        st.markdown("---")
        
        # Información y botón en la misma línea
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.info(f"📁 **{len(uploaded_files)} archivo(s) seleccionado(s)** - Listos para procesar")
        
        with col2:
            if st.button("🔄", 
                        help="Seleccionar otros archivos",
                        key="reset_files"):
                st.session_state.clear()
                st.rerun()



def show_system_status(session_manager):
    """Muestra estado del sistema en sidebar"""
    with st.sidebar.expander("🔧 Estado del Sistema", expanded=False):
        processed_count = len(session_manager.get_all_images())
        
        st.write(f"**Imágenes procesadas:** {processed_count}")
        
        if processed_count > 0:
            stats = session_manager.get_session_statistics()
            st.write(f"**Memoria aprox:** {stats.get('memory_usage_mb', 0):.1f} MB")
            st.write(f"**Tiempo sesión:** {stats.get('session_duration_minutes', 0):.1f} min")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Limpiar Temp", key="sidebar_cleanup_temp"):
                cleanup_temp_files()
                st.success("✅")
        
        with col2:
            if st.button("🔄 Reset", key="sidebar_reset_all"):
                session_manager.clear_all_images()
                st.success("✅")
                st.rerun()

def handle_processing_error(error, filename):
    """Maneja errores de procesamiento con mensajes específicos"""
    error_msg = str(error)
    
    if "No se detectó región de códigos" in error_msg:
        st.warning(f"⚠️ {filename}: La imagen no contiene códigos de barras detectables")
    elif "Imagen inválida" in error_msg:
        st.error(f"❌ {filename}: {error_msg}")
    elif "boxes referenced before assignment" in error_msg:
        st.error(f"❌ {filename}: Error en detección YOLO - imagen no procesable")
    else:
        st.error(f"❌ {filename}: Error inesperado - {error_msg}")
    
    return error_msg



def show_app_info():
    """Muestra información de la aplicación"""
    st.sidebar.markdown("---")
    st.sidebar.caption("💊 **FarmCode Detector v2.0**")
    st.sidebar.caption("Desarrollado para detección de códigos farmacéuticos")
    
    if st.sidebar.button("ℹ️ Info", help="Información de la aplicación"):
        st.sidebar.info("""
        **Componentes:**
        - YOLO v10 para detección
        - Gradientes para refinamiento
        - Expansión automática
        - Validación CIMA integrada
        """)

def main():
    """Función principal de la aplicación"""
    load_custom_css()
    
    st.markdown('<h1 class="header-title">Sistema de Detección de Códigos</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">Análisis Inteligente de Códigos Farmacéuticos</p>', unsafe_allow_html=True)
    
    if not IMPORTS_OK:
        st.error("No se pudieron cargar los módulos necesarios.")
        return
    
    session_manager = SessionManager()
    session_manager.initialize()
    
    config = create_sidebar()
    
    show_sidebar_history(session_manager)
    show_system_status(session_manager)
    show_app_info()
    
    tab1, tab2, tab3 = st.tabs(["📤 Subir Imágenes", "📊 Resultados", "🏥 Validación CIMA"])
    
    with tab1:
        upload_section(session_manager, config)
    
    with tab2:
        results_section(session_manager)
    
    with tab3:
        validation_section(session_manager)

def show_sidebar_history(session_manager):
    """Muestra historial de imágenes procesadas en sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Historial")
    
    processed_images = session_manager.get_all_images()
    
    if processed_images:
        st.sidebar.write(f"**{len(processed_images)} imágenes procesadas:**")
        
        for img_id, img_data in list(processed_images.items())[-5:]:
            with st.sidebar.container():
                col1, col2 = st.sidebar.columns([3, 1])
                
                with col1:
                    st.write(f"📄 {img_data['name'][:15]}...")
                    st.write(f"✅ {img_data['success_rate']*100:.0f}%")
                
                with col2:
                    if st.button("Ver", key=f"hist_{img_id}"):
                        session_manager.set_current_image(img_id)
                        st.rerun()
        
        if len(processed_images) > 5:
            st.sidebar.write(f"... y {len(processed_images) - 5} más")
    else:
        st.sidebar.info("Sin historial")

def upload_section(session_manager, config):
    """Sección de carga con limpieza automática de archivos"""
    st.header("Subir Imágenes de Cupones")
    
    st.markdown("""
    <div style="
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #f9f9f9;
        margin: 20px 0;
    ">
        <h4>📁 Arrastra imágenes aquí</h4>
        <p style="color: #666;">Hasta 10 archivos (máx. 10MB cada uno)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # USAR FORMULARIO PARA LIMPIEZA AUTOMÁTICA
    with st.form("upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Seleccionar imágenes",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Sube hasta 10 imágenes de cupones farmacéuticos"
        )
        
        # Botón de envío dentro del formulario
        submitted = st.form_submit_button(
            f"🚀 Procesar Archivos", 
            type="primary",
            use_container_width=True
        )
        
        if submitted and uploaded_files:
            if len(uploaded_files) > 10:
                st.error("❌ Demasiados archivos. Máximo 10 imágenes por lote.")
            else:
                process_batch_images(uploaded_files, session_manager, config)
    
    st.caption("⚠️ Los archivos se limpiarán automáticamente después del procesamiento")

def process_batch_images(uploaded_files, session_manager, config):
    """Procesa múltiples imágenes con configuración idéntica al notebook"""
    total_files = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    success_count = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        status_text.text(f"Procesando {uploaded_file.name} ({i+1}/{total_files})")
        
        try:
            temp_path = save_uploaded_file(uploaded_file)
            
            orchestrator = Orchestrator(
                yolo_model_path="../runs/detect/yolov10_train7/weights/best.pt",
                debug=True
            )
            
            results, error = orchestrator.process_image(temp_path)
            
            if not error:
                session_manager.add_processed_image(uploaded_file.name, results, config)
                success_count += 1
                st.success(f"✅ {uploaded_file.name} procesado correctamente")
            else:
                handle_processing_error(error, uploaded_file.name)
            
        except Exception as e:
            handle_processing_error(e, uploaded_file.name)
        
        time.sleep(0.1)
    
    cleanup_temp_files()
    progress_bar.progress(1.0)
    status_text.text(f"✅ Procesamiento completado: {success_count}/{total_files} exitosos")
    
    time.sleep(2)
    progress_bar.empty()
    status_text.empty()
    
    st.rerun()

def results_section(session_manager):
    """Sección de visualización de resultados"""
    processed_images = session_manager.get_all_images()
    
    if not processed_images:
        st.info("📭 Sube una imagen primero para ver los resultados")
        return
    
    st.header("Resultados de Detección")
    
    if len(processed_images) > 1:
        selected_id = session_manager.create_image_selector()
        st.markdown("---")
    
    current_results = session_manager.get_current_results()
    
    if current_results:
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            results_display = ResultsDisplay()
            results_display._show_barcode_region_corrected(current_results)
        
        with col2:
            st.markdown('<div class="floating-table">', unsafe_allow_html=True)
            
            grid_editor = GridEditor()
            grid_editor.create_simple_editor(current_results)
            
            st.markdown('</div>', unsafe_allow_html=True)

def validation_section(session_manager):
    """Sección de validación CIMA"""
    current_results = session_manager.get_current_results()
    
    if not current_results:
        st.info("📭 Procesa una imagen primero para validar códigos")
        return
    
    st.header("Validación CIMA")
    
    detected_codes = []
    for position, result in current_results['decoded_results'].items():
        if result['code'] != "No detectado" and result['code'] != "Código no encontrado":
            detected_codes.append({
                'position': position,
                'code': result['code'],
                'method': result['method']
            })
    
    if not detected_codes:
        st.warning("No hay códigos detectados para validar")
        return
    
    st.info(f"Se detectaron {len(detected_codes)} códigos para validar")
    
    st.subheader("Validación Individual")
    
    code_options = [f"Pos {item['position']}: {item['code']}" for item in detected_codes]
    
    selected_code = st.selectbox(
        "Selecciona un código para validar:",
        options=code_options,
        help="Selecciona un código para verificar en CIMA"
    )
    
    if selected_code:
        selected_position = int(selected_code.split(":")[0].replace("Pos ", ""))
        selected_code_value = selected_code.split(": ")[1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Código seleccionado:** {selected_code_value}")
            st.info(f"**Posición:** {selected_position}")
            
            if st.button("Verificar en CIMA", type="primary"):
                verify_single_code_with_cima(selected_code_value)
        
        with col2:
            st.markdown("### ℹ️ Información")
            st.info("""
            La validación CIMA verificará:
            - Si el código existe en la base de datos
            - Estado de comercialización
            - Problemas de suministro
            - Información del medicamento
            """)

def verify_single_code_with_cima(codigo_ean13):
    """Verifica código individual con CIMA"""
    try:
        if codigo_ean13.startswith(('840', '841', '842', '843', '844', '845', '846', '847', '848', '849')):
            codigo_nacional = codigo_ean13[6:12]
            
            with st.spinner(f"Verificando código nacional {codigo_nacional} en CIMA..."):
                
                try:
                    validator = CIMAValidator(rate_limit=1.0, debug=True)
                    result = validator.validar_medicamento(codigo_nacional)
                    
                    if result.get('valido'):
                        st.success("✅ Código nacional válido en CIMA")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Información del Medicamento")
                            st.write(f"**EAN-13 original:** {codigo_ean13}")
                            st.write(f"**Código Nacional:** {codigo_nacional}")
                            st.write(f"**Nombre:** {result.get('nombre', 'No disponible')}")
                            st.write(f"**Laboratorio:** {result.get('laboratorio', 'No disponible')}")
                            st.write(f"**Principio Activo:** {result.get('principio_activo', 'No disponible')}")
                        
                        with col2:
                            st.subheader("Estado del Medicamento")
                            
                            if result.get('autorizado'):
                                st.success("✅ Autorizado")
                            else:
                                st.error("❌ No autorizado")
                            
                            if result.get('problema_suministro'):
                                st.error("⚠️ Problemas de suministro")
                                if result.get('problema_info'):
                                    problema = result['problema_info']
                                    st.write(f"**Tipo:** {problema.get('tipo', 'No especificado')}")
                            else:
                                st.success("✅ Sin problemas de suministro")
                        
                        if result.get('ficha_tecnica'):
                            st.markdown(f"[📋 Ver Ficha Técnica Completa]({result['ficha_tecnica']})")
                        
                    else:
                        st.error("❌ Código nacional no encontrado en CIMA")
                        if result.get('error'):
                            st.write(f"**Error:** {result['error']}")
                
                except ImportError:
                    st.error("❌ CIMAValidator no disponible. Verifica la instalación.")
                except Exception as e:
                    st.error(f"❌ Error en validación CIMA: {str(e)}")
        else:
            st.warning("⚠️ Código no es español (no empieza por 84X)")
            
    except Exception as e:
        st.error(f"❌ Error en validación: {str(e)}")

if __name__ == "__main__":
    main()
