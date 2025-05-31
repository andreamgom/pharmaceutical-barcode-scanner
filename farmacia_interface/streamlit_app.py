# streamlit_app.py - VERSIÓN COMPLETA CON TODAS LAS MEJORAS
import streamlit as st


# Configuración de página
st.set_page_config(
    page_title="Sistema de Detección de Códigos Farmacéuticos",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)


import pandas as pd
from pathlib import Path
import time
from PIL import Image
import io
import cv2
import numpy as np

IMPORTS_OK = True
try:
    from core.hybrid_detector import HybridDetector
    from core.cima_validator import CIMAValidator
    from components.file_uploader import CustomFileUploader
    from components.results_display import ResultsDisplay
    from components.sidebar import create_sidebar
    from utils.file_utils import save_uploaded_file, cleanup_temp_files

except ImportError as e:
    st.error(f"Error importando módulos: {e}")
    IMPORTS_OK = False




# CSS básico integrado (SIN archivo externo)
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
    
    /* TABLA FLOTANTE MEJORADA */
    .floating-table {
        position: sticky;
        top: 80px;
        z-index: 1000;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        padding: 1rem;
        max-height: 80vh;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
    }
    
    /* SCROLL SUAVE */
    .floating-table::-webkit-scrollbar {
        width: 8px;
    }
    
    .floating-table::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    .floating-table::-webkit-scrollbar-thumb {
        background: #2E7D32;
        border-radius: 4px;
    }
    
    .floating-table::-webkit-scrollbar-thumb:hover {
        background: #1B5E20;
    }
    
    /* RESPONSIVE */
    @media (max-width: 768px) {
        .floating-table {
            position: relative;
            top: 0;
            max-height: none;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def create_simple_sidebar():
    """Sidebar simplificado que funciona"""
    st.sidebar.title("Configuración")
    
    # Modelo YOLO fijo
    yolo_model = "../runs/detect/yolov10_train7/weights/best.pt"
    
    # Verificar si existe
    model_path = Path(yolo_model)
    if model_path.exists():
        st.sidebar.success("✅ Modelo YOLO: yolov10_train7")
        size_mb = model_path.stat().st_size / (1024 * 1024)
        st.sidebar.info(f"Tamaño: {size_mb:.1f} MB")
    else:
        st.sidebar.error("❌ Modelo YOLO: No encontrado")
        st.sidebar.code(f"Buscando: {yolo_model}")
        yolo_model = "yolov8n.pt"  # Fallback
        st.sidebar.warning(f"Usando modelo por defecto: {yolo_model}")
    
    # Configuración mínima
    validate_with_cima = st.sidebar.checkbox("Validar con CIMA", value=True)
    
    # Estado del sistema
    with st.sidebar.expander("Estado del Sistema"):
        st.write("Detector: Híbrido")
        st.write("Gradientes: Habilitado")
        st.write(f"Modelo: {'Disponible' if model_path.exists() else 'Por defecto'}")
    
    return {
        'yolo_model': yolo_model,
        'use_gradient': True,
        'validate_with_cima': validate_with_cima,
        'debug_mode': True
    }

def main():
    load_custom_css()
    
    # Header principal
    st.markdown('<h1 class="header-title">Sistema de Detección de Códigos</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">Análisis Inteligente de Códigos Farmacéuticos</p>', unsafe_allow_html=True)
    
    if not IMPORTS_OK:
        st.error("No se pudieron cargar los módulos necesarios. Verifica la instalación.")
        return
    
    # Sidebar simplificado
    config = create_simple_sidebar()
    
    # Área principal
    tab1, tab2, tab3 = st.tabs(["Subir Imagen", "Resultados", "Validación CIMA"])
    
    with tab1:
        upload_section(config)
    
    with tab2:
        if 'detection_results' in st.session_state:
            # FORZAR COLAPSO DEL SIDEBAR EN RESULTADOS
            st.markdown("""
            <style>
            [data-testid="stSidebar"] {
                display: none !important;
            }
            .main .block-container {
                padding-left: 1rem !important;
                max-width: none !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            results_section()
        else:
            st.info("Sube una imagen primero para ver los resultados")
    
    with tab3:
        if 'detection_results' in st.session_state:
            validation_section()
        else:
            st.info("Procesa una imagen primero para validar códigos")


def upload_section(config):
    """Sección de subida de archivos"""
    st.header("Subir Imagen de Cupones")
    
    # Uploader personalizado
    uploader = CustomFileUploader()
    uploaded_file = uploader.create_uploader()
    
    if uploaded_file is not None:
        # Mostrar imagen
        col1, col2 = st.columns([2, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", use_column_width=True)
        
        with col2:
            st.markdown("### Información")
            st.write(f"**Nombre:** {uploaded_file.name}")
            st.write(f"**Tamaño:** {len(uploaded_file.getvalue())} bytes")
            st.write(f"**Formato:** {image.format}")
            st.write(f"**Dimensiones:** {image.size}")
        
        # Botón de procesamiento
        if st.button("Procesar Imagen", type="primary", use_container_width=True):
            process_image(uploaded_file, config)

def process_image(uploaded_file, config):
    """Procesa la imagen con el detector híbrido"""
    
    with st.spinner("Procesando imagen..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Guardar archivo temporal
            temp_path = save_uploaded_file(uploaded_file)
            progress_bar.progress(20)
            status_text.text("Archivo guardado...")
            
            # Ruta corregida del modelo (FUERA de farmacia_interface)
            yolo_model_path = config.get('yolo_model')
            if not Path(yolo_model_path).exists():
                # Intentar ruta relativa desde farmacia_interface
                yolo_model_path = f"../{yolo_model_path}"
                if not Path(yolo_model_path).exists():
                    st.error(f"Modelo YOLO no encontrado: {yolo_model_path}")
                    return
            
            # Inicializar detector híbrido CON DEBUG ACTIVADO
            detector = HybridDetector(
                yolo_model_path=yolo_model_path,
                use_gradient=True,  # Siempre True
                debug=True  # ACTIVAR DEBUG PARA VER QUE PASA CON HEADER
            )
            progress_bar.progress(40)
            status_text.text("Detector inicializado...")
            
            # Procesar imagen
            results, error = detector.process_image(temp_path)
            progress_bar.progress(80)
            status_text.text("Análisis completado...")
            
            if error:
                st.error(f"Error al procesar: {error}")
                return
            
            # Guardar resultados en session state
            st.session_state['detection_results'] = results
            st.session_state['processed_image_path'] = temp_path
            
            progress_bar.progress(100)
            status_text.text("Procesamiento completado!")
            
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"Error al procesar: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

def results_section():
    """Resultados con tabla flotante que SÍ se desplaza"""
    results = st.session_state['detection_results']
    
    st.header("Resultados de Detección")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Códigos Detectados", results.get('valid_codes', 0))
    with col2:
        st.metric("Total Posiciones", results.get('max_codes', 0))
    with col3:
        st.metric("Tasa de Éxito", f"{results.get('success_rate', 0)*100:.1f}%")
    with col4:
        header_status = "Sí" if results.get('header_detected', False) else "No"
        st.metric("Header Detectado", header_status)
    
    # LAYOUT CON TABLA FLOTANTE FUNCIONAL
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Región de Códigos")
        
        # SOLO imagen recortada limpia
        if 'barcode_region' in results and results['barcode_region'] is not None:
            st.image(results['barcode_region'], caption="Zona de códigos", use_column_width=True)
        elif 'original_image' in results:
            st.image(results['original_image'], caption="Imagen original", use_column_width=True)
        else:
            st.info("Imagen no disponible")
        
        # Información del grid
        st.write("**Información:**")
        rows, cols = results.get('grid_layout', (7, 4))
        st.write(f"Layout: {rows} filas × {cols} columnas")
        st.write(f"Header: {'Detectado' if results.get('header_detected', False) else 'No detectado'}")
        st.write(f"Método: {results.get('method', 'Híbrido')}")
        
        # ESPACIADOR PARA PERMITIR SCROLL LARGO
        st.markdown("<br>" * 20, unsafe_allow_html=True)
        st.write("**Información adicional:**")
        st.write("Puedes hacer scroll para ver cómo la tabla se mantiene visible.")
        st.markdown("<br>" * 20, unsafe_allow_html=True)
    
    with col2:
        # CONTENEDOR FLOTANTE
        st.markdown('<div class="floating-table">', unsafe_allow_html=True)
        st.subheader("Tabla de Códigos")
        
        # Tabla editable dentro del contenedor flotante
        display_editable_codes_grid_floating(results)
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_editable_codes_grid_floating(results):
    """Tabla editable optimizada para contenedor flotante"""
    
    rows, cols = results.get('grid_layout', (7, 4))
    max_codes = results.get('max_codes', 26)
    
    # Inicializar grid_data en session_state si no existe
    if 'grid_data' not in st.session_state:
        st.session_state['grid_data'] = create_initial_grid_data(results, rows, cols, max_codes)
    
    st.write(f"**Grid {rows}×{cols}**")
    
    # Editor de datos compacto
    edited_df = st.data_editor(
        st.session_state['grid_data'],
        use_container_width=True,
        num_rows="fixed",
        height=300,  # Altura fija para el contenedor flotante
        column_config={
            f'Col_{i+1}': st.column_config.TextColumn(
                f"C{i+1}",  # Nombres más cortos
                help=f"Códigos columna {i+1}",
                max_chars=13,
                validate="^[0-9]*$"
            ) for i in range(cols)
        },
        key="codes_editor_floating"
    )
    
    # Actualizar session_state
    st.session_state['grid_data'] = edited_df
    
    # Controles compactos
    st.write("**Controles:**")
    
    # Fila de controles compacta
    col1, col2 = st.columns(2)
    
    with col1:
        position_for_gap = st.number_input(
            "Pos:",
            min_value=1,
            max_value=max_codes,
            value=1,
            key="gap_pos_floating"
        )
    
    with col2:
        if st.button("🗑️ Hueco", key="gap_floating"):
            create_gap_fixed(position_for_gap, rows, cols)
    
    # Botones de acción
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Aplicar", type="primary", key="apply_floating"):
            apply_grid_changes_fixed(results)
    
    with col2:
        if st.button("🔄 Reset", key="reset_floating"):
            reset_grid_data(results, rows, cols, max_codes)
    
    # Descarga compacta
    csv_data = prepare_grid_for_download(st.session_state['grid_data'])
    st.download_button(
        label="📥 CSV",
        data=csv_data,
        file_name=f"codigos_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="download_floating"
    )


def display_editable_codes_grid(results):
    """Tabla editable simplificada solo con crear hueco"""
    
    rows, cols = results['grid_layout']
    max_codes = results['max_codes']
    
    # Inicializar grid_data en session_state si no existe
    if 'grid_data' not in st.session_state:
        st.session_state['grid_data'] = create_initial_grid_data(results, rows, cols, max_codes)
    
    st.write(f"**Grid {rows}×{cols}** - Edita los códigos:")
    
    # SOLO TABLA Y CONTROLES BÁSICOS
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Editor de datos usando session_state
        edited_df = st.data_editor(
            st.session_state['grid_data'],
            use_container_width=True,
            num_rows="fixed",
            column_config={
                f'Col_{i+1}': st.column_config.TextColumn(
                    f"Columna {i+1}",
                    help=f"Códigos de la columna {i+1}",
                    max_chars=13,
                    validate="^[0-9]*$"
                ) for i in range(cols)
            },
            key="codes_editor_main"
        )
        
        # Actualizar session_state con cambios
        st.session_state['grid_data'] = edited_df
    
    with col2:
        st.write("**Controles:**")
        
        # Solo selector de posición para crear hueco
        position_for_gap = st.number_input(
            "Posición:",
            min_value=1,
            max_value=max_codes,
            value=1,
            key="gap_position_selector"
        )
        
        # Solo botón de crear hueco
        if st.button("🗑️ Crear Hueco", key="create_gap"):
            create_gap_fixed(position_for_gap, rows, cols)
        
        st.write("---")
        
        if st.button("💾 Aplicar", type="primary"):
            apply_grid_changes_fixed(results)
        
        if st.button("🔄 Resetear"):
            reset_grid_data(results, rows, cols, max_codes)
        
        # Descarga
        csv_data = prepare_grid_for_download(st.session_state['grid_data'])
        st.download_button(
            label="📥 CSV",
            data=csv_data,
            file_name=f"codigos_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def prepare_grid_for_download(df_grid):
    """Prepara el grid para descarga en CSV"""
    
    # Convertir grid a lista plana
    codes_list = []
    for _, row in df_grid.iterrows():
        for col in df_grid.columns:
            code = row[col]
            if code and code.strip() and code != "Código no encontrado":
                codes_list.append(code.strip())
            else:
                codes_list.append("Código no encontrado")
    
    # Crear DataFrame para descarga
    df_download = pd.DataFrame({
        'Posicion': range(1, len(codes_list) + 1),
        'Codigo': codes_list
    })
    
    return df_download.to_csv(index=False)


def create_initial_grid_data(results, rows, cols, max_codes):
    """Crea datos iniciales del grid"""
    grid_data = []
    
    for row in range(rows):
        row_data = {}
        for col in range(cols):
            position = row * cols + col + 1
            
            if position <= max_codes and position in results['decoded_results']:
                code = results['decoded_results'][position]['code']
                if code == "No detectado":
                    row_data[f'Col_{col+1}'] = "Código no encontrado"
                else:
                    row_data[f'Col_{col+1}'] = code
            else:
                row_data[f'Col_{col+1}'] = "Código no encontrado"
        
        grid_data.append(row_data)
    
    return pd.DataFrame(grid_data)


def create_gap_fixed(position, rows, cols):
    """Crea hueco y desplaza códigos"""
    row_idx = (position - 1) // cols
    col_idx = (position - 1) % cols
    
    # Obtener todos los valores desde la posición hacia adelante
    values_to_shift = []
    
    for r in range(row_idx, rows):
        start_col = col_idx if r == row_idx else 0
        for c in range(start_col, cols):
            values_to_shift.append(st.session_state['grid_data'].iloc[r, c])
    
    # Insertar "Código no encontrado" al principio
    values_to_shift.insert(0, "Código no encontrado")
    values_to_shift = values_to_shift[:-1]  # Eliminar último
    
    # Redistribuir valores
    value_idx = 0
    for r in range(row_idx, rows):
        start_col = col_idx if r == row_idx else 0
        for c in range(start_col, cols):
            if value_idx < len(values_to_shift):
                st.session_state['grid_data'].iloc[r, c] = values_to_shift[value_idx]
                value_idx += 1
    
    st.success(f"Hueco creado en posición {position}")
    st.rerun()

def reset_grid_data(results, rows, cols, max_codes):
    """Resetea la tabla a valores originales"""
    st.session_state['grid_data'] = create_initial_grid_data(results, rows, cols, max_codes)
    st.success("Tabla reseteada a valores originales")
    st.rerun()

def apply_grid_changes_fixed(original_results):
    """Aplica cambios del grid editado"""
    rows, cols = original_results['grid_layout']
    updated_results = original_results.copy()
    
    # Actualizar desde session_state
    for row_idx in range(rows):
        for col_idx in range(cols):
            position = row_idx * cols + col_idx + 1
            
            if position <= original_results['max_codes']:
                new_code = st.session_state['grid_data'].iloc[row_idx, col_idx]
                
                if new_code and new_code.strip() and new_code != "Código no encontrado":
                    clean_code = new_code.strip()
                    if clean_code.isdigit() and len(clean_code) == 13:
                        updated_results['decoded_results'][position] = {
                            'code': clean_code,
                            'method': 'manual_edit',
                            'bbox': updated_results['decoded_results'].get(position, {}).get('bbox'),
                            'confidence': 1.0
                        }
                    else:
                        st.warning(f"Código inválido en posición {position}: {clean_code}")
                else:
                    updated_results['decoded_results'][position] = {
                        'code': "No detectado",
                        'method': "none",
                        'bbox': updated_results['decoded_results'].get(position, {}).get('bbox'),
                        'confidence': 0.0
                    }
    
    # Recalcular estadísticas
    valid_codes = sum(1 for r in updated_results['decoded_results'].values() 
                     if r['code'] != "No detectado")
    updated_results['valid_codes'] = valid_codes
    updated_results['success_rate'] = valid_codes / updated_results['max_codes']
    
    # Actualizar session state
    st.session_state['detection_results'] = updated_results
    
    st.success("Cambios aplicados correctamente!")
    st.rerun()

def validation_section():
    """Validación CIMA con código nacional EAN-13 CORREGIDO"""
    st.header("Validación Individual de Códigos")
    
    results = st.session_state['detection_results']
    
    # Extraer códigos válidos EAN-13
    valid_codes = []
    for position, data in results['decoded_results'].items():
        code = data['code']
        if code not in ["No detectado", "Código no encontrado"]:
            # Verificar que sea EAN-13 válido
            if code.isdigit() and len(code) == 13:
                # Extraer código nacional CORRECTO: posición 6-12 (ambos incluidos)
                if code.startswith(('840', '841', '842', '843', '844', '845', '846', '847', '848', '849')):
                    # Código nacional son los dígitos 6-12 (7 dígitos)
                    codigo_nacional = code[6:12]  # Posición 6-12 (índices 5-11)
                    valid_codes.append({
                        'ean13': code,
                        'codigo_nacional': codigo_nacional,
                        'posicion': position,
                        'metodo': data['method']
                    })
    
    if not valid_codes:
        st.warning("No hay códigos EAN-13 españoles válidos para validar")
        return
    
    st.info(f"Se detectaron {len(valid_codes)} códigos EAN-13 españoles válidos")
    
    # Selectbox para elegir código
    st.subheader("Seleccionar Código para Verificar")
    
    codigo_options = []
    for item in valid_codes:
        codigo_options.append(f"Pos {item['posicion']}: {item['ean13']} (CN: {item['codigo_nacional']})")
    
    selected_option = st.selectbox(
        "Elige un código para verificar en CIMA:",
        options=codigo_options,
        index=None,
        placeholder="Selecciona un código...",
        help="Se usará el código nacional (7 dígitos: posición 6-12) para consultar CIMA"
    )
    
    if selected_option:
        # Extraer información
        selected_ean13 = selected_option.split(": ")[1].split(" (CN:")[0]
        selected_cn = selected_option.split("CN: ")[1].split(")")[0]
        selected_position = selected_option.split("Pos ")[1].split(":")[0]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.info(f"**EAN-13 completo:** {selected_ean13}")
            st.info(f"**Código Nacional (pos 6-12):** {selected_cn}")
            st.info(f"**Posición:** {selected_position}")
            
            if st.button("Verificar en CIMA", type="primary"):
                verify_single_code_with_cima(selected_cn, selected_ean13)
        
        with col2:
            st.markdown("### ⚠️ Aviso Legal")
            st.warning("""
            **Uso solo para verificación individual**
            
            Se consulta el código nacional (7 dígitos: posición 6-12) en CIMA.
            Sistema NO comercializable según términos CIMA.
            """)


def verify_single_code_with_cima(codigo_nacional, ean13_completo):
    """Verifica código nacional (7 dígitos) con CIMA"""
    
    validator = CIMAValidator(rate_limit=1.0, debug=True)
    
    with st.spinner(f"Verificando código nacional {codigo_nacional} en CIMA..."):
        # Usar código nacional para CIMA
        result = validator.validar_medicamento(codigo_nacional)
    
    # Mostrar resultado
    if result.get('valido'):
        st.success("✅ Código nacional válido en CIMA")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Información del Medicamento")
            st.write(f"**EAN-13 original:** {ean13_completo}")
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
            st.markdown(f"[📋 Ficha Técnica]({result['ficha_tecnica']})")
    
    else:
        st.error("❌ Código nacional no encontrado en CIMA")
        if result.get('error'):
            st.write(f"**Error:** {result['error']}")
    
    # Guardar en session state para referencia
    if 'individual_validations' not in st.session_state:
        st.session_state['individual_validations'] = []
    
    st.session_state['individual_validations'].append({
        'codigo': codigo_nacional,
        'resultado': result,
        'timestamp': result.get('timestamp_consulta', 'No disponible')
    })
    
    # Mostrar historial de consultas
    show_validation_history()

def show_validation_history():
    """Muestra historial de validaciones individuales"""
    
    if 'individual_validations' not in st.session_state or not st.session_state['individual_validations']:
        return
    
    st.subheader("Historial de Verificaciones")
    
    # Crear tabla del historial
    history_data = []
    for validation in st.session_state['individual_validations']:
        history_data.append({
            'Código': validation['codigo'],
            'Válido': '✅' if validation['resultado'].get('valido') else '❌',
            'Nombre': validation['resultado'].get('nombre', 'No disponible')[:50] + '...' if validation['resultado'].get('nombre', '') else 'N/A',
            'Timestamp': validation['timestamp']
        })
    
    df_history = pd.DataFrame(history_data)
    st.dataframe(df_history, use_container_width=True)
    
    # Botón para limpiar historial
    if st.button("🗑️ Limpiar Historial"):
        st.session_state['individual_validations'] = []
        st.rerun()

if __name__ == "__main__":
    main()
