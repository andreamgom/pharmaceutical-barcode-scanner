import streamlit as st
import cv2
import numpy as np
from PIL import Image

class ResultsDisplay:
    """Componente mejorado para mostrar resultados"""
    
    def __init__(self):
        pass
    
    def show_detection_results(self, results_data):
        """Muestra resultados con imagen corregida"""
        
        if not results_data:
            st.info("📭 No hay resultados para mostrar")
            return
        
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            self._show_barcode_region_corrected(results_data)
        
        with col2:
            self._show_detection_stats(results_data)
    
    def _show_barcode_region_corrected(self, results_data):
        """Muestra la región recortada CORREGIDA"""
        st.subheader("🎯 Región de Códigos Detectada")
        
        barcode_image = None
        image_source = "No encontrada"
        
        # Prioridad 1: barcode_region directa
        if 'barcode_region' in results_data and results_data['barcode_region'] is not None:
            barcode_image = results_data['barcode_region']
            image_source = "barcode_region"
        
        # Prioridad 2: Recrear desde original + crop_info
        elif ('original_image' in results_data and 
              'crop_info' in results_data and 
              results_data['crop_info']):
            barcode_image = self._recreate_barcode_region(results_data)
            image_source = "recreada desde crop_info"
        
        # Prioridad 3: processed_image como fallback
        elif 'processed_image' in results_data:
            barcode_image = results_data['processed_image']
            image_source = "processed_image (fallback)"
        
        if barcode_image is not None:
            try:
                # CONVERSIÓN CORRECTA DE FORMATO
                display_image = self._convert_image_for_display(barcode_image)
                
                st.image(
                    display_image,
                    caption=f"Zona recortada por YOLO - Fuente: {image_source}",
                    use_container_width=True
                )
                
                # INFORMACIÓN DEL RECORTE
                with st.expander("📋 Información del recorte", expanded=False):
                    self._show_crop_info(results_data, barcode_image)
                
            except Exception as e:
                st.error(f"❌ Error mostrando imagen: {str(e)}")
                self._show_debug_info(results_data, barcode_image)
        else:
            st.warning("⚠️ No se pudo obtener la imagen recortada")
            self._show_debug_info(results_data, None)
    
    def _recreate_barcode_region(self, results_data):
        """Recrea la región recortada desde imagen original y crop_info"""
        try:
            original_image = results_data['original_image']
            crop_info = results_data['crop_info']
            
            # Obtener coordenadas del recorte
            crop_bbox = crop_info.get('crop_bbox')
            if crop_bbox:
                x1, y1, x2, y2 = crop_bbox
                
                # Asegurar que las coordenadas están dentro de los límites
                h, w = original_image.shape[:2]
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(x1, min(x2, w))
                y2 = max(y1, min(y2, h))
                
                # Extraer región
                cropped = original_image[y1:y2, x1:x2]
                
                if cropped.size > 0:
                    return cropped
            
            return None
            
        except Exception as e:
            st.error(f"Error recreando región: {e}")
            return None
    
    def _convert_image_for_display(self, image):
        """Convierte imagen a formato correcto para Streamlit"""
        if isinstance(image, np.ndarray):
            # Si es array numpy
            if len(image.shape) == 3:
                # Imagen en color - convertir BGR a RGB
                if image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
            else:
                # Imagen en escala de grises
                image_rgb = image
            
            return Image.fromarray(image_rgb)
        
        elif isinstance(image, Image.Image):
            return image
        
        else:
            return Image.fromarray(np.array(image))
    
    def _show_crop_info(self, results_data, barcode_image):
        """Muestra información detallada del recorte"""
        crop_info = results_data.get('crop_info', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Coordenadas:**")
            st.write(f"Offset X: {crop_info.get('offset_x', 'N/A')}")
            st.write(f"Offset Y: {crop_info.get('offset_y', 'N/A')}")
            
            if 'crop_bbox' in crop_info:
                bbox = crop_info['crop_bbox']
                st.write(f"Bbox: {bbox}")
        
        with col2:
            st.write("**Dimensiones:**")
            if isinstance(barcode_image, np.ndarray):
                h, w = barcode_image.shape[:2]
                st.write(f"Alto: {h}px")
                st.write(f"Ancho: {w}px")
                st.write(f"Ratio: {w/h:.2f}")
            
            st.write(f"Layout: {crop_info.get('layout_type', 'N/A')}")
    
    def _show_debug_info(self, results_data, barcode_image):
        """Muestra información de debug"""
        st.write("**🔍 Información de Debug:**")
        st.write(f"- Claves en results_data: {list(results_data.keys())}")
        
        if barcode_image is not None:
            st.write(f"- Tipo de imagen: {type(barcode_image)}")
            if hasattr(barcode_image, 'shape'):
                st.write(f"- Shape: {barcode_image.shape}")
        
        # Verificar crop_info
        if 'crop_info' in results_data:
            crop_info = results_data['crop_info']
            st.write(f"- Claves en crop_info: {list(crop_info.keys())}")
    
    def _show_detection_stats(self, results_data):
        """Muestra estadísticas de detección"""
        st.subheader("📊 Estadísticas")
        
        if 'decoded_results' in results_data:
            decoded_results = results_data['decoded_results']
            valid_codes = sum(1 for r in decoded_results.values() 
                            if r.get('code', '') not in ['No detectado', ''])
            total_codes = len(decoded_results)
            
            st.metric("Códigos Válidos", f"{valid_codes}/{total_codes}")
            st.metric("Tasa de Éxito", f"{valid_codes/total_codes*100:.1f}%" if total_codes > 0 else "0%")
            
            # Estadísticas de expansión
            if 'decoding_stats' in results_data:
                decoding_stats = results_data['decoding_stats']
                expansion_successes = decoding_stats.get('expansion_successes', 0)
                if expansion_successes > 0:
                    st.metric("Códigos por Expansión", expansion_successes)
            
            # Información del header
            header_detected = results_data.get('header_detected', False)
            if header_detected:
                st.success("✅ Header detectado")
            else:
                st.info("ℹ️ Sin header")
