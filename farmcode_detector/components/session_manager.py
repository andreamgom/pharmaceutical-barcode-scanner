# components/session_manager.py
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import uuid
import cv2
import numpy as np
import time
from datetime import datetime


class SessionManager:
    """Maneja estado de sesión para múltiples imágenes procesadas"""
    
    def __init__(self):
        self.images_key = "processed_images"
        self.current_key = "current_image_id"
        self.counter_key = "image_counter"
        self.settings_key = "app_settings"
        
    def initialize(self):
        """Inicializa estado de sesión con valores por defecto"""
        
        # Diccionario de imágenes procesadas
        if self.images_key not in st.session_state:
            st.session_state[self.images_key] = {}
        
        # ID de imagen actual
        if self.current_key not in st.session_state:
            st.session_state[self.current_key] = None
            
        # Contador para IDs únicos
        if self.counter_key not in st.session_state:
            st.session_state[self.counter_key] = 0
            
        # Configuraciones de la app
        if self.settings_key not in st.session_state:
            st.session_state[self.settings_key] = {
                'auto_switch_to_results': True,
                'show_processing_details': True,
                'auto_cleanup_old_images': True
            }
    
    def add_processed_image(self, image_name, results, config):
        """Añade imagen procesada"""
        
        # PRESERVAR LA IMAGEN RECORTADA
        barcode_region = None
        if 'barcode_region' in results and results['barcode_region'] is not None:
            # Convertir array numpy a formato compatible con Streamlit
            import cv2
            img_array = results['barcode_region']
            if isinstance(img_array, np.ndarray):
                # Convertir BGR a RGB para Streamlit
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    barcode_region = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                else:
                    barcode_region = img_array
            else:
                barcode_region = img_array
        
        # Crear datos procesados completos
        processed_data = {
            'id': f"img_{len(st.session_state.processed_images) + 1}",
            'name': image_name,
            'timestamp': time.time(),
            'config': config,
            'valid_codes': results.get('valid_codes', 0),
            'max_codes': results.get('max_codes', 0),
            'success_rate': results.get('success_rate', 0),
            'decoded_results': results.get('decoded_results', {}),
            'barcode_region': barcode_region,
            'processing_time': results.get('processing_time', 0),
            'expansion_stats': results.get('expansion_stats', {}),
            'merge_stats': results.get('merge_stats', {})
        }
        
        # Guardar en session state
        image_id = processed_data['id']
        st.session_state.processed_images[image_id] = processed_data
        st.session_state.current_image_id = image_id



    def get_current_results(self) -> Optional[Dict]:
        """Obtiene los resultados de la imagen actual con acceso directo"""
        current_id = self.get_current_image_id()
        if current_id:
            images = self.get_all_images()
            image_data = images.get(current_id, {})
            
            # Devolver los resultados con acceso directo a las propiedades
            if 'results' in image_data:
                results = image_data['results'].copy()
                results.update({
                    'valid_codes': image_data.get('valid_codes', 0),
                    'max_codes': image_data.get('max_codes', 0),
                    'success_rate': image_data.get('success_rate', 0),
                    'processing_time': image_data.get('processing_time', 0),
                    'image_name': image_data.get('name')
                })
                return results
            else:
                return image_data
        return None

    
    def get_all_images(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene todas las imágenes procesadas"""
        return st.session_state.get(self.images_key, {})
    
    def get_current_image_id(self) -> Optional[str]:
        """Obtiene ID de la imagen actual"""
        return st.session_state.get(self.current_key)

    
    def get_current_image_info(self) -> Optional[Dict[str, Any]]:
        """Obtiene información completa de la imagen actual"""
        current_id = self.get_current_image_id()
        if current_id and current_id in st.session_state.get(self.images_key, {}):
            return st.session_state[self.images_key][current_id]
        return None
    
    def set_current_image(self, image_id: str) -> bool:
        """
        Establece imagen actual
        
        Args:
            image_id: ID de la imagen a establecer como actual
            
        Returns:
            True si se estableció correctamente, False si no existe
        """
        if image_id in st.session_state.get(self.images_key, {}):
            st.session_state[self.current_key] = image_id
            return True
        return False
    
    def remove_image(self, image_id: str) -> bool:
        """
        Elimina imagen del estado de sesión
        
        Args:
            image_id: ID de la imagen a eliminar
            
        Returns:
            True si se eliminó correctamente
        """
        if image_id in st.session_state.get(self.images_key, {}):
            del st.session_state[self.images_key][image_id]
            
            # Si era la imagen actual, cambiar a otra
            if st.session_state.get(self.current_key) == image_id:
                remaining_images = list(st.session_state[self.images_key].keys())
                st.session_state[self.current_key] = remaining_images[0] if remaining_images else None
            
            return True
        return False
    
    def clear_all_images(self):
        """Limpia todas las imágenes procesadas"""
        st.session_state[self.images_key] = {}
        st.session_state[self.current_key] = None
        st.session_state[self.counter_key] = 0
    
    def create_image_selector(self) -> Optional[str]:
        """Crea selector de imágenes si hay múltiples - CORREGIDO"""
        images = self.get_all_images()
        
        if len(images) <= 1:
            return None
        
        st.markdown("### 📁 Imágenes Procesadas")
        
        # Crear opciones para selectbox
        options = {}
        for img_id, img_data in images.items():
            try:
                # CORREGIDO: Manejo robusto de timestamp
                if 'timestamp' in img_data and hasattr(img_data['timestamp'], 'strftime'):
                    # Es un objeto datetime
                    timestamp = img_data['timestamp'].strftime("%H:%M:%S")
                elif 'processed_at' in img_data:
                    # Es un string ISO, convertir a datetime
                    from datetime import datetime
                    timestamp_dt = datetime.fromisoformat(img_data['processed_at'].replace('Z', '+00:00'))
                    timestamp = timestamp_dt.strftime("%H:%M:%S")
                else:
                    # Fallback si no hay timestamp
                    timestamp = "Unknown"
                
                success_info = f"{img_data['valid_codes']}/{img_data['max_codes']} códigos"
                display_name = f"{img_data['name']} ({timestamp}) - {success_info}"
                options[display_name] = img_id
                
            except Exception as e:
                # Fallback en caso de error
                print(f"Error procesando timestamp para {img_id}: {e}")
                display_name = f"{img_data['name']} - {img_data['valid_codes']}/{img_data['max_codes']} códigos"
                options[display_name] = img_id
        
        # Obtener índice actual
        current_id = self.get_current_image_id()
        current_index = 0
        if current_id:
            try:
                current_index = list(options.values()).index(current_id)
            except ValueError:
                current_index = 0
        
        # Selector
        selected_display = st.selectbox(
            "Seleccionar imagen:",
            options=list(options.keys()),
            index=current_index,
            help="Cambia entre las imágenes procesadas"
        )
        
        selected_id = options[selected_display]
        self.set_current_image(selected_id)
        
        # Mostrar información rápida de la imagen seleccionada
        self._show_quick_image_info(selected_id)
        
        return selected_id

    def _show_quick_image_info(self, image_id: str):
        """Muestra información rápida de la imagen seleccionada - CORREGIDO"""
        if image_id not in st.session_state.get(self.images_key, {}):
            return
        
        img_data = st.session_state[self.images_key][image_id]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Códigos", f"{img_data['valid_codes']}/{img_data['max_codes']}")
        
        with col2:
            st.metric("Éxito", f"{img_data['success_rate']*100:.1f}%")
        
        with col3:
            st.metric("Tiempo", f"{img_data['processing_time']:.1f}s")
        
        with col4:
            # Botón para eliminar
            if st.button("🗑️", key=f"delete_{image_id}", help="Eliminar esta imagen"):
                self.remove_image(image_id)
                st.rerun()

    
    def create_images_summary(self) -> Dict[str, Any]:
        """Crea resumen de todas las imágenes procesadas"""
        images = self.get_all_images()
        
        if not images:
            return {}
        
        total_images = len(images)
        total_codes = sum(img['valid_codes'] for img in images.values())
        total_positions = sum(img['max_codes'] for img in images.values())
        avg_success_rate = sum(img['success_rate'] for img in images.values()) / total_images
        total_processing_time = sum(img['processing_time'] for img in images.values())
        
        # Métodos más usados
        methods = [img['method'] for img in images.values()]
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        most_used_method = max(method_counts, key=method_counts.get) if method_counts else "Unknown"
        
        return {
            'total_images': total_images,
            'total_codes_detected': total_codes,
            'total_positions': total_positions,
            'average_success_rate': avg_success_rate,
            'total_processing_time': total_processing_time,
            'most_used_method': most_used_method,
            'method_distribution': method_counts,
            'images_by_success': {
                'high_success': len([img for img in images.values() if img['success_rate'] >= 0.8]),
                'medium_success': len([img for img in images.values() if 0.5 <= img['success_rate'] < 0.8]),
                'low_success': len([img for img in images.values() if img['success_rate'] < 0.5])
            }
        }
    
    def show_batch_summary(self):
        """Muestra resumen de procesamiento por lotes"""
        summary = self.create_images_summary()
        
        if not summary:
            st.info("No hay imágenes procesadas para mostrar resumen")
            return
        
        st.markdown("### 📊 Resumen del Lote")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Imágenes Procesadas",
                summary['total_images'],
                delta=f"{summary['total_codes_detected']} códigos totales"
            )
        
        with col2:
            st.metric(
                "Tasa Promedio",
                f"{summary['average_success_rate']*100:.1f}%",
                delta=f"{summary['total_positions']} posiciones totales"
            )
        
        with col3:
            st.metric(
                "Tiempo Total",
                f"{summary['total_processing_time']:.1f}s",
                delta=f"{summary['total_processing_time']/summary['total_images']:.1f}s promedio"
            )
        
        with col4:
            st.metric(
                "Método Principal",
                summary['most_used_method'].split('(')[0][:15] + "..." if len(summary['most_used_method']) > 15 else summary['most_used_method'],
                delta=f"{summary['method_distribution'][summary['most_used_method']]} usos"
            )
        
        # Distribución por éxito
        success_dist = summary['images_by_success']
        if success_dist['high_success'] + success_dist['medium_success'] + success_dist['low_success'] > 0:
            st.markdown("**Distribución por Éxito:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success(f"Alto éxito (≥80%): {success_dist['high_success']}")
            with col2:
                st.warning(f"Éxito medio (50-80%): {success_dist['medium_success']}")
            with col3:
                st.error(f"Bajo éxito (<50%): {success_dist['low_success']}")
    
    def export_batch_results(self) -> str:
        """Exporta resultados de todas las imágenes en formato JSON"""
        images = self.get_all_images()
        summary = self.create_images_summary()
        
        export_data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'total_images': len(images),
                'app_version': 'FarmaScan v2.0'
            },
            'summary': summary,
            'images': {}
        }
        
        # Añadir datos de cada imagen (sin las imágenes en sí)
        for img_id, img_data in images.items():
            export_data['images'][img_id] = {
                'name': img_data['name'],
                'timestamp': img_data['timestamp'].isoformat(),
                'processing_time': img_data['processing_time'],
                'valid_codes': img_data['valid_codes'],
                'max_codes': img_data['max_codes'],
                'success_rate': img_data['success_rate'],
                'method': img_data['method'],
                'config': img_data.get('config', {}),
                'decoded_results': img_data['results'].get('decoded_results', {})
            }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la sesión actual"""
        images = self.get_all_images()
        if not images:
            return {'session_active': False}

        # Calcular tiempo de sesión - CORREGIDO
        timestamps = [img['timestamp'] for img in images.values()]
        session_start_timestamp = min(timestamps)
        
        # CONVERTIR float timestamp a datetime
        session_start = datetime.fromtimestamp(session_start_timestamp)
        session_duration = datetime.now() - session_start

        return {
            'session_active': True,
            'session_start': session_start,
            'session_duration_minutes': session_duration.total_seconds() / 60,
            'images_processed': len(images),
            'current_image': self.get_current_image_id(),
            'memory_usage_mb': len(str(images)) / (1024 * 1024),  # Aproximado
            'settings': st.session_state.get(self.settings_key, {})
        }

    
    def cleanup_old_images(self, max_images: int = 10):
        """Limpia imágenes antiguas si hay demasiadas"""
        images = self.get_all_images()
        
        if len(images) <= max_images:
            return
        
        # Ordenar por timestamp y mantener solo las más recientes
        sorted_images = sorted(
            images.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        # Mantener solo las más recientes
        to_keep = dict(sorted_images[:max_images])
        
        # Actualizar estado
        st.session_state[self.images_key] = to_keep
        
        # Verificar que la imagen actual sigue existiendo
        current_id = self.get_current_image_id()
        if current_id and current_id not in to_keep:
            st.session_state[self.current_key] = list(to_keep.keys())[0] if to_keep else None
    
    def update_image_results(self, image_id: str, updated_results: Dict[str, Any]):
        """Actualiza los resultados de una imagen específica"""
        if image_id in st.session_state.get(self.images_key, {}):
            st.session_state[self.images_key][image_id]['results'] = updated_results
            
            # Actualizar métricas derivadas
            st.session_state[self.images_key][image_id]['valid_codes'] = updated_results.get('valid_codes', 0)
            st.session_state[self.images_key][image_id]['success_rate'] = updated_results.get('success_rate', 0)
            
            return True
        return False
    
    def migrate_old_data(self):
        """Migra datos antiguos al nuevo formato"""
        images = self.get_all_images()
        
        for img_id, img_data in images.items():
            # Si existe processed_at pero no timestamp
            if 'processed_at' in img_data and 'timestamp' not in img_data:
                try:
                    # Convertir ISO string a datetime
                    timestamp_dt = datetime.fromisoformat(img_data['processed_at'].replace('Z', '+00:00'))
                    img_data['timestamp'] = timestamp_dt
                    st.session_state[self.images_key][img_id] = img_data
                except:
                    # Fallback: usar tiempo actual
                    img_data['timestamp'] = datetime.now()
                    st.session_state[self.images_key][img_id] = img_data
