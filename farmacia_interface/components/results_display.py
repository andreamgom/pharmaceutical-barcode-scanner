import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

class ResultsDisplay:
    """Clase para mostrar resultados de forma profesional y limpia"""
    
    def __init__(self):
        self.colors = {
            'success': '#2ECC71',
            'warning': '#F39C12', 
            'error': '#E74C3C',
            'info': '#3498DB',
            'neutral': '#95A5A6'
        }
    
    def show_detection_metrics(self, results):
        """Muestra métricas principales de detección"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Códigos Detectados",
                value=results['valid_codes'],
                delta=f"{results['success_rate']*100:.1f}% éxito"
            )
        
        with col2:
            st.metric(
                label="Total Posiciones", 
                value=results['max_codes'],
                delta=f"{results['max_codes'] - results['valid_codes']} faltantes"
            )
        
        with col3:
            header_status = "Detectado" if results['header_detected'] else "No detectado"
            st.metric(
                label="Header",
                value=header_status,
                delta=f"Layout: {results['grid_layout'][0]}x{results['grid_layout'][1]}"
            )
        
        with col4:
            processing_time = results.get('processing_time', 0)
            st.metric(
                label="Tiempo Procesamiento",
                value=f"{processing_time:.2f}s",
                delta=f"Método: {results.get('method', 'Desconocido')}"
            )
    
    def show_annotated_image(self, results):
        """Muestra imagen anotada con detecciones"""
        
        st.subheader("Imagen Procesada")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if 'annotated_image' in results:
                st.image(results['annotated_image'], 
                        caption="Detecciones marcadas", 
                        use_column_width=True)
            else:
                st.warning("Imagen anotada no disponible")
        
        with col2:
            st.markdown("### Leyenda")
            st.markdown("""
            - **Verde**: Códigos detectados
            - **Rojo**: Posiciones sin código
            - **Magenta**: Header detectado
            - **Cyan**: Región de códigos
            - **Números**: Posición en grid
            """)
            
            # Información adicional
            if results.get('hybrid_info'):
                st.markdown("### Información Híbrida")
                hybrid_info = results['hybrid_info']
                
                if hybrid_info.get('yolo_header_used'):
                    st.success("Header detectado por YOLO")
                if hybrid_info.get('gradient_ordering_used'):
                    st.success("Ordenamiento por Gradientes")
                if hybrid_info.get('barcode_region_cropped'):
                    st.info("Región recortada automáticamente")
    
    def show_codes_table(self, results):
        """Muestra tabla detallada de códigos detectados"""
        
        st.subheader("Códigos Detectados")
        
        # Preparar datos para la tabla
        table_data = []
        decoded_results = results.get('decoded_results', {})
        
        for position in range(1, results['max_codes'] + 1):
            if position in decoded_results:
                result = decoded_results[position]
                table_data.append({
                    'Posición': position,
                    'Código': result['code'],
                    'Método': result['method'],
                    'Confianza': f"{result['confidence']:.3f}",
                    'Estado': self._get_code_status(result['code'])
                })
            else:
                table_data.append({
                    'Posición': position,
                    'Código': "No detectado",
                    'Método': "none",
                    'Confianza': "0.000",
                    'Estado': "Sin detección"
                })
        
        df = pd.DataFrame(table_data)
        
        # Aplicar colores según estado
        def color_rows(row):
            if row['Estado'] == 'Detectado':
                return ['background-color: #d4edda'] * len(row)
            elif row['Estado'] == 'Sin detección':
                return ['background-color: #f8d7da'] * len(row)
            else:
                return ['background-color: #fff3cd'] * len(row)
        
        styled_df = df.style.apply(color_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Estadísticas rápidas
        detected_count = len([r for r in table_data if r['Estado'] == 'Detectado'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Códigos Válidos", detected_count)
        with col2:
            st.metric("Sin Detectar", results['max_codes'] - detected_count)
        with col3:
            success_rate = (detected_count / results['max_codes']) * 100
            st.metric("Tasa de Éxito", f"{success_rate:.1f}%")
    
    def show_decoding_statistics(self, results):
        """Muestra estadísticas de métodos de decodificación"""
        
        st.subheader("Estadísticas de Decodificación")
        
        decoding_stats = results.get('decoding_stats', {})
        
        if not decoding_stats:
            st.warning("No hay estadísticas de decodificación disponibles")
            return
        
        # Filtrar estadísticas con valores > 0
        filtered_stats = {k: v for k, v in decoding_stats.items() if v > 0}
        
        if not filtered_stats:
            st.info("No se detectaron códigos con ningún método")
            return
        
        # Crear gráfico
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico de barras
            fig, ax = plt.subplots(figsize=(10, 6))
            
            methods = list(filtered_stats.keys())
            counts = list(filtered_stats.values())
            
            bars = ax.bar(methods, counts, color=self.colors['info'], alpha=0.7)
            
            # Añadir valores en las barras
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontweight='bold')
            
            ax.set_title('Métodos de Decodificación Utilizados')
            ax.set_xlabel('Método')
            ax.set_ylabel('Cantidad de Códigos')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig)
        
        with col2:
            st.markdown("### Resumen")
            
            total_detected = sum(filtered_stats.values())
            st.metric("Total Detectados", total_detected)
            
            # Método más exitoso
            if filtered_stats:
                best_method = max(filtered_stats, key=filtered_stats.get)
                st.metric("Método Principal", best_method)
                st.metric("Códigos con Método Principal", filtered_stats[best_method])
    
    def show_download_options(self, results):
        """Muestra opciones de descarga"""
        
        st.subheader("Descargar Resultados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV de códigos
            csv_data = self._prepare_csv_data(results)
            st.download_button(
                label="Descargar CSV",
                data=csv_data,
                file_name=f"codigos_{results['image_name'].replace('.jpg', '')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON simple
            json_simple = self._prepare_simple_json(results)
            st.download_button(
                label="Descargar JSON Simple",
                data=json_simple,
                file_name=f"codigos_{results['image_name'].replace('.jpg', '')}_simple.json",
                mime="application/json"
            )
        
        with col3:
            # JSON completo
            json_complete = self._prepare_complete_json(results)
            st.download_button(
                label="Descargar JSON Completo",
                data=json_complete,
                file_name=f"codigos_{results['image_name'].replace('.jpg', '')}_complete.json",
                mime="application/json"
            )
    
    def _get_code_status(self, code):
        """Determina el estado de un código"""
        if code == "No detectado":
            return "Sin detección"
        elif code == "Código no encontrado":
            return "Error decodificación"
        else:
            return "Detectado"
    
    def _prepare_csv_data(self, results):
        """Prepara datos en formato CSV"""
        table_data = []
        decoded_results = results.get('decoded_results', {})
        
        for position in range(1, results['max_codes'] + 1):
            if position in decoded_results:
                result = decoded_results[position]
                table_data.append({
                    'Posicion': position,
                    'Codigo': result['code'],
                    'Metodo': result['method'],
                    'Confianza': result['confidence']
                })
        
        df = pd.DataFrame(table_data)
        return df.to_csv(index=False)
    
    def _prepare_simple_json(self, results):
        """Prepara JSON simple (solo lista de códigos)"""
        codes_list = []
        decoded_results = results.get('decoded_results', {})
        
        for position in range(1, results['max_codes'] + 1):
            if position in decoded_results:
                code = decoded_results[position]['code']
                codes_list.append(code if code != "No detectado" else "Código no encontrado")
            else:
                codes_list.append("Código no encontrado")
        
        return json.dumps(codes_list, indent=2)
    
    def _prepare_complete_json(self, results):
        """Prepara JSON completo con toda la información"""
        json_data = {
            'image_name': results['image_name'],
            'method': results.get('method', 'Desconocido'),
            'processing_time': results.get('processing_time', 0),
            'header_detected': results['header_detected'],
            'grid_layout': {
                'rows': results['grid_layout'][0],
                'cols': results['grid_layout'][1]
            },
            'max_codes': results['max_codes'],
            'valid_codes': results['valid_codes'],
            'success_rate': results['success_rate'],
            'codes': {}
        }
        
        decoded_results = results.get('decoded_results', {})
        for position in range(1, results['max_codes'] + 1):
            if position in decoded_results:
                json_data['codes'][str(position)] = decoded_results[position]
            else:
                json_data['codes'][str(position)] = {
                    'code': "No detectado",
                    'method': "none",
                    'confidence': 0.0
                }
        
        return json.dumps(json_data, indent=2)
