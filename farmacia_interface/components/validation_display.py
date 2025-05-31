# components/validation_display.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict

class ValidationDisplay:
    """Componente para mostrar resultados de validación CIMA"""
    
    def __init__(self):
        self.colors = {
            'valid': '#2ECC71',
            'invalid': '#E74C3C', 
            'warning': '#F39C12',
            'info': '#3498DB',
            'neutral': '#95A5A6'
        }
    
    def show_validation_summary(self, validation_df):
        """Muestra resumen de validación CIMA"""
        
        if validation_df.empty:
            st.warning("No hay datos de validación para mostrar")
            return
        
        st.subheader("Resumen de Validación CIMA")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        total_codes = len(validation_df)
        valid_codes = validation_df['valido'].sum()
        commercialized = validation_df['comercializado'].sum() if 'comercializado' in validation_df.columns else 0
        supply_problems = validation_df['problema_suministro'].sum() if 'problema_suministro' in validation_df.columns else 0
        
        with col1:
            st.metric(
                "Total Códigos",
                total_codes,
                delta=f"{(valid_codes/total_codes*100):.1f}% válidos"
            )
        
        with col2:
            st.metric(
                "Códigos Válidos",
                valid_codes,
                delta=f"{total_codes - valid_codes} inválidos"
            )
        
        with col3:
            st.metric(
                "Comercializados",
                commercialized,
                delta=f"{(commercialized/valid_codes*100):.1f}%" if valid_codes > 0 else "0%"
            )
        
        with col4:
            st.metric(
                "Problemas Suministro",
                supply_problems,
                delta="⚠️ Requieren atención" if supply_problems > 0 else "✅ Sin problemas"
            )
    
    def show_validation_table(self, validation_df):
        """Muestra tabla detallada de validación"""
        
        st.subheader("Detalle de Validación")
        
        if validation_df.empty:
            st.info("No hay datos para mostrar")
            return
        
        # Preparar datos para mostrar
        display_columns = ['codigo', 'valido', 'nombre', 'comercializado', 'laboratorio']
        
        # Añadir columnas opcionales si existen
        optional_columns = ['problema_suministro', 'principio_activo', 'autorizado']
        for col in optional_columns:
            if col in validation_df.columns:
                display_columns.append(col)
        
        # Filtrar columnas que existen
        available_columns = [col for col in display_columns if col in validation_df.columns]
        df_display = validation_df[available_columns].copy()
        
        # Aplicar estilos
        def style_validation_table(row):
            if not row['valido']:
                return ['background-color: #ffebee'] * len(row)
            elif row.get('problema_suministro', False):
                return ['background-color: #fff3e0'] * len(row)
            elif row.get('comercializado', True):
                return ['background-color: #e8f5e8'] * len(row)
            else:
                return ['background-color: #f5f5f5'] * len(row)
        
        styled_df = df_display.style.apply(style_validation_table, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Opciones de filtrado
        with st.expander("Filtros"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                show_valid_only = st.checkbox("Solo códigos válidos")
            with col2:
                show_commercialized_only = st.checkbox("Solo comercializados")
            with col3:
                show_problems_only = st.checkbox("Solo con problemas")
            
            if show_valid_only or show_commercialized_only or show_problems_only:
                filtered_df = validation_df.copy()
                
                if show_valid_only:
                    filtered_df = filtered_df[filtered_df['valido'] == True]
                if show_commercialized_only and 'comercializado' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['comercializado'] == True]
                if show_problems_only and 'problema_suministro' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['problema_suministro'] == True]
                
                st.write(f"Resultados filtrados: {len(filtered_df)} de {len(validation_df)}")
                if not filtered_df.empty:
                    st.dataframe(filtered_df[available_columns], use_container_width=True)
    
    def show_validation_charts(self, validation_df):
        """Muestra gráficas de validación"""
        
        if validation_df.empty:
            return
        
        st.subheader("Análisis Visual")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfica de validez
            valid_counts = validation_df['valido'].value_counts()
            
            fig_valid = go.Figure(data=[
                go.Pie(
                    labels=['Válidos', 'Inválidos'],
                    values=[valid_counts.get(True, 0), valid_counts.get(False, 0)],
                    hole=0.3,
                    marker_colors=[self.colors['valid'], self.colors['invalid']]
                )
            ])
            fig_valid.update_layout(
                title="Distribución de Validez",
                height=400
            )
            st.plotly_chart(fig_valid, use_container_width=True)
        
        with col2:
            # Gráfica de comercialización
            if 'comercializado' in validation_df.columns:
                comm_counts = validation_df['comercializado'].value_counts()
                
                fig_comm = go.Figure(data=[
                    go.Pie(
                        labels=['Comercializados', 'No Comercializados'],
                        values=[comm_counts.get(True, 0), comm_counts.get(False, 0)],
                        hole=0.3,
                        marker_colors=[self.colors['info'], self.colors['neutral']]
                    )
                ])
                fig_comm.update_layout(
                    title="Estado de Comercialización",
                    height=400
                )
                st.plotly_chart(fig_comm, use_container_width=True)
        
        # Gráfica de laboratorios
        if 'laboratorio' in validation_df.columns:
            st.subheader("Distribución por Laboratorio")
            
            lab_counts = validation_df['laboratorio'].value_counts().head(10)
            
            if not lab_counts.empty:
                fig_lab = px.bar(
                    x=lab_counts.values,
                    y=lab_counts.index,
                    orientation='h',
                    title="Top 10 Laboratorios",
                    labels={'x': 'Cantidad de Códigos', 'y': 'Laboratorio'}
                )
                fig_lab.update_layout(height=400)
                st.plotly_chart(fig_lab, use_container_width=True)
    
    def show_supply_problems(self, validation_df):
        """Muestra análisis de problemas de suministro"""
        
        if 'problema_suministro' not in validation_df.columns:
            return
        
        problems_df = validation_df[validation_df['problema_suministro'] == True]
        
        if problems_df.empty:
            st.success("No se detectaron problemas de suministro")
            return
        
        st.subheader("Problemas de Suministro Detectados")
        st.warning(f"Se encontraron {len(problems_df)} códigos con problemas de suministro")
        
        # Mostrar códigos problemáticos
        problem_columns = ['codigo', 'nombre', 'laboratorio']
        if 'problema_info' in problems_df.columns:
            problem_columns.append('problema_info')
        
        available_problem_columns = [col for col in problem_columns if col in problems_df.columns]
        
        st.dataframe(
            problems_df[available_problem_columns],
            use_container_width=True
        )
        
        # Análisis de tipos de problemas
        if 'problema_info' in problems_df.columns:
            problem_types = {}
            for _, row in problems_df.iterrows():
                if row['problema_info'] and isinstance(row['problema_info'], dict):
                    tipo = row['problema_info'].get('tipo', 'No especificado')
                    problem_types[tipo] = problem_types.get(tipo, 0) + 1
            
            if problem_types:
                st.subheader("Tipos de Problemas")
                
                fig_problems = px.bar(
                    x=list(problem_types.keys()),
                    y=list(problem_types.values()),
                    title="Distribución de Tipos de Problemas",
                    labels={'x': 'Tipo de Problema', 'y': 'Cantidad'}
                )
                st.plotly_chart(fig_problems, use_container_width=True)
    
    def show_download_validation_options(self, validation_df):
        """Opciones de descarga para validación"""
        
        if validation_df.empty:
            return
        
        st.subheader("Descargar Resultados de Validación")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV completo
            csv_data = validation_df.to_csv(index=False)
            st.download_button(
                label="Descargar CSV Completo",
                data=csv_data,
                file_name=f"validacion_cima_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Solo códigos válidos
            valid_df = validation_df[validation_df['valido'] == True]
            if not valid_df.empty:
                valid_csv = valid_df.to_csv(index=False)
                st.download_button(
                    label="Descargar Solo Válidos",
                    data=valid_csv,
                    file_name=f"codigos_validos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Códigos con problemas
            if 'problema_suministro' in validation_df.columns:
                problems_df = validation_df[validation_df['problema_suministro'] == True]
                if not problems_df.empty:
                    problems_csv = problems_df.to_csv(index=False)
                    st.download_button(
                        label="Descargar Problemas",
                        data=problems_csv,
                        file_name=f"problemas_suministro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

    def validar_medicamento_individual(self, codigo: str) -> Dict:
        """Validación individual con más detalles para interfaz"""
        
        resultado_base = self.validar_medicamento(codigo)
        
        if resultado_base.get('valido'):
            # Añadir información extra para la interfaz
            try:
                # Intentar obtener más información si está disponible
                codigo_clean = str(codigo).strip()
                
                # Información adicional de presentación
                data_extra = self._make_request("medicamento", {"cn": codigo_clean})
                
                if data_extra and not data_extra.get('error'):
                    resultado_base.update({
                        'presentacion': data_extra.get('presentacion', 'No disponible'),
                        'via_administracion': data_extra.get('viaAdministracion', 'No disponible'),
                        'dosis': data_extra.get('dosis', 'No disponible'),
                        'excipientes': data_extra.get('excipientes', 'No disponible'),
                        'condiciones_prescripcion': data_extra.get('condicionesPrescripcion', 'No disponible')
                    })
            
            except Exception as e:
                if self.debug:
                    print(f"Error obteniendo información extra: {e}")
        
        return resultado_base
