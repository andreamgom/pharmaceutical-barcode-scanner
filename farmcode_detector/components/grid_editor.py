# components/grid_editor.py
import streamlit as st
import pandas as pd
import time
import json
from typing import Dict, Any


class GridEditor:
    """Editor que se actualiza automáticamente al cambiar imagen y se ajusta al layout"""
    
    def __init__(self):
        self.session_key = "grid_data"
        self.selected_position_key = "selected_position"
        self.current_image_key = "current_image_for_grid" 
    
    def create_simple_editor(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Editor que se ajusta dinámicamente al layout (con/sin header)"""
        
        max_codes = results.get('max_codes', 26)
        
        if max_codes == 24:  # Termina en /24 = con header
            rows, cols = 6, 4
            header_detected = True
        elif max_codes == 26:  # Termina en /26 = sin header
            rows, cols = 7, 4
            header_detected = False
        else:
            # Fallback al método original
            rows, cols = results.get('grid_layout', (7, 4))
            header_detected = results.get('header_detected', False)
        
        # DETECTAR CAMBIO DE IMAGEN
        current_image_id = st.session_state.get('current_image_id', None)
        stored_image_id = st.session_state.get(self.current_image_key, None)
        
        # Si cambió la imagen, FORZAR recreación de grid_data
        if current_image_id != stored_image_id:
            st.session_state[self.session_key] = self._create_initial_grid_data(results, rows, cols, max_codes)
            st.session_state[self.current_image_key] = current_image_id
            st.session_state[self.selected_position_key] = 1
        
        # Si no existe grid_data, crearlo
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = self._create_initial_grid_data(results, rows, cols, max_codes)
        
        if self.selected_position_key not in st.session_state:
            st.session_state[self.selected_position_key] = 1
        
        if header_detected:
            st.markdown(
                '<div class="grid-header-with">📋 Grid 6×4 (con header)</div>', 
                unsafe_allow_html=True
            )
            st.caption("Formulario con encabezado - 24 posiciones máximas")
        else:
            st.markdown(
                '<div class="grid-header-without">📄 Grid 7×4 (sin header)</div>', 
                unsafe_allow_html=True
            )
            st.caption("Formulario sin encabezado - 26 posiciones máximas")
        
        table_height = min(400, (rows * 45) + 100)  # Más altura para colores
        
        edited_df = st.data_editor(
            st.session_state[self.session_key],
            use_container_width=True,
            num_rows="fixed",
            height=table_height,
            column_config={
                f'Col_{i+1}': st.column_config.TextColumn(
                    f"C{i+1}",
                    help=f"Códigos columna {i+1}",
                    max_chars=13,
                    validate="^[0-9-]*$"  
                ) for i in range(cols)
            },
            key=f"codes_editor_{current_image_id}_{rows}_{cols}"
        )
        
        st.session_state[self.session_key] = edited_df
        
        
        self._render_simple_controls(rows, cols, max_codes, results)
        
        return edited_df
    
    def _create_initial_grid_data(self, results: Dict[str, Any], rows: int, cols: int, max_codes: int) -> pd.DataFrame:
        """Crea datos iniciales con guiones y preparación para colores"""
        grid_data = []
        
        for row in range(rows):
            row_data = {}
            for col in range(cols):
                position = row * cols + col + 1
                
                if position <= max_codes:
                    if position in results.get('decoded_results', {}):
                        code = results['decoded_results'][position]['code']
                        if code == "No detectado":
                            row_data[f'Col_{col+1}'] = "-" 
                        else:
                            row_data[f'Col_{col+1}'] = code
                    else:
                        row_data[f'Col_{col+1}'] = "-"
                else:
                    row_data[f'Col_{col+1}'] = "-"
            
            grid_data.append(row_data)
        
        return pd.DataFrame(grid_data)
    
    
    def _render_simple_controls(self, rows: int, cols: int, max_codes: int, results: Dict[str, Any]):
        """Controles mejorados con información del layout detectado automáticamente"""
        
        st.markdown("---")
        
        # Información del layout actual
        header_detected = max_codes == 24
        layout_description = f"Layout detectado: {rows} filas × {cols} columnas"
        if header_detected:
            layout_description += " (con header - max_codes: 24)"
        else:
            layout_description += " (sin header - max_codes: 26)"
        
        st.caption(layout_description)
        
        # Selector de posición ajustado al layout real
        max_position = min(max_codes, rows * cols)
        
        selected_position = st.selectbox(
            "Seleccionar posición:",
            options=list(range(1, max_position + 1)),
            index=min(st.session_state[self.selected_position_key] - 1, max_position - 1),
            format_func=lambda x: f"Posición {x}",
            key=f"position_selector_{max_codes}"
        )
        st.session_state[self.selected_position_key] = selected_position
        
        # Información de posición
        row_num = ((selected_position - 1) // cols) + 1
        col_num = ((selected_position - 1) % cols) + 1
        
        # Verificar que la posición existe en el grid
        if row_num <= rows and col_num <= cols:
            current_code = st.session_state[self.session_key].iloc[row_num-1, col_num-1]
        else:
            current_code = "Fuera de rango"
        
        if current_code != "-" and current_code != "Fuera de rango":
            st.success(f"**Posición {selected_position}:** Fila {row_num}, Columna {col_num} - ✅ `{current_code}`")
        else:
            st.info(f"**Posición {selected_position}:** Fila {row_num}, Columna {col_num} - ❌ `{current_code}`")
        
        # Crear hueco
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⬅️ Hueco hacia atrás", use_container_width=True, key=f"gap_back_{max_codes}"):
                self._create_gap_backward(selected_position, rows, cols)
        
        with col2:
            if st.button("➡️ Hueco hacia adelante", use_container_width=True, key=f"gap_forward_{max_codes}"):
                self._create_gap_forward(selected_position, rows, cols)
        
        st.markdown("---")
        
        # ACCIONES CON DESCARGAS INTEGRADAS
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Aplicar Cambios", type="primary", use_container_width=True, key=f"apply_{max_codes}"):
                self.apply_changes(results)
        
        with col2:
            # DESCARGA JSON
            json_data = self._prepare_json_download(st.session_state[self.session_key], results)
            st.download_button(
                label="📄 JSON",
                data=json_data,
                file_name=f"codigos_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
                key=f"json_download_{max_codes}"
            )
        
        with col3:
            # DESCARGA CSV
            csv_data = self._prepare_csv_download(st.session_state[self.session_key])
            st.download_button(
                label="📊 CSV",
                data=csv_data,
                file_name=f"codigos_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"csv_download_{max_codes}"
            )
        
        # Reset
        if st.button("🔄 Resetear", use_container_width=True, key=f"reset_{max_codes}"):
            self.reset_grid_data(results, rows, cols, max_codes)
    
    def _create_gap_forward(self, position: int, rows: int, cols: int):
        """Crea hueco hacia adelante - 🆕 usando guiones"""
        row_idx = (position - 1) // cols
        col_idx = (position - 1) % cols
        
        values_to_shift = []
        
        for r in range(row_idx, rows):
            start_col = col_idx if r == row_idx else 0
            for c in range(start_col, cols):
                values_to_shift.append(st.session_state[self.session_key].iloc[r, c])
        
        values_to_shift.insert(0, "-")
        values_to_shift = values_to_shift[:-1]
        
        value_idx = 0
        for r in range(row_idx, rows):
            start_col = col_idx if r == row_idx else 0
            for c in range(start_col, cols):
                if value_idx < len(values_to_shift):
                    st.session_state[self.session_key].iloc[r, c] = values_to_shift[value_idx]
                    value_idx += 1
        
        st.success(f"✅ Hueco creado en posición {position} (hacia adelante)")
        st.rerun()
    
    def _create_gap_backward(self, position: int, rows: int, cols: int):
        """Crea hueco hacia atrás - 🆕 usando guiones"""
        row_idx = (position - 1) // cols
        col_idx = (position - 1) % cols
        
        values_to_shift = []
        
        for r in range(0, row_idx + 1):
            end_col = col_idx if r == row_idx else cols - 1
            for c in range(0, end_col + 1):
                values_to_shift.append(st.session_state[self.session_key].iloc[r, c])
        
        if values_to_shift:
            values_to_shift = values_to_shift[:-1]
            values_to_shift.append("-") 
        
        value_idx = 0
        for r in range(0, row_idx + 1):
            end_col = col_idx if r == row_idx else cols - 1
            for c in range(0, end_col + 1):
                if value_idx < len(values_to_shift):
                    st.session_state[self.session_key].iloc[r, c] = values_to_shift[value_idx]
                    value_idx += 1
        
        st.success(f"✅ Hueco creado en posición {position} (hacia atrás)")
        st.rerun()
    
    def apply_changes(self, original_results: Dict[str, Any]):
        """Aplica cambios considerando guiones como "No detectado" """
        # Detectar layout automáticamente
        max_codes = original_results.get('max_codes', 26)
        if max_codes == 24:
            rows, cols = 6, 4
        elif max_codes == 26:
            rows, cols = 7, 4
        else:
            rows, cols = original_results['grid_layout']
        
        updated_results = original_results.copy()
        
        for row_idx in range(rows):
            for col_idx in range(cols):
                position = row_idx * cols + col_idx + 1
                
                if position <= original_results['max_codes']:
                    new_code = st.session_state[self.session_key].iloc[row_idx, col_idx]
                    
                    if new_code and new_code.strip() and new_code != "-":
                        clean_code = new_code.strip()
                        if clean_code.isdigit() and len(clean_code) == 13:
                            updated_results['decoded_results'][position] = {
                                'code': clean_code,
                                'method': 'manual_edit',
                                'bbox': updated_results['decoded_results'].get(position, {}).get('bbox'),
                                'confidence': 1.0
                            }
                        else:
                            st.warning(f"⚠️ Código inválido en posición {position}: {clean_code}")
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
        
        # Actualizar en SessionManager si existe
        try:
            if hasattr(st.session_state, 'current_image_id') and st.session_state.current_image_id:
                from .session_manager import SessionManager
                session_manager = SessionManager()
                session_manager.update_image_results(st.session_state.current_image_id, updated_results)
        except:
            pass
        
        st.success("✅ Cambios aplicados correctamente!")
        st.rerun()
    
    def reset_grid_data(self, results: Dict[str, Any], rows: int, cols: int, max_codes: int):
        """Reset tabla a valores originales"""
        st.session_state[self.session_key] = self._create_initial_grid_data(results, rows, cols, max_codes)
        st.session_state[self.selected_position_key] = 1
        st.success("✅ Tabla reseteada a valores originales")
        st.rerun()
    
    def _prepare_json_download(self, df_grid: pd.DataFrame, results: Dict[str, Any]) -> str:
        """Prepara JSON considerando guiones"""
        codes_list = []
        for _, row in df_grid.iterrows():
            for col in df_grid.columns:
                code = row[col]
                if code and code.strip() and code != "-": 
                    codes_list.append(code.strip())
                else:
                    codes_list.append("No encontrado")
        
        # Detectar layout automáticamente
        max_codes = results.get('max_codes', 26)
        if max_codes == 24:
            grid_type = "6x4_con_header"
        elif max_codes == 26:
            grid_type = "7x4_sin_header"
        else:
            grid_type = "desconocido"
        
        data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'image_name': results.get('image_name'),
            'total_codes': len(codes_list),
            'valid_codes': len([c for c in codes_list if c != "No encontrado"]),
            'grid_layout': {
                'rows': len(df_grid),
                'cols': len(df_grid.columns),
                'type': grid_type,
                'max_codes': max_codes
            },
            'codes': codes_list,
            'processing_info': {
                'method': results.get('method'),
                'processing_time': results.get('processing_time', 0),
                'success_rate': results.get('success_rate', 0)
            }
        }
        
        return json.dumps(data, indent=2)
    
    def _prepare_csv_download(self, df_grid: pd.DataFrame) -> str:
        """Prepara CSV considerando guiones"""
        codes_list = []
        for _, row in df_grid.iterrows():
            for col in df_grid.columns:
                code = row[col]
                if code and code.strip() and code != "-":
                    codes_list.append(code.strip())
                else:
                    codes_list.append("No encontrado")
        
        df_download = pd.DataFrame({
            'Posicion': range(1, len(codes_list) + 1),
            'Codigo': codes_list
        })
        
        return df_download.to_csv(index=False)
    
    def get_grid_statistics(self) -> Dict[str, Any]:
        """Estadísticas considerando guiones"""
        if self.session_key not in st.session_state:
            return {}
        
        df_grid = st.session_state[self.session_key]
        total_positions = df_grid.size
        filled_positions = sum(1 for _, row in df_grid.iterrows() 
                              for col in df_grid.columns 
                              if row[col] and row[col].strip() and row[col] != "-")
        
        return {
            'total_positions': total_positions,
            'filled_positions': filled_positions,
            'empty_positions': total_positions - filled_positions,
            'fill_rate': filled_positions / total_positions if total_positions > 0 else 0,
            'selected_position': st.session_state.get(self.selected_position_key, 1)
        }
    
    def clear_session_data(self):
        """Limpia datos de sesión del grid editor"""
        keys_to_clear = [self.session_key, self.selected_position_key, self.current_image_key]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    def validate_grid_data(self, df_grid: pd.DataFrame) -> Dict[str, Any]:
        """Valida datos considerando guiones"""
        errors = []
        warnings = []
        
        for row_idx, row in df_grid.iterrows():
            for col_idx, value in enumerate(row):
                if value and value.strip() and value != "-":
                    code = value.strip()
                    position = row_idx * len(df_grid.columns) + col_idx + 1
                    
                    # Validar longitud
                    if not code.isdigit():
                        errors.append(f"Posición {position}: '{code}' contiene caracteres no numéricos")
                    elif len(code) < 8:
                        warnings.append(f"Posición {position}: '{code}' es muy corto (< 8 dígitos)")
                    elif len(code) > 13:
                        errors.append(f"Posición {position}: '{code}' es muy largo (> 13 dígitos)")
                    elif len(code) == 13:
                        # Validar EAN-13 si es posible
                        if not self._validate_ean13(code):
                            warnings.append(f"Posición {position}: '{code}' no es un EAN-13 válido")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'total_issues': len(errors) + len(warnings)
        }
    
    def _validate_ean13(self, code: str) -> bool:
        """Valida código EAN-13 usando dígito de control"""
        if len(code) != 13 or not code.isdigit():
            return False
        
        # Calcular dígito de control
        odd_sum = sum(int(code[i]) for i in range(0, 12, 2))
        even_sum = sum(int(code[i]) for i in range(1, 12, 2))
        
        total = odd_sum + (even_sum * 3)
        check_digit = (10 - (total % 10)) % 10
        
        return check_digit == int(code[12])
