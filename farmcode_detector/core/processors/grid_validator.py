# core/processors/grid_validator.py
class GridValidator:
    """Validador de coherencia de grid farmacéutico"""
    
    def __init__(self, debug=False):
        self.debug = debug
    
    def validate_grid_data(self, decoded_results, grid_shape):
        """Valida la coherencia del grid decodificado"""
        
        errors = []
        warnings = []
        
        rows, cols = grid_shape
        
        # Validar estructura básica
        expected_positions = self._get_expected_positions(rows, cols)
        
        for position in expected_positions:
            if position not in decoded_results:
                errors.append(f"Posición {position} faltante en resultados")
                continue
            
            result = decoded_results[position]
            code = result.get('code', 'No detectado')
            
            # Validar códigos EAN-13
            if code != "No detectado" and code != "Código no encontrado":
                if not self._validate_ean13_format(code):
                    errors.append(f"Posición {position}: código '{code}' no es EAN-13 válido")
                elif not self._validate_spanish_prefix(code):
                    warnings.append(f"Posición {position}: código '{code}' no parece español")
        
        # Validar restricciones de layout farmacéutico
        layout_errors = self._validate_layout_constraints(decoded_results, grid_shape)
        errors.extend(layout_errors)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'total_issues': len(errors) + len(warnings)
        }
    
    def _get_expected_positions(self, rows, cols):
        """Obtiene posiciones esperadas según el layout"""
        if rows == 6:  # Con header
            return set(range(1, 25))  # 1-24
        elif rows == 7:  # Sin header
            return set(range(1, 25)) | {27, 28}  # 1-24 + 27,28
        else:
            return set(range(1, rows * cols + 1))
    
    def _validate_ean13_format(self, code):
        """Valida formato EAN-13"""
        if not code or len(code) != 13 or not code.isdigit():
            return False
        
        # Validar dígito de control
        try:
            odd_sum = sum(int(code[i]) for i in range(0, 12, 2))
            even_sum = sum(int(code[i]) for i in range(1, 12, 2))
            total = odd_sum + (even_sum * 3)
            check_digit = (10 - (total % 10)) % 10
            return check_digit == int(code[12])
        except:
            return False
    
    def _validate_spanish_prefix(self, code):
        """Valida si el código tiene prefijo español (84x)"""
        return code.startswith(('840', '841', '842', '843', '844', '845', '846', '847', '848', '849'))
    
    def _validate_layout_constraints(self, decoded_results, grid_shape):
        """Valida restricciones específicas del layout farmacéutico"""
        errors = []
        rows, cols = grid_shape
        
        # Contar códigos por fila
        for row_idx in range(rows):
            codes_in_row = 0
            for col_idx in range(cols):
                position = row_idx * cols + col_idx + 1
                if position in decoded_results:
                    code = decoded_results[position].get('code', 'No detectado')
                    if code != "No detectado":
                        codes_in_row += 1
            
            # Validar restricciones por fila
            if row_idx < 6 and codes_in_row > 4:
                errors.append(f"Fila {row_idx + 1}: demasiados códigos ({codes_in_row}/4 max)")
            elif row_idx == 6 and codes_in_row > 2:
                errors.append(f"Fila 7: demasiados códigos ({codes_in_row}/2 max)")
        
        return errors
