# core/barcode_decoder.py
import cv2
import numpy as np
from pyzbar import pyzbar
import zxingcpp
from collections import Counter

class BarcodeDecoder:
    """Decodificador robusto de códigos de barras con múltiples técnicas"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.preprocessor = None  # Se inyectará desde hybrid_detector
    
    def set_preprocessor(self, preprocessor):
        """Inyecta el preprocessor"""
        self.preprocessor = preprocessor
    
    def decode_with_preprocessing(self, roi_gray):
        """Decodifica usando múltiples técnicas de preprocesamiento"""
        if not self.preprocessor:
            return {}
            
        preprocessing_techniques = self.preprocessor.get_all_preprocessing_techniques(roi_gray)
        results = {}

        for technique_name, processed_image in preprocessing_techniques:
            # Intentar con pyzbar
            pyzbar_result = self._try_pyzbar(processed_image, technique_name)
            
            # Intentar con zxing-cpp
            zxingcpp_result = self._try_zxingcpp(processed_image, technique_name)

            # Guardar resultados si hay éxito
            if pyzbar_result or zxingcpp_result:
                results[technique_name] = {
                    'pyzbar': pyzbar_result,
                    'zxingcpp': zxingcpp_result,
                    'processed_image': processed_image
                }

        return results
    
    def _try_pyzbar(self, processed_image, technique_name):
        """Intenta decodificar con pyzbar"""
        try:
            decoded_objects = pyzbar.decode(processed_image, symbols=[
                pyzbar.ZBarSymbol.EAN13,
                pyzbar.ZBarSymbol.CODE128,
                pyzbar.ZBarSymbol.UPCA,
                pyzbar.ZBarSymbol.EAN8
            ])
            
            if decoded_objects:
                data = decoded_objects[0].data.decode('utf-8')
                if data.isdigit() and len(data) >= 8:
                    return {
                        'data': data,
                        'type': decoded_objects[0].type,
                        'technique': technique_name,
                        'confidence': 1.0  # pyzbar no da confianza
                    }
        except Exception as e:
            if self.debug:
                print(f"Error pyzbar con {technique_name}: {e}")
        
        return None
    
    def _try_zxingcpp(self, processed_image, technique_name):
        """Intenta decodificar con zxingcpp"""
        try:
            zxing_results = zxingcpp.read_barcodes(processed_image)
            if zxing_results:
                result = zxing_results[0]
                if result.text.isdigit() and len(result.text) >= 8:
                    return {
                        'data': result.text,
                        'type': result.format.name,
                        'technique': technique_name,
                        'confidence': getattr(result, 'confidence', 1.0)
                    }
        except Exception as e:
            if self.debug:
                print(f"Error zxingcpp con {technique_name}: {e}")
        
        return None
    
    def decode_barcode_hybrid(self, roi):
        """Método principal de decodificación híbrida"""
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi.copy()
        
        # Usar todas las técnicas de preprocesamiento
        preprocessing_results = self.decode_with_preprocessing(roi_gray)
        
        if preprocessing_results:
            # Seleccionar el mejor resultado
            return self._select_best_result(preprocessing_results)
        
        return "No detectado", "none"
    
    def _select_best_result(self, preprocessing_results):
        """Selecciona el mejor resultado de todas las técnicas"""
        all_codes = []
        
        # Recopilar todos los códigos detectados
        for technique_name, results in preprocessing_results.items():
            if results['pyzbar']:
                code = results['pyzbar']['data']
                method = f"pyzbar_{technique_name}"
                all_codes.append((code, method, len(code)))
            
            if results['zxingcpp']:
                code = results['zxingcpp']['data']
                method = f"zxingcpp_{technique_name}"
                all_codes.append((code, method, len(code)))
        
        if not all_codes:
            return "No detectado", "none"
        
        # Estrategia de selección mejorada
        return self._apply_selection_strategy(all_codes)
    
    def _apply_selection_strategy(self, all_codes):
        """Aplica estrategia inteligente para seleccionar el mejor código"""
        
        # Estrategia 1: Priorizar códigos de 13 dígitos (EAN-13)
        thirteen_digit_codes = [code for code in all_codes if code[2] == 13]
        
        if thirteen_digit_codes:
            # Si hay múltiples códigos de 13 dígitos, tomar el más común
            code_counts = Counter([code[0] for code in thirteen_digit_codes])
            most_common_code = code_counts.most_common(1)[0][0]
            
            # Encontrar el método del código más común
            best_result = next(code for code in thirteen_digit_codes if code[0] == most_common_code)
            return best_result[0], best_result[1]
        
        # Estrategia 2: Si no hay códigos de 13 dígitos, tomar el más largo
        if all_codes:
            best_result = max(all_codes, key=lambda x: x[2])
            return best_result[0], best_result[1]
        
        return "No detectado", "none"
    
    def validate_ean13(self, code):
        """Valida un código EAN-13 usando el dígito de control"""
        if len(code) != 13 or not code.isdigit():
            return False
        
        # Calcular dígito de control
        odd_sum = sum(int(code[i]) for i in range(0, 12, 2))
        even_sum = sum(int(code[i]) for i in range(1, 12, 2))
        
        total = odd_sum + (even_sum * 3)
        check_digit = (10 - (total % 10)) % 10
        
        return check_digit == int(code[12])
    
    def apply_spanish_ean13_patterns(self, code):
        """Aplica patrones conocidos de EAN-13 españoles"""
        if len(code) != 13:
            return code
        
        spanish_patterns = ['847', '840', '841', '842', '843', '844', '845', '846', '848', '849']
        
        # Si no empieza por patrón español, intentar corregir
        if not any(code.startswith(pattern) for pattern in spanish_patterns):
            # Intentar con 847 (más común)
            if code.startswith('641') or code.startswith('647'):
                candidate = '847' + code[3:]
                if self.validate_ean13(candidate):
                    if self.debug:
                        print(f"Patrón español aplicado: {code} → {candidate}")
                    return candidate
        
        return code
