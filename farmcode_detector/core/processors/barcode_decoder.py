# core/processors/barcode_decoder.py
import cv2
import numpy as np
from pyzbar import pyzbar
from collections import Counter
import re


class BarcodeDecoder:
    """Decodificador robusto de códigos de barras con filtrado farmacéutico español estricto"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.preprocessor = None
        
        # 🆕 PREFIJOS FARMACÉUTICOS ESPAÑOLES VÁLIDOS
        self.spanish_pharma_prefixes = [
            '840', '841', '842', '843', '844', '845', '846', '847', '848', '849'
        ]
        
        # 🆕 PATRONES SOSPECHOSOS A RECHAZAR
        self.suspicious_patterns = [
            r'^0{4,}',          # 4 o más ceros consecutivos al inicio
            r'0{6,}',           # 6 o más ceros consecutivos en cualquier lugar
            r'(\d)\1{6,}',      # Mismo dígito repetido 7 o más veces
            r'^1{8,}',          # 8 o más unos consecutivos
            r'^9{8,}',          # 8 o más nueves consecutivos
            r'123456789',       # Secuencia ascendente obvia
            r'987654321',       # Secuencia descendente obvia
        ]
        
        # 🆕 ESTADÍSTICAS DE RECHAZO
        self.rejection_stats = {
            'wrong_prefix': 0,
            'invalid_ean13': 0,
            'suspicious_pattern': 0,
            'too_short': 0,
            'non_numeric': 0,
            'corrected_codes': 0,
            'total_rejected': 0
        }
    
    def set_preprocessor(self, preprocessor):
        """Inyecta el preprocessor desde el pipeline"""
        self.preprocessor = preprocessor
        if self.debug:
            print(f"✅ Preprocessor inyectado: {self.preprocessor is not None}")
    
    def _is_valid_spanish_pharma_code(self, code):
        """🆕 VALIDACIÓN ESTRICTA DE CÓDIGOS FARMACÉUTICOS ESPAÑOLES"""
        if not code or not isinstance(code, str):
            return False, "Código vacío o inválido"
        
        # 1. Debe ser numérico
        if not code.isdigit():
            self.rejection_stats['non_numeric'] += 1
            return False, f"Contiene caracteres no numéricos: '{code}'"
        
        # 2. Debe tener exactamente 13 dígitos para EAN-13
        if len(code) != 13:
            if len(code) < 8:
                self.rejection_stats['too_short'] += 1
                return False, f"Código demasiado corto ({len(code)} dígitos): '{code}'"
            else:
                # Códigos de 8-12 dígitos pueden ser válidos pero no EAN-13
                return True, f"Código válido pero no EAN-13 ({len(code)} dígitos)"
        
        # 3. Debe empezar con prefijo farmacéutico español
        prefix = code[:3]
        if prefix not in self.spanish_pharma_prefixes:
            self.rejection_stats['wrong_prefix'] += 1
            return False, f"Prefijo no farmacéutico español: '{prefix}' en '{code}'"
        
        # 4. Validar estructura EAN-13
        if not self._validate_ean13_structure(code):
            self.rejection_stats['invalid_ean13'] += 1
            return False, f"Estructura EAN-13 inválida: '{code}'"
        
        # 5. Verificar patrones sospechosos
        for pattern in self.suspicious_patterns:
            if re.search(pattern, code):
                self.rejection_stats['suspicious_pattern'] += 1
                return False, f"Patrón sospechoso detectado ({pattern}): '{code}'"
        
        # ✅ Código válido
        return True, f"Código farmacéutico español válido: '{code}'"
    
    def _validate_ean13_structure(self, code):
        """🆕 VALIDACIÓN ESTRICTA DE EAN-13 CON DÍGITO DE CONTROL"""
        if len(code) != 13 or not code.isdigit():
            return False
        
        # Calcular dígito de control EAN-13
        odd_sum = sum(int(code[i]) for i in range(0, 12, 2))
        even_sum = sum(int(code[i]) for i in range(1, 12, 2))
        
        total = odd_sum + (even_sum * 3)
        check_digit = (10 - (total % 10)) % 10
        
        return check_digit == int(code[12])
    
    def _attempt_code_correction(self, code):
        """🆕 CORRECCIÓN INTELIGENTE DE ERRORES COMUNES"""
        original_code = code
        corrections_applied = []
        
        if not code or not isinstance(code, str):
            return code, corrections_applied
        
        # Corrección 1: Eliminar caracteres no numéricos
        if not code.isdigit():
            code = re.sub(r'[^0-9]', '', code)
            if code != original_code:
                corrections_applied.append(f"Caracteres no numéricos eliminados")
        
        # Corrección 2: Ajustar longitud a 13 dígitos
        if len(code) == 12:
            # Intentar agregar dígito de control
            for check_digit in range(10):
                candidate = code + str(check_digit)
                if self._validate_ean13_structure(candidate):
                    code = candidate
                    corrections_applied.append(f"Dígito de control añadido: {check_digit}")
                    break
        elif len(code) == 14 and code.startswith('0'):
            # Eliminar cero inicial
            code = code[1:]
            corrections_applied.append("Cero inicial eliminado")
        
        # Corrección 3: Intentar corregir prefijos comunes mal leídos
        prefix_corrections = {
            '640': '840',  # 6 leído como 8
            '641': '841',
            '647': '847',
            '940': '840',  # 9 leído como 8
            '947': '847'
        }
        
        if len(code) >= 3:
            current_prefix = code[:3]
            if current_prefix in prefix_corrections:
                corrected_prefix = prefix_corrections[current_prefix]
                code = corrected_prefix + code[3:]
                corrections_applied.append(f"Prefijo corregido: {current_prefix} → {corrected_prefix}")
        
        # Corrección 4: Corregir dígitos de control incorrectos
        if len(code) == 13 and code.isdigit():
            prefix = code[:3]
            if prefix in self.spanish_pharma_prefixes:
                # Recalcular dígito de control
                base_code = code[:12]
                correct_check_digit = self._calculate_ean13_check_digit(base_code)
                if code[12] != str(correct_check_digit):
                    code = base_code + str(correct_check_digit)
                    corrections_applied.append(f"Dígito control corregido: {original_code[12]} → {correct_check_digit}")
        
        return code, corrections_applied
    
    def _calculate_ean13_check_digit(self, base_code):
        """🆕 CALCULA EL DÍGITO DE CONTROL EAN-13 CORRECTO"""
        if len(base_code) != 12:
            return None
        
        odd_sum = sum(int(base_code[i]) for i in range(0, 12, 2))
        even_sum = sum(int(base_code[i]) for i in range(1, 12, 2))
        total = odd_sum + (even_sum * 3)
        check_digit = (10 - (total % 10)) % 10
        return check_digit
    
    def _process_detected_code(self, raw_code, technique_name):
        """🆕 PROCESAMIENTO COMPLETO CON FILTRADO Y CORRECCIÓN"""
        if not raw_code:
            return None
        
        # Paso 1: Verificar código original
        is_valid, reason = self._is_valid_spanish_pharma_code(raw_code)
        
        if is_valid:
            if self.debug:
                print(f"      ✅ Código original válido: '{raw_code}' ({reason})")
            return {
                'data': raw_code,
                'technique': technique_name,
                'confidence': 1.0,
                'corrected': False,
                'corrections': [],
                'validation_reason': reason
            }
        
        if self.debug:
            print(f"      ❌ Código original rechazado: '{raw_code}' ({reason})")
        
        # Paso 2: Intentar corrección
        corrected_code, corrections = self._attempt_code_correction(raw_code)
        
        if corrections:
            if self.debug:
                print(f"      🔧 Correcciones aplicadas: {corrections}")
            
            # Verificar código corregido
            is_valid_corrected, reason_corrected = self._is_valid_spanish_pharma_code(corrected_code)
            
            if is_valid_corrected:
                self.rejection_stats['corrected_codes'] += 1
                if self.debug:
                    print(f"      ✅ Código corregido válido: '{raw_code}' → '{corrected_code}'")
                
                return {
                    'data': corrected_code,
                    'technique': technique_name,
                    'confidence': 0.8,  # Menor confianza para códigos corregidos
                    'corrected': True,
                    'corrections': corrections,
                    'original_code': raw_code,
                    'validation_reason': reason_corrected
                }
        
        # Paso 3: Código no salvable
        self.rejection_stats['total_rejected'] += 1
        if self.debug:
            print(f"      ❌ Código no salvable: '{raw_code}' (rechazado definitivamente)")
        
        return None
    
    def _expand_roi_for_failed_detection(self, image, bbox, expansion_factor=1.5):
        """Expansión HORIZONTAL AGRESIVA específica para códigos de barras"""
        x, y, w, h = bbox
        h_img, w_img = image.shape[:2]
        
        # EXPANSIÓN PRINCIPALMENTE HORIZONTAL (códigos de barras son horizontales)
        if expansion_factor <= 1.5:
            expansion_w = int(w * 0.4)  # 40% más ancho
            expansion_h = int(h * 0.2)  # 20% más alto
        elif expansion_factor <= 2.0:
            expansion_w = int(w * 0.8)  # 80% más ancho
            expansion_h = int(h * 0.3)  # 30% más alto
        else:
            expansion_w = int(w * 1.2)  # 120% más ancho
            expansion_h = int(h * 0.4)  # 40% más alto
        
        # Expandir hacia afuera manteniendo proporción
        new_x = max(0, x - expansion_w)
        new_y = max(0, y - expansion_h)
        new_x2 = min(w_img, x + w + expansion_w)
        new_y2 = min(h_img, y + h + expansion_h)
        
        # Calcular nuevas dimensiones
        new_w = new_x2 - new_x
        new_h = new_y2 - new_y
        
        if self.debug:
            print(f"      Expansión HORIZONTAL: ({x},{y},{w},{h}) → ({new_x},{new_y},{new_w},{new_h})")
            print(f"      Factor {expansion_factor}x: +{expansion_w}px horizontal, +{expansion_h}px vertical")
        
        return (new_x, new_y, new_w, new_h)
    
    def _decode_with_expansion_techniques(self, roi):
        """Técnicas específicas para ROIs expandidos con filtrado farmacéutico"""
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi.copy()
        
        # Técnicas optimizadas para códigos expandidos
        expansion_techniques = [
            ("original_expanded", roi_gray),
            ("resize_2x_expanded", cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
            ("resize_3x_expanded", cv2.resize(roi_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)),
            ("high_contrast_expanded", self._apply_expansion_contrast(roi_gray)),
            ("threshold_expanded", self._apply_expansion_threshold(roi_gray)),
            ("morphology_expanded", self._apply_expansion_morphology(roi_gray)),
        ]
        
        if self.debug:
            print(f"      Aplicando {len(expansion_techniques)} técnicas para expansión")
        
        # Probar cada técnica específica
        for technique_name, processed_image in expansion_techniques:
            # Intentar con pyzbar
            try:
                decoded_objects = pyzbar.decode(processed_image, symbols=[
                    pyzbar.ZBarSymbol.EAN13,
                    pyzbar.ZBarSymbol.CODE128,
                    pyzbar.ZBarSymbol.UPCA,
                    pyzbar.ZBarSymbol.EAN8
                ])
                
                if decoded_objects:
                    raw_code = decoded_objects[0].data.decode('utf-8')
                    
                    # 🆕 PROCESAR CON FILTRADO FARMACÉUTICO
                    processed_result = self._process_detected_code(raw_code, technique_name)
                    if processed_result:
                        return processed_result['data'], f"pyzbar_{technique_name}"
                    
            except Exception as e:
                if self.debug:
                    print(f"      Error en {technique_name}: {e}")
        
        # Si fallan las técnicas específicas, usar método completo
        return self.decode_barcode_region(roi)
    
    def _apply_expansion_contrast(self, image):
        """Contraste específico para ROIs expandidos"""
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        return clahe.apply(image)
    
    def _apply_expansion_threshold(self, image):
        """Umbralización específica para ROIs expandidos"""
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    def _apply_expansion_morphology(self, image):
        """Morfología específica para ROIs expandidos"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    def _decode_single_roi_with_expansion(self, image, bbox, position):
        """🆕 DECODIFICA UN ROI CON EXPANSIÓN Y FILTRADO FARMACÉUTICO"""
        original_bbox = bbox
        expansion_attempts = [1.0, 1.4, 1.8, 2.2, 2.6]
        
        for attempt, factor in enumerate(expansion_attempts):
            if attempt > 0:
                bbox = self._expand_roi_for_failed_detection(image, original_bbox, factor)
                if self.debug:
                    print(f"    Intento {attempt + 1} con expansión HORIZONTAL {factor}x")
            
            x, y, w, h = bbox
            
            # Extraer ROI
            roi = image[y:y+h, x:x+w]
            
            if roi.size == 0:
                continue
                
            # VERIFICAR TAMAÑO MÍNIMO DEL ROI
            if roi.shape[0] < 10 or roi.shape[1] < 20:
                if self.debug:
                    print(f"      ROI demasiado pequeño: {roi.shape}")
                continue
            
            # USAR TÉCNICAS ESPECÍFICAS PARA EXPANSIÓN CON FILTRADO
            if attempt > 0:
                code_value, method = self._decode_with_expansion_techniques(roi)
            else:
                code_value, method = self.decode_barcode_hybrid(roi)
            
            if code_value != "No detectado":
                # 🆕 VALIDACIÓN FINAL DEL CÓDIGO DETECTADO
                is_valid, reason = self._is_valid_spanish_pharma_code(code_value)
                
                if is_valid:
                    if attempt > 0 and self.debug:
                        print(f"    ✅ Éxito con expansión HORIZONTAL {factor}x: {code_value}")
                    return {
                        'code': code_value,
                        'method': f"expansion_horizontal_{method}" if attempt > 0 else method,
                        'bbox': bbox,
                        'confidence': 1.0,
                        'validation_reason': reason
                    }
                else:
                    if self.debug:
                        print(f"    ❌ Código detectado pero rechazado: {code_value} ({reason})")
        
        if self.debug:
            print(f"    ❌ Falló incluso con expansión HORIZONTAL máxima")
        
        return {
            'code': "No detectado",
            'method': "expansion_horizontal_failed",
            'confidence': 0.0,
            'bbox': original_bbox
        }
    
    def decode_grid_complete(self, original_image, detection_result, max_codes):
        """🆕 MÉTODO PRINCIPAL CON FILTRADO FARMACÉUTICO ESTRICTO"""
        
        decoded_results = {}
        decoding_stats = {
            "original": 0, "contrast": 0, "brightness_up": 0, "brightness_down": 0,
            "clahe": 0, "gaussian_blur": 0, "median_blur": 0, "bilateral": 0,
            "sharpen": 0, "adaptive_thresh": 0, "otsu_thresh": 0, "morphology": 0,
            "gamma_12": 0, "gamma_08": 0, "unsharp": 0, "hist_eq": 0, "noise_reduction": 0,
            "edge_enhancement": 0, "contrast_stretch": 0, "gaussian_pyramid": 0,
            "top_hat": 0, "black_hat": 0,
            "resize_2x": 0, "resize_3x": 0, "resize_4x": 0, "resize_5x": 0, "resize_6x": 0,
            "expansion_successes": 0, "expansion_horizontal_failed": 0,
            "none": 0
        }
        
        # 🆕 REINICIAR ESTADÍSTICAS DE RECHAZO
        self.rejection_stats = {key: 0 for key in self.rejection_stats.keys()}
        
        if self.debug:
            print(f"🔧 DECODIFICANDO CON FILTRADO FARMACÉUTICO ESPAÑOL ESTRICTO...")
            print(f"   Preprocessor disponible: {self.preprocessor is not None}")
        
        if not detection_result or not detection_result.get('decoded_results'):
            # Crear resultados vacíos
            for position in range(1, max_codes + 1):
                decoded_results[position] = {
                    'code': "No detectado",
                    'method': "none",
                    'bbox': None,
                    'confidence': 0.0
                }
                decoding_stats["none"] += 1
            
            return decoded_results, decoding_stats
        
        # Decodificación con filtrado farmacéutico
        for position in range(1, max_codes + 1):
            if position in detection_result['decoded_results']:
                detection = detection_result['decoded_results'][position]
                
                if detection and detection.get('bbox'):
                    if self.debug and position <= 5:
                        print(f"  Decodificando pos {position}: bbox={detection['bbox']}")
                    
                    # USAR DECODIFICACIÓN CON FILTRADO FARMACÉUTICO
                    result = self._decode_single_roi_with_expansion(original_image, detection['bbox'], position)
                    
                    decoded_results[position] = {
                        'code': result['code'],
                        'method': result['method'],
                        'bbox': result['bbox'],
                        'confidence': detection.get('confidence', 1.0),
                        'validation_reason': result.get('validation_reason', '')
                    }
                    
                    # Actualizar estadísticas
                    if result['method'].startswith('expansion_horizontal_'):
                        decoding_stats["expansion_successes"] += 1
                        base_method = result['method'].replace('expansion_horizontal_', '').split('_')[1] if '_' in result['method'].replace('expansion_horizontal_', '') else result['method'].replace('expansion_horizontal_', '')
                    elif result['method'] == "expansion_horizontal_failed":
                        decoding_stats["expansion_horizontal_failed"] += 1
                        base_method = "expansion_horizontal_failed"
                    else:
                        base_method = result['method'].split('_')[1] if '_' in result['method'] else result['method']
                    
                    if base_method in decoding_stats:
                        decoding_stats[base_method] += 1
                    else:
                        decoding_stats["none"] += 1
                        
                else:
                    decoded_results[position] = {
                        'code': "No detectado",
                        'method': "none",
                        'bbox': None,
                        'confidence': 0.0
                    }
                    decoding_stats["none"] += 1
            else:
                decoded_results[position] = {
                    'code': "No detectado",
                    'method': "none",
                    'bbox': None,
                    'confidence': 0.0
                }
                decoding_stats["none"] += 1
        
        if self.debug:
            valid_codes = sum(1 for r in decoded_results.values() if r['code'] != "No detectado")
            expansion_successes = decoding_stats["expansion_successes"]
            expansion_failures = decoding_stats["expansion_horizontal_failed"]
            
            print(f"   ✅ Decodificación completada: {valid_codes}/{max_codes} códigos válidos")
            print(f"   🔍 Éxitos por expansión: {expansion_successes}")
            print(f"   ❌ Fallos de expansión: {expansion_failures}")
            
            # 🆕 MOSTRAR ESTADÍSTICAS DE FILTRADO
            print(f"   🚫 ESTADÍSTICAS DE FILTRADO:")
            print(f"      Prefijo incorrecto: {self.rejection_stats['wrong_prefix']}")
            print(f"      EAN-13 inválido: {self.rejection_stats['invalid_ean13']}")
            print(f"      Patrón sospechoso: {self.rejection_stats['suspicious_pattern']}")
            print(f"      Códigos corregidos: {self.rejection_stats['corrected_codes']}")
            print(f"      Total rechazados: {self.rejection_stats['total_rejected']}")
            
            # Mostrar estadísticas de métodos usados
            used_methods = {k: v for k, v in decoding_stats.items() if v > 0}
            if used_methods:
                print(f"   📊 Métodos exitosos: {used_methods}")
        
        return decoded_results, decoding_stats
    
    def decode_barcode_region(self, roi):
        """🆕 MÉTODO PRINCIPAL CON FILTRADO FARMACÉUTICO"""
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi.copy()
        
        # Usar todas las técnicas de preprocesamiento
        preprocessing_results = self.decode_with_preprocessing(roi_gray)
        
        if preprocessing_results:
            # Seleccionar el mejor resultado con filtrado
            return self._select_best_result_with_filtering(preprocessing_results)
        
        return "No detectado", "none"
    
    def decode_barcode_hybrid(self, roi):
        """Alias para compatibilidad con código existente"""
        if self.debug:
            print(f"    🔧 Decodificando con preprocessor: {self.preprocessor is not None}")
        
        return self.decode_barcode_region(roi)
    
    def decode_with_preprocessing(self, roi_gray):
        """🆕 DECODIFICA CON PREPROCESAMIENTO Y FILTRADO FARMACÉUTICO"""
        if not self.preprocessor:
            if self.debug:
                print("⚠️ No hay preprocessor disponible, usando técnicas básicas")
            return self._decode_with_basic_techniques(roi_gray)
        
        try:
            preprocessing_techniques = self.preprocessor.get_all_preprocessing_techniques(roi_gray)
            
            if self.debug:
                print(f"    📋 Aplicando {len(preprocessing_techniques)} técnicas de preprocesamiento")
            
            results = {}
            
            for technique_name, processed_image in preprocessing_techniques:
                # Intentar con pyzbar
                pyzbar_result = self._try_pyzbar_with_filtering(processed_image, technique_name)
                
                # Guardar resultados solo si son válidos farmacéuticamente
                if pyzbar_result:
                    results[technique_name] = {
                        'pyzbar': pyzbar_result,
                        'processed_image': processed_image
                    }
                    
                    if self.debug:
                        code = pyzbar_result['data']
                        corrected = " (corregido)" if pyzbar_result.get('corrected', False) else ""
                        print(f"      ✅ {technique_name}: '{code}'{corrected}")
            
            return results
            
        except Exception as e:
            if self.debug:
                print(f"    ❌ Error en preprocesamiento: {e}")
            return self._decode_with_basic_techniques(roi_gray)
    
    def _try_pyzbar_with_filtering(self, processed_image, technique_name):
        """🆕 PYZBAR CON FILTRADO FARMACÉUTICO ESTRICTO"""
        try:
            decoded_objects = pyzbar.decode(processed_image, symbols=[
                pyzbar.ZBarSymbol.EAN13,
                pyzbar.ZBarSymbol.CODE128,
                pyzbar.ZBarSymbol.UPCA,
                pyzbar.ZBarSymbol.EAN8
            ])
            
            if decoded_objects:
                raw_code = decoded_objects[0].data.decode('utf-8')
                
                # 🆕 PROCESAR CON FILTRADO FARMACÉUTICO
                processed_result = self._process_detected_code(raw_code, technique_name)
                
                if processed_result:
                    return {
                        'data': processed_result['data'],
                        'type': decoded_objects[0].type,
                        'technique': technique_name,
                        'confidence': processed_result['confidence'],
                        'corrected': processed_result.get('corrected', False),
                        'corrections': processed_result.get('corrections', []),
                        'original_code': processed_result.get('original_code'),
                        'validation_reason': processed_result.get('validation_reason', '')
                    }
                
        except Exception as e:
            if self.debug:
                print(f"Error pyzbar con {technique_name}: {e}")
        
        return None
    
    def _decode_with_basic_techniques(self, roi_gray):
        """🆕 TÉCNICAS BÁSICAS CON FILTRADO FARMACÉUTICO"""
        if self.debug:
            print("    🔧 Usando técnicas básicas con filtrado farmacéutico")
        
        basic_techniques = [
            ("original", roi_gray),
            ("clahe", self._apply_basic_clahe(roi_gray)),
            ("contrast", self._enhance_basic_contrast(roi_gray)),
            ("adaptive_thresh", self._apply_basic_adaptive_threshold(roi_gray)),
            ("resize_2x", cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        ]
        
        results = {}
        for technique_name, processed_image in basic_techniques:
            pyzbar_result = self._try_pyzbar_with_filtering(processed_image, technique_name)
            
            if pyzbar_result:
                results[technique_name] = {
                    'pyzbar': pyzbar_result,
                    'processed_image': processed_image
                }
        
        return results
    
    def _select_best_result_with_filtering(self, preprocessing_results):
        """🆕 SELECCIONA EL MEJOR RESULTADO CON FILTRADO FARMACÉUTICO"""
        all_codes = []
        
        # Recopilar todos los códigos válidos farmacéuticamente
        for technique_name, results in preprocessing_results.items():
            if results['pyzbar']:
                code = results['pyzbar']['data']
                method = f"pyzbar_{technique_name}"
                confidence = results['pyzbar'].get('confidence', 1.0)
                corrected = results['pyzbar'].get('corrected', False)
                
                # Priorizar códigos no corregidos
                priority = 1.0 if not corrected else 0.8
                
                all_codes.append((code, method, len(code), confidence * priority, corrected))
        
        if not all_codes:
            return "No detectado", "none"
        
        # Estrategia de selección mejorada con prioridad farmacéutica
        return self._apply_pharma_selection_strategy(all_codes)
    
    def _apply_pharma_selection_strategy(self, all_codes):
        """🆕 ESTRATEGIA DE SELECCIÓN ESPECÍFICA PARA CÓDIGOS FARMACÉUTICOS"""
        
        # Estrategia 1: Priorizar códigos EAN-13 farmacéuticos españoles
        ean13_pharma_codes = [
            code for code in all_codes 
            if code[2] == 13 and code[0][:3] in self.spanish_pharma_prefixes
        ]
        
        if ean13_pharma_codes:
            # Si hay múltiples códigos EAN-13 farmacéuticos, tomar el de mayor confianza
            best_result = max(ean13_pharma_codes, key=lambda x: x[3])  # Por confianza
            return best_result[0], best_result[1]
        
        # Estrategia 2: Códigos de 13 dígitos (aunque no sean farmacéuticos españoles)
        thirteen_digit_codes = [code for code in all_codes if code[2] == 13]
        
        if thirteen_digit_codes:
            best_result = max(thirteen_digit_codes, key=lambda x: x[3])
            return best_result[0], best_result[1]
        
        # Estrategia 3: El código más largo con mayor confianza
        if all_codes:
            best_result = max(all_codes, key=lambda x: (x[2], x[3]))  # Por longitud y confianza
            return best_result[0], best_result[1]
        
        return "No detectado", "none"
    
    def get_rejection_report(self):
        """🆕 OBTIENE REPORTE DETALLADO DE RECHAZOS"""
        total_processed = sum(self.rejection_stats.values())
        
        report = {
            'total_codes_processed': total_processed,
            'rejection_breakdown': self.rejection_stats.copy(),
            'rejection_rate': self.rejection_stats['total_rejected'] / total_processed if total_processed > 0 else 0,
            'correction_rate': self.rejection_stats['corrected_codes'] / total_processed if total_processed > 0 else 0
        }
        
        return report
    
    def validate_ean13(self, code):
        """Valida un código EAN-13 usando el dígito de control (mantenido por compatibilidad)"""
        return self._validate_ean13_structure(code)
    
    def apply_spanish_ean13_patterns(self, code):
        """🆕 APLICA PATRONES FARMACÉUTICOS ESPAÑOLES CON VALIDACIÓN ESTRICTA"""
        if len(code) != 13:
            return code
        
        # Si ya es un código farmacéutico español válido, devolverlo
        if code[:3] in self.spanish_pharma_prefixes and self._validate_ean13_structure(code):
            return code
        
        # Intentar corrección automática
        corrected_code, corrections = self._attempt_code_correction(code)
        
        if corrections and corrected_code != code:
            is_valid, _ = self._is_valid_spanish_pharma_code(corrected_code)
            if is_valid:
                if self.debug:
                    print(f"Patrón farmacéutico español aplicado: {code} → {corrected_code}")
                return corrected_code
        
        return code
    
    # Métodos básicos para cuando no hay preprocessor
    def _apply_basic_clahe(self, image):
        """CLAHE básico"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)
    
    def _enhance_basic_contrast(self, image):
        """Mejora de contraste básica"""
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return clahe.apply(image)
    
    def _apply_basic_adaptive_threshold(self, image):
        """Umbralización adaptativa básica"""
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    def create_decoding_report(self, decoded_results, validation_results=None):
        """🆕 CREA REPORTE DETALLADO CON INFORMACIÓN FARMACÉUTICA"""
        stats = self.get_decoding_statistics(decoded_results)
        rejection_report = self.get_rejection_report()
        
        report = {
            'summary': {
                'total_positions': stats['total_codes'],
                'successfully_decoded': stats['valid_codes'],
                'failed_to_decode': stats['failed_codes'],
                'success_rate': f"{stats['success_rate']:.2%}"
            },
            'library_performance': {
                'pyzbar': stats['pyzbar_codes']
            },
            'method_breakdown': stats['method_breakdown'],
            'pharma_filtering': {
                'total_processed': rejection_report['total_codes_processed'],
                'rejected_codes': rejection_report['rejection_breakdown']['total_rejected'],
                'corrected_codes': rejection_report['rejection_breakdown']['corrected_codes'],
                'rejection_rate': f"{rejection_report['rejection_rate']:.2%}",
                'correction_rate': f"{rejection_report['correction_rate']:.2%}",
                'rejection_reasons': {
                    'wrong_prefix': rejection_report['rejection_breakdown']['wrong_prefix'],
                    'invalid_ean13': rejection_report['rejection_breakdown']['invalid_ean13'],
                    'suspicious_pattern': rejection_report['rejection_breakdown']['suspicious_pattern'],
                    'non_numeric': rejection_report['rejection_breakdown']['non_numeric'],
                    'too_short': rejection_report['rejection_breakdown']['too_short']
                }
            }
        }
        
        if validation_results:
            valid_count = sum(1 for v in validation_results.values() if v['valid'])
            report['validation'] = {
                'valid_codes': valid_count,
                'invalid_codes': len(validation_results) - valid_count,
                'validation_rate': f"{valid_count / len(validation_results):.2%}" if validation_results else "0%"
            }
        
        return report
    
    def get_decoding_statistics(self, decoded_results):
        """🆕 ESTADÍSTICAS DE DECODIFICACIÓN CON INFORMACIÓN FARMACÉUTICA"""
        total_codes = len(decoded_results)
        valid_codes = sum(1 for r in decoded_results.values() if r['code'] != "No detectado")
        
        # Agrupar por método
        method_stats = {}
        for result in decoded_results.values():
            method = result['method']
            if method in method_stats:
                method_stats[method] += 1
            else:
                method_stats[method] = 1
        
        # Contar códigos farmacéuticos españoles válidos
        spanish_pharma_codes = sum(
            1 for r in decoded_results.values() 
            if r['code'] != "No detectado" and len(r['code']) >= 3 and r['code'][:3] in self.spanish_pharma_prefixes
        )
        
        # Agrupar por librería (solo pyzbar)
        pyzbar_count = sum(1 for r in decoded_results.values() 
                          if r['method'].startswith('pyzbar_'))
        
        return {
            'total_codes': total_codes,
            'valid_codes': valid_codes,
            'spanish_pharma_codes': spanish_pharma_codes,
            'success_rate': valid_codes / total_codes if total_codes > 0 else 0,
            'pharma_rate': spanish_pharma_codes / total_codes if total_codes > 0 else 0,
            'pyzbar_codes': pyzbar_count,
            'failed_codes': total_codes - valid_codes,
            'method_breakdown': method_stats
        }
