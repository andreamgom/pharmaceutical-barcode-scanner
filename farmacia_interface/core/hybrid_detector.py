# core/hybrid_detector.py
import cv2
import numpy as np
from pathlib import Path
import sys
import json
import time
from pyzbar import pyzbar
import zxingcpp
from skimage.filters import threshold_otsu

# Importar tus detectores existentes
sys.path.append('..')
from gradient_detector import GradientDetector
from yolo_detector import YOLODetector

# Importar las clases auxiliares
from .barcode_preprocessor import BarcodePreprocessor
from .barcode_decoder import BarcodeDecoder
from .rectangle_merger import RectangleMerger

class NumpyEncoder(json.JSONEncoder):
    """Encoder personalizado para tipos de NumPy"""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                           np.int16, np.int32, np.int64, np.uint8,
                           np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class HybridDetector:
    """
    Detector híbrido COMPLETO que combina:
    1. YOLO para header y detección de zona barcode
    2. Recorte de zona barcode (SIN corrección de perspectiva)
    3. Gradient detector en zona recortada para detección y ordenamiento
    4. Decodificación híbrida robusta con todas las técnicas
    """
    
    def __init__(self, yolo_model_path=None, use_gradient=True, debug=False):
        self.debug = debug
        self.use_gradient = use_gradient
        
        # Inicializar detectores base
        if yolo_model_path:
            self.yolo_detector = YOLODetector(yolo_model_path)
        else:
            self.yolo_detector = YOLODetector()
        
        if use_gradient:
            self.gradient_detector = GradientDetector()
        
        # Inicializar componentes auxiliares
        self.preprocessor = BarcodePreprocessor()
        self.decoder = BarcodeDecoder(debug=debug)
        self.decoder.set_preprocessor(self.preprocessor)
        self.merger = RectangleMerger(debug=debug)
    
    def is_image_valid(self, image):
        """Verifica si la imagen tiene calidad suficiente (COPIADO DE TUS DETECTORES)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Verificar con pyzbar primero
        barcodes_pyzbar = pyzbar.decode(gray)
        if len(barcodes_pyzbar) > 0:
            return True, "Imagen válida - códigos detectados con pyzbar"
        
        # Verificar con zxingcpp como respaldo
        try:
            barcodes_zxing = zxingcpp.read_barcodes(gray)
            if len(barcodes_zxing) > 0:
                return True, "Imagen válida - códigos detectados con zxingcpp"
        except:
            pass
        
        # Verificar brillo general
        if np.mean(gray) < 50:
            return False, "Imagen demasiado oscura"
        if np.mean(gray) > 200:
            return False, "Imagen demasiado brillante"
        
        return False, "No se detectaron códigos de barras en la imagen"
    
    def process_image(self, image_path):
        """Procesa imagen con debug COMPLETO para header"""
        
        start_time = time.time()
        
        # Cargar imagen
        if isinstance(image_path, str):
            original_image = cv2.imread(image_path)
            image_name = Path(image_path).name
        else:
            original_image = image_path
            image_name = "unknown_image"
        
        if original_image is None:
            return None, "Error: No se pudo cargar la imagen"
        
        print(f"=== PROCESANDO IMAGEN: {image_name} ===")
        print(f"Dimensiones imagen: {original_image.shape}")
        
        # PASO 1: Detección YOLO con debug COMPLETO
        print("PASO 1: Ejecutando detección YOLO...")
        
        # Usar el detector YOLO directamente con debug
        if hasattr(self.yolo_detector, 'model'):
            # Ejecutar YOLO con confianza MÁS BAJA para header
            results = self.yolo_detector.model(original_image, conf=0.1, verbose=True)  # Confianza muy baja
            
            print(f"YOLO ejecutado con confianza 0.1")
            print(f"Resultados YOLO: {len(results)} detecciones")
            
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                print(f"Total detecciones: {len(boxes)}")
                
                # Analizar TODAS las detecciones
                for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = self.yolo_detector.class_names.get(int(cls), f"clase_{int(cls)}")
                    
                    print(f"  Detección {i+1}:")
                    print(f"    Clase: {class_name} (ID: {int(cls)})")
                    print(f"    Confianza: {conf:.3f}")
                    print(f"    BBox: ({x1}, {y1}, {x2}, {y2})")
                    print(f"    Área: {(x2-x1) * (y2-y1)}")
                    
                    # Verificar específicamente headers
                    if class_name == 'header':
                        print(f"    *** HEADER ENCONTRADO ***")
            else:
                print("❌ NO SE DETECTARON OBJETOS")
        
        # Continuar con el análisis normal
        yolo_results, yolo_error = self.yolo_detector.process_image(image_path)
        
        if yolo_error:
            return None, f"Error YOLO: {yolo_error}"
        
        # PASO 2: Analizar detecciones de header con debug
        header_detected = yolo_results['header_detected']
        yolo_detections = yolo_results['detections']
        
        print(f"PASO 2: Análisis de header:")
        print(f"  Header detectado por process_image: {header_detected}")
        print(f"  Detecciones header en results: {len(yolo_detections.get('header', []))}")
        print(f"  Detecciones barcode: {len(yolo_detections.get('barcode', []))}")
        print(f"  Detecciones code: {len(yolo_detections.get('code', []))}")
        
        # Mostrar detalles de TODAS las detecciones
        for detection_type, detections in yolo_detections.items():
            print(f"  {detection_type.upper()}:")
            for i, det in enumerate(detections):
                print(f"    {i+1}: bbox={det['bbox']}, conf={det['confidence']:.3f}, area={det.get('area', 'N/A')}")
        
        # Si no hay header, forzar búsqueda manual
        if not header_detected:
            print("⚠️ HEADER NO DETECTADO - Intentando detección manual...")
            header_detected = self._force_header_detection(original_image)
            print(f"Detección manual de header: {header_detected}")

        # PASO 3: AHORA recortar zona de códigos (DESPUÉS de detectar header)
        if self.debug:
            print("PASO 3: Recortando zona de códigos...")
        cropped_image, crop_info = self._crop_barcode_region(original_image, yolo_detections)
        
        # PASO 4: Usar detector de gradientes en zona recortada
        if self.use_gradient and cropped_image is not None:
            gradient_results = self._process_with_gradient_complete(cropped_image, header_detected)
            
            if gradient_results:
                # Ajustar coordenadas de gradient al espacio original
                adjusted_results = self._adjust_coordinates_to_original(
                    gradient_results, crop_info, original_image.shape
                )
            else:
                adjusted_results = yolo_results
        else:
            adjusted_results = yolo_results
        
        # PASO 5: DECODIFICACIÓN HÍBRIDA ROBUSTA EN ZONA ORIGINAL
        if adjusted_results and 'decoded_results' in adjusted_results:
            final_decoded_results, final_decoding_stats = self._decode_hybrid_codes_complete(
                original_image, adjusted_results['decoded_results']
            )
        else:
            final_decoded_results = {}
            final_decoding_stats = {"none": 26}
        
        # PASO 6: Combinar resultados finales
        final_results = self._merge_results_complete(
            yolo_results, adjusted_results, header_detected, 
            final_decoded_results, final_decoding_stats
        )
        final_results['processing_time'] = time.time() - start_time
        
        return final_results, None
    
    def _force_header_detection(self, image):
        """Fuerza detección de header usando heurísticas"""
        
        # Buscar en la parte superior de la imagen (primer 20%)
        h, w = image.shape[:2]
        top_region = image[:int(h * 0.2), :]
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
        
        # Buscar texto/patrones en la parte superior
        # Usar detección de contornos para encontrar regiones de texto
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Si hay suficientes contornos en la parte superior, probablemente hay header
        significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        if len(significant_contours) > 5:  # Umbral heurístico
            print(f"  Heurística: {len(significant_contours)} contornos significativos en parte superior")
            return True
        
        return False
    
    def _crop_barcode_region(self, image, detections):
        """Recorta la región de códigos con MARGEN INFERIOR EXTRA"""
        
        if not detections.get('barcode'):
            if self.debug:
                print("No se detectó región barcode, usando imagen completa")
            return image, {'offset_x': 0, 'offset_y': 0, 'scale': 1.0}
        
        # Tomar el barcode con mayor área (más confiable)
        barcode_detections = detections['barcode']
        best_barcode = max(barcode_detections, key=lambda x: x['area'])
        
        x1, y1, x2, y2 = best_barcode['bbox']
        
        # MÁRGENES ASIMÉTRICOS - MÁS MARGEN INFERIOR
        margin_x = int((x2 - x1) * 0.1)      # 10% horizontal
        margin_y_top = int((y2 - y1) * 0.1)  # 10% superior
        margin_y_bottom = int((y2 - y1) * 0.25)  # 25% inferior (MÁS MARGEN)
        
        h, w = image.shape[:2]
        
        # Coordenadas de recorte con márgenes asimétricos
        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y_top)
        crop_x2 = min(w, x2 + margin_x)
        crop_y2 = min(h, y2 + margin_y_bottom)  # Margen inferior mayor
        
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        crop_info = {
            'offset_x': crop_x1,
            'offset_y': crop_y1,
            'original_bbox': (x1, y1, x2, y2),
            'crop_bbox': (crop_x1, crop_y1, crop_x2, crop_y2),
            'scale': 1.0
        }
        
        if self.debug:
            print(f"Imagen recortada con margen inferior extra: {cropped.shape}")
            print(f"Márgenes aplicados - X: {margin_x}px, Y superior: {margin_y_top}px, Y inferior: {margin_y_bottom}px")
        
        return cropped, crop_info


    def _merge_results_complete(self, yolo_results, gradient_results, header_detected, 
                           final_decoded_results, final_decoding_stats):
        """Combina todos los resultados incluyendo imagen recortada limpia"""
        
        if gradient_results is None:
            if self.debug:
                print("Usando solo resultados de YOLO")
            return yolo_results

        # Determinar layout basado en header de YOLO
        if header_detected:
            grid_layout = (6, 4)
            max_codes = 24
        else:
            grid_layout = (7, 4)
            max_codes = 26

        # Asegurar que tenemos todas las posiciones
        complete_decoded_results = {}
        for position in range(1, max_codes + 1):
            if position in final_decoded_results:
                complete_decoded_results[position] = final_decoded_results[position]
            else:
                complete_decoded_results[position] = {
                    'code': "No detectado",
                    'method': "none",
                    'bbox': None,
                    'confidence': 0.0
                }
                final_decoding_stats["none"] += 1

        # Calcular códigos válidos
        valid_codes = sum(1 for r in complete_decoded_results.values() if r['code'] != "No detectado")

        # Crear resultado híbrido completo
        merged = {
            'image_name': yolo_results['image_name'],
            'original_image': yolo_results['original_image'],
            'method': "Híbrido Completo (YOLO + Gradientes + Decodificación Robusta)",
            
            # Información de header de YOLO (más confiable)
            'header_detected': header_detected,
            'grid_layout': grid_layout,
            'max_codes': max_codes,
            
            # Detección, ordenamiento y decodificación completa
            'decoded_results': complete_decoded_results,
            'valid_codes': valid_codes,
            'success_rate': float(valid_codes / max_codes),
            'decoding_stats': final_decoding_stats,
            
            # AÑADIR: Imagen recortada limpia (SIN anotaciones)
            'barcode_region': None,  # Se llenará abajo
            
            # Información adicional
            'detections': yolo_results['detections'],
            'yolo_header_confidence': self._get_header_confidence(yolo_results['detections']),
            'hybrid_info': {
                'yolo_header_used': True,
                'gradient_ordering_used': True,
                'barcode_region_cropped': True,
                'hybrid_decoding_used': True,
                'merge_applied': True,
                'all_preprocessing_used': True
            }
        }

        # GUARDAR imagen recortada LIMPIA (sin anotaciones)
        if yolo_results['detections'].get('barcode'):
            cropped_clean, _ = self._crop_barcode_region(
                yolo_results['original_image'], 
                yolo_results['detections']
            )
            merged['barcode_region'] = cropped_clean

        # Crear imagen anotada híbrida (SEPARADA de la limpia)
        merged['annotated_image'] = self._create_hybrid_annotation(
            yolo_results['original_image'], 
            complete_decoded_results,
            yolo_results['detections']
        )

        if self.debug:
            print(f"Resultado híbrido completo: {merged['valid_codes']} códigos válidos")

        return merged

    
    def _process_with_gradient_complete(self, cropped_image, header_detected):
        """Procesa imagen recortada con detector de gradientes COMPLETO"""
        
        # Guardar imagen temporal para gradient detector
        temp_dir = Path("data/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / "cropped_for_gradient.jpg"
        
        cv2.imwrite(str(temp_path), cropped_image)
        
        if self.debug:
            print("Procesando zona recortada con gradient detector completo...")
        
        # USAR MÉTODO COMPLETO DE GRADIENT DETECTOR (detect_and_order_barcodes_like_paste)
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            
            # Calcular gradientes (EXACTO DE TU GRADIENT_DETECTOR.PY)
            Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            
            # Parámetros fijos como en tu código
            energy_factor = 0.7
            threshold_percentile = 92
            
            # Mapa de energía
            energy_map = np.abs(Ix) - energy_factor * np.abs(Iy)
            energy_map = np.clip(energy_map, 0, 255)
            
            # Suavizado
            kernel_size = max(15, min(gray.shape) // 60)
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            smoothed = cv2.filter2D(energy_map, -1, kernel)
            
            # Umbralización
            heatmap_data = smoothed / np.max(smoothed) if np.max(smoothed) > 0 else smoothed
            threshold = np.percentile(heatmap_data, threshold_percentile)
            heatmap_selective = np.where(heatmap_data > threshold, heatmap_data, 0)
            
            # Convertir a binario
            binary_heatmap = (heatmap_selective > 0).astype(np.uint8) * 255
            
            # Encontrar contornos
            contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar con criterios de tu gradient_detector
            valid_regions = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h if h > 0 else 0
                area = cv2.contourArea(cnt)
                
                if (w > 30 and h > 15 and
                    1.2 < aspect_ratio < 5.0 and
                    area > 400):
                    valid_regions.append((x, y, w, h))
            
            if self.debug:
                print(f"Regiones detectadas inicialmente: {len(valid_regions)}")
            
            # Ordenamiento simple primero
            ordered_barcodes = self._simple_grid_ordering(valid_regions)
            
            if self.debug:
                print(f"Después del ordenamiento simple: {len(ordered_barcodes)} códigos")
            
            # Verificar si necesita merge
            needs_merge = self._check_if_merge_needed(ordered_barcodes)
            
            if needs_merge:
                if self.debug:
                    print("Detectadas inconsistencias, aplicando merge...")
                
                # Aplicar merge usando RectangleMerger
                valid_regions_for_merge = [barcode['bbox'] for barcode in ordered_barcodes]
                merged_regions = self.merger.merge_rectangles_by_layout_constraints(
                    valid_regions_for_merge,
                    max_codes_per_row=[4, 4, 4, 4, 4, 4, 2]
                )
                
                # Reordenar después del merge
                ordered_barcodes = self._simple_grid_ordering(merged_regions)
                
                if self.debug:
                    print(f"Después del merge: {len(ordered_barcodes)} códigos")
            else:
                if self.debug:
                    print("Ordenamiento simple suficiente, no se necesita merge")
            
            # Crear estructura de resultados compatible
            decoded_results = {}
            for barcode in ordered_barcodes:
                decoded_results[barcode['position']] = {
                    'code': "No detectado",  # Se decodificará después
                    'method': "gradient_detection",
                    'bbox': barcode['bbox'],
                    'confidence': barcode['confidence']
                }
            
            # Forzar header_detected del YOLO (más confiable)
            if header_detected:
                grid_layout = (6, 4)
                max_codes = 24
            else:
                grid_layout = (7, 4)
                max_codes = 26
            
            gradient_results = {
                'decoded_results': decoded_results,
                'header_detected': header_detected,
                'grid_layout': grid_layout,
                'max_codes': max_codes,
                'original_image': cropped_image
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error en gradient detector completo: {e}")
            gradient_results = None
        
        # Limpiar archivo temporal
        if temp_path.exists():
            temp_path.unlink()
        
        return gradient_results
    
    def _simple_grid_ordering(self, regions):
        """Ordenamiento simple en grid (COPIADO DE TU GRADIENT_DETECTOR.PY)"""
        if not regions:
            return []

        # Convertir a lista de centroides con rectángulos
        centroids = []
        for region in regions:
            if len(region) == 4:  # (x, y, w, h)
                x, y, w, h = region
                cx = x + w/2
                cy = y + h/2
                centroids.append((cx, cy, region))

        # Ordenar por filas (Y) primero
        centroids.sort(key=lambda c: c[1])  # Ordenar por Y

        # Agrupar en filas
        rows = []
        current_row = [centroids[0]]
        for i in range(1, len(centroids)):
            cx, cy, region = centroids[i]
            last_cy = current_row[-1][1]
            
            # Si la diferencia en Y es pequeña, está en la misma fila
            if abs(cy - last_cy) < 50:  # Threshold para misma fila
                current_row.append(centroids[i])
            else:
                # Nueva fila
                rows.append(current_row)
                current_row = [centroids[i]]

        # Añadir la última fila
        if current_row:
            rows.append(current_row)

        # Ordenar cada fila por X
        for row in rows:
            row.sort(key=lambda c: c[0])  # Ordenar por X

        # Crear lista final ordenada
        ordered_barcodes = []
        position = 1
        for row in rows:
            for cx, cy, region in row:
                ordered_barcodes.append({
                    'position': position,
                    'bbox': region,
                    'centroid': (cx, cy),
                    'confidence': 1.0
                })
                position += 1

        return ordered_barcodes
    
    def _check_if_merge_needed(self, ordered_barcodes):
        """Verifica si se necesita merge (COPIADO DE TU GRADIENT_DETECTOR.PY)"""
        # Criterio 1: Demasiados códigos (más de 26)
        if len(ordered_barcodes) > 26:
            if self.debug:
                print(f"Criterio 1: Demasiados códigos ({len(ordered_barcodes)} > 26)")
            return True

        # Criterio 2: Filas con demasiados códigos
        rows = self._group_by_rows(ordered_barcodes)
        max_codes_per_row = [4, 4, 4, 4, 4, 4, 2]

        for i, row in enumerate(rows):
            expected_count = max_codes_per_row[i] if i < len(max_codes_per_row) else 4
            if len(row) > expected_count:
                if self.debug:
                    print(f"Criterio 2: Fila {i+1} tiene {len(row)} códigos (esperado: {expected_count})")
                return True

        # Criterio 3: Códigos muy pequeños (posibles fragmentos)
        small_codes = [b for b in ordered_barcodes if b['bbox'][2] * b['bbox'][3] < 300]
        if len(small_codes) > len(ordered_barcodes) * 0.3:  # Más del 30% son pequeños
            if self.debug:
                print(f"Criterio 3: Muchos códigos pequeños ({len(small_codes)}/{len(ordered_barcodes)})")
            return True

        if self.debug:
            print("No se detectaron inconsistencias")
        return False
    
    def _group_by_rows(self, ordered_barcodes):
        """Agrupa códigos por filas (COPIADO DE TU GRADIENT_DETECTOR.PY)"""
        if not ordered_barcodes:
            return []

        # Ordenar por Y
        sorted_codes = sorted(ordered_barcodes, key=lambda b: b['centroid'][1])

        rows = []
        current_row = [sorted_codes[0]]

        for i in range(1, len(sorted_codes)):
            current_y = sorted_codes[i]['centroid'][1]
            last_y = current_row[-1]['centroid'][1]

            if abs(current_y - last_y) < 50:  # Misma fila
                current_row.append(sorted_codes[i])
            else:
                rows.append(current_row)
                current_row = [sorted_codes[i]]

        if current_row:
            rows.append(current_row)

        return rows
    
    def _adjust_coordinates_to_original(self, gradient_results, crop_info, original_shape):
        """Ajusta las coordenadas del gradient detector al espacio de imagen original"""
        
        if not gradient_results or not gradient_results.get('decoded_results'):
            return gradient_results

        offset_x = crop_info['offset_x']
        offset_y = crop_info['offset_y']

        # Ajustar coordenadas de bounding boxes
        adjusted_decoded = {}

        for position, result in gradient_results['decoded_results'].items():
            adjusted_result = result.copy()

            if result.get('bbox'):
                x, y, w, h = result['bbox']
                # Trasladar coordenadas al espacio original
                adjusted_bbox = (x + offset_x, y + offset_y, w, h)
                adjusted_result['bbox'] = adjusted_bbox

            adjusted_decoded[position] = adjusted_result

        # Crear nueva estructura con coordenadas ajustadas
        adjusted_results = gradient_results.copy()
        adjusted_results['decoded_results'] = adjusted_decoded

        return adjusted_results
    
    def _decode_hybrid_codes_complete(self, original_image, detected_positions):
        """Decodifica códigos usando TODAS las técnicas de preprocesamiento"""
        
        decoded_results = {}
        
        # Estadísticas completas como en tu gradient_detector
        decoding_stats = {
            "original": 0, "contrast": 0, "brightness_up": 0, "brightness_down": 0,
            "clahe": 0, "gaussian_blur": 0, "median_blur": 0, "bilateral": 0,
            "sharpen": 0, "adaptive_thresh": 0, "otsu_thresh": 0, "morphology": 0,
            "resize_2x": 0, "resize_3x": 0, "resize_4x": 0, "resize_5x": 0, "resize_6x": 0,
            "none": 0
        }
        
        for position, detection in detected_positions.items():
            if detection.get('bbox'):
                # Extraer ROI de la imagen original
                if len(detection['bbox']) == 4:
                    x, y, w, h = detection['bbox']
                    x1, y1, x2, y2 = x, y, x + w, y + h
                else:
                    x1, y1, x2, y2 = detection['bbox']
                
                margin = 20
                roi = original_image[max(0, y1-margin):min(original_image.shape[0], y2+margin),
                                   max(0, x1-margin):min(original_image.shape[1], x2+margin)]
                
                if roi.size > 0:
                    # Usar método de decodificación híbrido COMPLETO
                    code_value, method = self.decoder.decode_barcode_hybrid(roi)
                    
                    decoded_results[position] = {
                        'code': code_value,
                        'method': method,
                        'bbox': detection['bbox'],
                        'confidence': float(detection.get('confidence', 1.0))
                    }
                    
                    # Actualizar estadísticas
                    if method in decoding_stats:
                        decoding_stats[method] += 1
                    else:
                        decoding_stats["none"] += 1
                else:
                    decoded_results[position] = {
                        'code': "No detectado",
                        'method': "none",
                        'bbox': detection['bbox'],
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
        
        return decoded_results, decoding_stats
    
    def _merge_results_complete(self, yolo_results, gradient_results, header_detected, final_decoded_results, final_decoding_stats):
        """Combina todos los resultados de forma completa incluyendo imagen recortada LIMPIA"""
        
        if gradient_results is None:
            if self.debug:
                print("Usando solo resultados de YOLO")
            return yolo_results

        # Determinar layout basado en header de YOLO (DETECTADO ANTES DE RECORTAR)
        if header_detected:
            grid_layout = (6, 4)
            max_codes = 24
        else:
            grid_layout = (7, 4)
            max_codes = 26

        # Asegurar que tenemos todas las posiciones
        complete_decoded_results = {}
        for position in range(1, max_codes + 1):
            if position in final_decoded_results:
                complete_decoded_results[position] = final_decoded_results[position]
            else:
                complete_decoded_results[position] = {
                    'code': "No detectado",
                    'method': "none",
                    'bbox': None,
                    'confidence': 0.0
                }
                final_decoding_stats["none"] += 1

        # Calcular códigos válidos
        valid_codes = sum(1 for r in complete_decoded_results.values() if r['code'] != "No detectado")

        # GUARDAR imagen recortada LIMPIA (ANTES de cualquier anotación)
        barcode_region_clean = None
        if yolo_results['detections'].get('barcode'):
            cropped_clean, _ = self._crop_barcode_region(
                yolo_results['original_image'], 
                yolo_results['detections']
            )
            barcode_region_clean = cropped_clean  # Imagen SIN anotaciones
            
            if self.debug:
                print(f"Imagen recortada limpia guardada: {cropped_clean.shape}")

        # Crear resultado híbrido completo
        merged = {
            'image_name': yolo_results['image_name'],
            'original_image': yolo_results['original_image'],
            'method': "Híbrido Completo (YOLO + Gradientes + Decodificación Robusta)",
            
            # Información de header de YOLO (detectado ANTES del recorte)
            'header_detected': header_detected,
            'grid_layout': grid_layout,
            'max_codes': max_codes,
            
            # Detección, ordenamiento y decodificación completa
            'decoded_results': complete_decoded_results,
            'valid_codes': valid_codes,
            'success_rate': float(valid_codes / max_codes),
            'decoding_stats': final_decoding_stats,
            
            # IMAGEN RECORTADA LIMPIA (sin anotaciones)
            'barcode_region': barcode_region_clean,
            
            # Información adicional
            'detections': yolo_results['detections'],
            'yolo_header_confidence': self._get_header_confidence(yolo_results['detections']),
            'hybrid_info': {
                'yolo_header_used': True,
                'gradient_ordering_used': True,
                'barcode_region_cropped': True,
                'hybrid_decoding_used': True,
                'merge_applied': True,
                'all_preprocessing_used': True,
                'header_detected_before_crop': header_detected
            }
        }

        # Crear imagen anotada híbrida (SEPARADA de la limpia)
        merged['annotated_image'] = self._create_hybrid_annotation(
            yolo_results['original_image'], 
            complete_decoded_results,
            yolo_results['detections']
        )

        if self.debug:
            print(f"Resultado híbrido completo: {merged['valid_codes']} códigos válidos")
            print(f"Header detectado antes del recorte: {header_detected}")
            print(f"Imagen recortada disponible: {barcode_region_clean is not None}")

        return merged

    
    def _get_header_confidence(self, detections):
        """Obtiene la confianza de detección de header de YOLO"""
        if detections.get('header'):
            return max(det['confidence'] for det in detections['header'])
        return 0.0
    
    def _create_hybrid_annotation(self, image, decoded_results, yolo_detections):
        """Crea imagen anotada combinando información de ambos detectores"""
        
        annotated = image.copy()

        # Dibujar detecciones de códigos
        for position, result in decoded_results.items():
            if result.get('bbox'):
                if len(result['bbox']) == 4:
                    x, y, w, h = result['bbox']
                    x1, y1, x2, y2 = x, y, x + w, y + h
                else:
                    x1, y1, x2, y2 = result['bbox']

                # Color según estado del código
                if result['code'] != "No detectado":
                    color = (0, 255, 0)  # Verde para códigos detectados
                    thickness = 2
                else:
                    color = (0, 0, 255)  # Rojo para no detectados
                    thickness = 1

                # Dibujar rectángulo
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

                # Dibujar número de posición
                cv2.putText(annotated, str(position), (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Dibujar header si fue detectado por YOLO
        for header_det in yolo_detections.get('header', []):
            x1, y1, x2, y2 = header_det['bbox']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Magenta para header
            cv2.putText(annotated, "HEADER", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Dibujar región de barcode usada para recorte
        for barcode_det in yolo_detections.get('barcode', []):
            x1, y1, x2, y2 = barcode_det['bbox']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 1)  # Cyan para barcode

        return annotated
    
    def save_results_to_json(self, results, base_dir='data/predictions/hybrid'):
        """Guarda los resultados híbridos en formato JSON"""
        
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        image_name_clean = results['image_name'].replace('.jpg', '')
        
        # JSON COMPLETO
        json_data_complete = {
            'image_name': results['image_name'],
            'method': results['method'],
            'processing_time': results.get('processing_time', 0),
            'header_detected': bool(results['header_detected']),
            'grid_layout': {
                'rows': int(results['grid_layout'][0]),
                'cols': int(results['grid_layout'][1])
            },
            'max_codes': int(results['max_codes']),
            'valid_codes': int(results['valid_codes']),
            'success_rate': float(results['success_rate']),
            'decoding_stats': results.get('decoding_stats', {}),
            'hybrid_info': results.get('hybrid_info', {}),
            'codes': {}
        }
        
        # JSON SIMPLE
        codes_list = []
        
        # Llenar ambos formatos
        for position in range(1, results['max_codes'] + 1):
            if position in results['decoded_results']:
                code = results['decoded_results'][position]['code']
                method = results['decoded_results'][position]['method']
                confidence = float(results['decoded_results'][position]['confidence'])
                
                # Para JSON completo
                json_data_complete['codes'][str(position)] = {
                    'code': code,
                    'method': method,
                    'confidence': confidence
                }
                
                # Para JSON simple
                codes_list.append(code if code != "No detectado" else "Código no encontrado")
            else:
                # Para JSON completo
                json_data_complete['codes'][str(position)] = {
                    'code': "No detectado",
                    'method': "none",
                    'confidence': 0.0
                }
                
                # Para JSON simple
                codes_list.append("Código no encontrado")
        
        # Guardar archivos
        filepath_complete = Path(base_dir) / f"{image_name_clean}_complete.json"
        with open(filepath_complete, 'w') as f:
            json.dump(json_data_complete, f, indent=2, cls=NumpyEncoder)
        
        filepath_simple = Path(base_dir) / f"{image_name_clean}_simple.json"
        with open(filepath_simple, 'w') as f:
            json.dump(codes_list, f, indent=2, cls=NumpyEncoder)
        
        return str(filepath_complete), str(filepath_simple)
