# core/detectors/grid_detector.py - VERSIÓN CONSERVADORA QUE FUNCIONA

import cv2
import numpy as np
from pathlib import Path

class GridDetector:
    """Detector conservador - cambios mínimos sobre el código original"""
    
    def __init__(self, debug=False):
        self.debug = debug
        # Parámetros originales que funcionaban
        self.energy_factor = 0.67
        self.threshold_percentile = 89
        
    def detect_with_full_validation(self, barcode_result, header_result, barcode_detector, grid_processor, merger):
        """Método principal SIN cambios drásticos"""
        grid_config = header_result['grid_config']
        
        if self.debug:
            print("🔧 DETECCIÓN CONSERVADORA CON GRADIENTES")
        
        if barcode_result['detection_success']:
            # Usar método original con MÍNIMAS mejoras
            ordered_detections = self._detect_with_minimal_improvements(
                barcode_result['cropped_image']
            )
            
            # Crear estructura compatible
            decoded_results = {}
            for detection in ordered_detections:
                if detection['position'] <= grid_config['max_codes']:
                    decoded_results[detection['position']] = {
                        'code': "No detectado",
                        'method': "gradient_detection",
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence']
                    }
            
            return {
                'decoded_results': decoded_results,
                'header_detected': header_result['header_detected'],
                'grid_layout': (grid_config['rows'], grid_config['cols']),
                'max_codes': grid_config['max_codes'],
                'original_image': barcode_result['cropped_image']
            }
        else:
            return {
                'decoded_results': {},
                'header_detected': header_result['header_detected'],
                'grid_layout': (grid_config['rows'], grid_config['cols']),
                'max_codes': grid_config['max_codes']
            }
    
    def _detect_with_minimal_improvements(self, cropped_image):
        """Detección con MÍNIMAS mejoras - casi idéntica al original"""
        if self.debug:
            print("  Ejecutando detección conservadora...")
        
        try:
            # PASO 1: Escala de grises (ORIGINAL)
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            
            # PASO 2: Gradientes (ORIGINAL)
            Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            
            # PASO 3: Mapa de energía (ORIGINAL)
            energy_map = np.abs(Ix) - self.energy_factor * np.abs(Iy)
            energy_map = np.clip(energy_map, 0, 255)
            
            # PASO 4: Suavizado (ORIGINAL)
            kernel_size = max(15, min(gray.shape) // 60)
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            smoothed = cv2.filter2D(energy_map, -1, kernel)
            
            # PASO 5: Umbralización (ORIGINAL)
            heatmap_data = smoothed / np.max(smoothed) if np.max(smoothed) > 0 else smoothed
            threshold = np.percentile(heatmap_data, self.threshold_percentile)
            heatmap_selective = np.where(heatmap_data > threshold, heatmap_data, 0)
            
            # PASO 6: Convertir a binario (ORIGINAL)
            binary_heatmap = (heatmap_selective > 0).astype(np.uint8) * 255
            
            # PASO 7: Encontrar contornos (ORIGINAL)
            contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # PASO 8: Filtrado ORIGINAL (sin cambios estrictos)
            valid_regions = []
            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h if h > 0 else 0
                area = cv2.contourArea(cnt)
                
                # CRITERIOS ORIGINALES - no más estrictos
                if (w > 30 and h > 15 and 1.2 < aspect_ratio < 5.0 and area > 400):
                    valid_regions.append((x, y, w, h))
            
            # PASO 9: Ordenamiento ORIGINAL
            if valid_regions:
                ordered_barcodes = self._simple_grid_ordering_original(valid_regions)
            else:
                ordered_barcodes = []
            
            if self.debug:
                print(f"  Códigos detectados: {len(ordered_barcodes)}")
            
            return ordered_barcodes
            
        except Exception as e:
            if self.debug:
                print(f"  Error en gradientes: {e}")
            return []
    
    def _simple_grid_ordering_original(self, regions):
        """Ordenamiento EXACTO como el original - SIN cambios"""
        if not regions:
            return []
        
        # Convertir a centroides
        centroids = []
        for region in regions:
            if len(region) == 4:
                x, y, w, h = region
                cx = x + w/2
                cy = y + h/2
                centroids.append((cx, cy, region))
        
        # Ordenar por Y primero
        centroids.sort(key=lambda c: c[1])
        
        # Agrupar en filas con tolerancia ORIGINAL
        rows = []
        if centroids:
            current_row = [centroids[0]]
            for i in range(1, len(centroids)):
                cx, cy, region = centroids[i]
                last_cy = current_row[-1][1]
                
                if abs(cy - last_cy) < 50:  # Tolerancia ORIGINAL
                    current_row.append(centroids[i])
                else:
                    rows.append(current_row)
                    current_row = [centroids[i]]
            
            if current_row:
                rows.append(current_row)
        
        # Ordenar cada fila por X
        for row in rows:
            row.sort(key=lambda c: c[0])
        
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
    
    # Método de compatibilidad SIN cambios
    def detect_codes_in_grid(self, image, grid_config):
        """Método de compatibilidad - ORIGINAL"""
        ordered_barcodes = self._detect_with_minimal_improvements(image)
        
        result = {
            'decoded_results': {}
        }
        
        for barcode in ordered_barcodes:
            result['decoded_results'][barcode['position']] = {
                'bbox': barcode['bbox'],
                'confidence': barcode['confidence']
            }
        
        return result
    
    def adjust_coordinates_to_original(self, detection_result, crop_info, original_shape, processed_shape):
        """Ajuste de coordenadas - ORIGINAL sin cambios"""
        if not detection_result or not detection_result.get('decoded_results'):
            return detection_result
        
        offset_x = crop_info['offset_x']
        offset_y = crop_info['offset_y']
        
        # Calcular factores de escala
        scale_x = processed_shape[1] / original_shape[1]
        scale_y = processed_shape[0] / original_shape[0]
        
        # Ajustar coordenadas
        adjusted_decoded = {}
        for position, result in detection_result['decoded_results'].items():
            if result is not None:
                adjusted_result = result.copy()
                if result.get('bbox'):
                    x, y, w, h = result['bbox']
                    
                    # Ajustar al espacio de imagen procesada
                    adjusted_x = x + offset_x
                    adjusted_y = y + offset_y
                    
                    # Escalar de vuelta al espacio original
                    final_x = int(adjusted_x / scale_x)
                    final_y = int(adjusted_y / scale_y)
                    final_w = int(w / scale_x)
                    final_h = int(h / scale_y)
                    
                    adjusted_result['bbox'] = (final_x, final_y, final_w, final_h)
                
                adjusted_decoded[position] = adjusted_result
        
        adjusted_results = detection_result.copy()
        adjusted_results['decoded_results'] = adjusted_decoded
        
        return adjusted_results
