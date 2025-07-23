import cv2
import numpy as np
import pandas as pd
from pyzbar import pyzbar
import zxingcpp
from pathlib import Path
import json
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

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

# COPIAR EXACTAMENTE TUS CLASES DE PASTE.TXT
class FolioDetector:
    """Clase para detectar y corregir la perspectiva del folio/formulario"""
    
    def __init__(self):
        self.debug = True
    
    def detect_folio_contour(self, image):
        """Detecta el contorno del folio/formulario en la imagen"""
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar blur para reducir ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detectar bordes con múltiples umbrales para mayor robustez
        edges1 = cv2.Canny(blurred, 50, 150)
        edges2 = cv2.Canny(blurred, 30, 100)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Operaciones morfológicas para conectar bordes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área y forma
        folio_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # El folio debe tener un área significativa (al menos 20% de la imagen)
            min_area = 0.2 * image.shape[0] * image.shape[1]
            max_area = 0.9 * image.shape[0] * image.shape[1]
            
            if min_area < area < max_area:
                # Aproximar el contorno a un polígono
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Buscar contornos con 4 vértices (rectangulares)
                if len(approx) == 4:
                    folio_candidates.append((area, approx))
        
        # Ordenar por área (el folio debería ser el más grande)
        folio_candidates.sort(key=lambda x: x[0], reverse=True)
        
        if folio_candidates:
            return folio_candidates[0][1].reshape(4, 2).astype(np.float32)
        
        return None
    
    def detect_folio_by_content(self, image):
        """Detecta el folio basándose en el contenido (códigos de barras)"""
        
        # Usar el detector de mapa de calor para encontrar códigos de barras
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calcular gradientes para detectar códigos de barras
        Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        # Mapa de energía enfocado en códigos de barras
        energy_map = np.abs(Ix) - 0.5 * np.abs(Iy)
        energy_map = np.clip(energy_map, 0, 255).astype(np.uint8)
        
        # Suavizado
        kernel = np.ones((31, 31), np.float32) / (31 * 31)
        smoothed = cv2.filter2D(energy_map, -1, kernel)
        
        # Umbralización
        thresh_val = threshold_otsu(smoothed)
        binary = (smoothed > thresh_val * 0.7).astype(np.uint8) * 255
        
        # Operaciones morfológicas para conectar regiones
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 20))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large)
        
        # Encontrar el contorno que engloba todas las regiones de códigos
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Encontrar el rectángulo que engloba todas las detecciones
            all_points = np.vstack(contours)
            rect = cv2.minAreaRect(all_points)
            
            # Convertir a 4 puntos
            box = cv2.boxPoints(rect)
            return self.order_points(box.astype(np.float32))
        
        return None
    
    def order_points(self, pts):
        """Ordena los puntos en el orden: superior-izquierda, superior-derecha, inferior-derecha, inferior-izquierda"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Suma y diferencia de coordenadas
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Superior-izquierda: suma mínima
        rect[0] = pts[np.argmin(s)]
        # Inferior-derecha: suma máxima
        rect[2] = pts[np.argmax(s)]
        # Superior-derecha: diferencia mínima
        rect[1] = pts[np.argmin(diff)]
        # Inferior-izquierda: diferencia máxima
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def calculate_destination_size(self, pts):
        """Calcula el tamaño óptimo para el folio corregido"""
        (tl, tr, br, bl) = pts
        
        # Calcular ancho
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        # Calcular alto
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        return maxWidth, maxHeight
    
    def detect_and_correct_folio(self, image):
        """Detecta automáticamente el folio y corrige su perspectiva"""
        
        print("Detectando folio en la imagen...")
        
        # Método 1: Detección por contornos
        folio_points = self.detect_folio_contour(image)
        
        if folio_points is None:
            print("Método 1 falló, probando detección por contenido...")
            # Método 2: Detección por contenido de códigos de barras
            folio_points = self.detect_folio_by_content(image)
        
        if folio_points is None:
            print("No se pudo detectar automáticamente el folio")
            return None
        
        print("Folio detectado automáticamente")
        
        # Ordenar puntos
        folio_points = self.order_points(folio_points)
        
        # Calcular tamaño de destino
        maxWidth, maxHeight = self.calculate_destination_size(folio_points)
        
        # Puntos de destino (rectángulo perfecto)
        dst_points = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)
        
        # Calcular matriz de transformación
        M = cv2.getPerspectiveTransform(folio_points, dst_points)
        
        # Aplicar transformación
        corrected_folio = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return corrected_folio, M, folio_points, dst_points

class BarcodePreprocessor:
    """Clase para aplicar diferentes técnicas de preprocesamiento"""
    
    def enhance_contrast(self, image):
        """Mejora el contraste usando CLAHE limitado"""
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return clahe.apply(image)
    
    def adjust_brightness(self, image, value):
        """Ajusta el brillo de la imagen"""
        if value > 0:
            shadow = value
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + value
        
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        
        return cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    
    def apply_clahe(self, image):
        """Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)
    
    def sharpen_image(self, image):
        """Aplica filtro de nitidez"""
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    def adaptive_threshold(self, image):
        """Aplica umbralización adaptativa"""
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    def otsu_threshold(self, image):
        """Aplica umbralización de Otsu"""
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    def morphology_operations(self, image):
        """Aplica operaciones morfológicas"""
        kernel = np.ones((2,2), np.uint8)
        # Primero cerrar para conectar líneas
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        # Luego abrir para limpiar ruido
        return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    def get_all_preprocessing_techniques(self, roi_gray):
        """Retorna todas las técnicas de preprocesamiento aplicadas"""
        preprocessing_techniques = [
            ("original", roi_gray),
            ("contrast", self.enhance_contrast(roi_gray)),
            ("brightness_up", self.adjust_brightness(roi_gray, 30)),
            ("brightness_down", self.adjust_brightness(roi_gray, -30)),
            ("clahe", self.apply_clahe(roi_gray)),
            ("gaussian_blur", cv2.GaussianBlur(roi_gray, (3, 3), 0)),
            ("median_blur", cv2.medianBlur(roi_gray, 3)),
            ("bilateral", cv2.bilateralFilter(roi_gray, 9, 75, 75)),
            ("sharpen", self.sharpen_image(roi_gray)),
            ("adaptive_thresh", self.adaptive_threshold(roi_gray)),
            ("otsu_thresh", self.otsu_threshold(roi_gray)),
            ("morphology", self.morphology_operations(roi_gray)),
            ("resize_2x", cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
            ("resize_3x", cv2.resize(roi_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)),
            ("resize_4x", cv2.resize(roi_gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)),
            ("resize_5x", cv2.resize(roi_gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)),
            ("resize_6x", cv2.resize(roi_gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC))
        ]
        return preprocessing_techniques

# ===== NUEVAS FUNCIONES DE MERGE INTELIGENTE =====

def merge_rectangles_by_layout_constraints(valid_regions, max_codes_per_row=[4,4,4,4,4,4,2]):
    """
    Une rectángulos basándose en las restricciones del layout:
    - Filas 1-6: máximo 4 códigos cada una
    - Fila 7: máximo 2 códigos
    """
    
    if len(valid_regions) <= 1:
        return valid_regions
    
    print(f"Aplicando merge inteligente basado en layout...")
    print(f"   Regiones iniciales: {len(valid_regions)}")
    
    # Calcular centroides y agrupar por filas
    centroids = []
    for x, y, w, h in valid_regions:
        cx = x + w/2
        cy = y + h/2
        centroids.append((cx, cy, (x, y, w, h)))
    
    # Agrupar por filas (threshold_row = 50 como en tu código original)
    threshold_row = 35
    rows = []
    
    for cx, cy, rect in sorted(centroids, key=lambda c: c[1]):
        placed = False
        for row in rows:
            row_y_avg = np.mean([c[1] for c in row])
            if abs(row_y_avg - cy) < threshold_row:
                row.append((cx, cy, rect))
                placed = True
                break
        if not placed:
            rows.append([(cx, cy, rect)])
    
    # Ordenar filas por Y
    rows.sort(key=lambda row: np.mean([c[1] for c in row]))
    
    print(f"   Filas detectadas: {len(rows)}")
    for i, row in enumerate(rows, 1):
        print(f"     Fila {i}: {len(row)} códigos")
    
    # APLICAR MERGE INTELIGENTE FILA POR FILA
    merged_regions = []
    
    for row_idx, row in enumerate(rows):
        if row_idx >= len(max_codes_per_row):
            max_codes_in_row = 2  # Por defecto para filas extra
        else:
            max_codes_in_row = max_codes_per_row[row_idx]
        
        print(f"   🔧 Procesando fila {row_idx + 1}: {len(row)} códigos → max permitido: {max_codes_in_row}")
        
        if len(row) <= max_codes_in_row:
            # No hay exceso, mantener como está
            for cx, cy, rect in row:
                merged_regions.append(rect)
            print(f"     Fila {row_idx + 1}: Sin merge necesario")
        else:
            # HAY EXCESO: Aplicar merge inteligente
            print(f"     Fila {row_idx + 1}: Aplicando merge ({len(row)} → {max_codes_in_row})")
            
            # Ordenar por X dentro de la fila
            row.sort(key=lambda c: c[0])
            
            # Aplicar merge agresivo para reducir a max_codes_in_row
            merged_row = merge_row_to_target_count(row, max_codes_in_row)
            
            for rect in merged_row:
                merged_regions.append(rect)
            
            print(f"     Fila {row_idx + 1}: Merge completado ({len(row)} → {len(merged_row)})")
    
    print(f"Merge inteligente completado: {len(valid_regions)} → {len(merged_regions)} regiones")
    
    return merged_regions

def merge_row_to_target_count(row, target_count):
    """Reduce una fila a exactamente target_count códigos mediante merge inteligente"""
    if len(row) <= target_count:
        return [rect for cx, cy, rect in row]

    # Extraer solo los rectángulos
    rectangles = [rect for cx, cy, rect in row]
    
    # Mientras tengamos más códigos de los permitidos, hacer merge
    while len(rectangles) > target_count:
        # Encontrar el par de rectángulos más cercanos horizontalmente
        min_distance = float('inf')
        merge_idx = 0
        
        for i in range(len(rectangles) - 1):
            x1, y1, w1, h1 = rectangles[i]
            x2, y2, w2, h2 = rectangles[i + 1]
            
            # MEJORAR: Calcular distancia entre bordes, no centros
            distance = x2 - (x1 + w1)  # Distancia horizontal entre rectángulos
            
            # AÑADIR: Verificar que estén realmente alineados
            center_y1 = y1 + h1/2
            center_y2 = y2 + h2/2
            y_alignment = abs(center_y1 - center_y2)
            max_height = max(h1, h2)
            
            # Solo considerar para merge si están bien alineados
            if y_alignment < max_height * 0.4 and distance < min_distance:
                min_distance = distance
                merge_idx = i
        
        # Merge los dos rectángulos más cercanos y alineados
        rect1 = rectangles[merge_idx]
        rect2 = rectangles[merge_idx + 1]
        merged_rect = merge_two_rectangles(rect1, rect2)
        
        # Reemplazar los dos rectángulos por el merged
        rectangles = (rectangles[:merge_idx] + 
                     [merged_rect] + 
                     rectangles[merge_idx + 2:])
        
        print(f"Merged rectángulos en posiciones {merge_idx} y {merge_idx + 1} (distancia: {min_distance:.1f}px)")

    return rectangles


def merge_two_rectangles(rect1, rect2):
    """Une dos rectángulos de manera controlada para cupones farmacéuticos"""
    return merge_rectangles_horizontal_aware(rect1, rect2)

def validate_merge_quality(original_rects, merged_rect):
    """Valida que el merge no sea demasiado agresivo"""
    mx, my, mw, mh = merged_rect
    merged_area = mw * mh
    
    # Calcular área total de rectángulos originales
    total_original_area = sum(w * h for x, y, w, h in original_rects)
    
    # El merge no debería ser más de 2.5x el área original
    if merged_area > total_original_area * 2.5:
        return False, "Merge demasiado grande"
    
    # Verificar que el aspect ratio no sea extremo
    aspect_ratio = mw / mh if mh > 0 else 0
    if aspect_ratio > 8.0 or aspect_ratio < 0.5:
        return False, "Aspect ratio extremo"
    
    return True, "Merge válido"



def merge_rectangles_horizontal_aware(rect1, rect2):
    """Merge que considera si los rectángulos están alineados horizontalmente"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    # Calcular centros Y
    center_y1 = y1 + h1/2
    center_y2 = y2 + h2/2
    
    # Verificar si están en la misma fila (alineados horizontalmente)
    max_height = max(h1, h2)
    y_difference = abs(center_y1 - center_y2)
    
    if y_difference < max_height * 0.5:  # Están en la misma fila
        # Merge horizontal conservador
        min_x = min(x1, x2)
        max_x = max(x1 + w1, x2 + w2)
        
        # Para Y, usar el rango que cubra ambos pero sin exceso
        min_y = min(y1, y2)
        max_y = max(y1 + h1, y2 + h2)
        
        # Limitar la altura para evitar rectángulos demasiado altos
        height = max_y - min_y
        if height > max_height * 1.3:  # Máximo 30% más alto
            # Usar altura promedio centrada
            avg_height = (h1 + h2) / 2
            center_y = (center_y1 + center_y2) / 2
            min_y = int(center_y - avg_height/2)
            height = int(avg_height)
        
        result_rect = (min_x, min_y, max_x - min_x, height)
    else:
        # No están alineados horizontalmente, merge muy conservador
        padding = 5
        min_x = min(x1 - padding, x2 - padding)
        min_y = min(y1 - padding, y2 - padding)
        max_x = max(x1 + w1 + padding, x2 + w2 + padding)
        max_y = max(y1 + h1 + padding, y2 + h2 + padding)
        
        result_rect = (min_x, min_y, max_x - min_x, max_y - min_y)
    
    # VALIDAR EL RESULTADO ANTES DE DEVOLVERLO
    valid, message = validate_merge_quality([rect1, rect2], result_rect)
    if not valid:
        print(f"⚠️ Merge rechazado: {message}, usando merge conservador")
        # Si el merge no es válido, usar merge conservador
        padding = 3
        return (min(rect1[0], rect2[0]) - padding, 
                min(rect1[1], rect2[1]) - padding,
                max(rect1[0] + rect1[2], rect2[0] + rect2[2]) - min(rect1[0], rect2[0]) + 2*padding,
                max(rect1[1] + rect1[3], rect2[1] + rect2[3]) - min(rect1[1], rect2[1]) + 2*padding)
    
    return result_rect

def validate_layout_constraints(ordered_barcodes, max_codes_per_row=[4,4,4,4,4,4,2]):
    """
    Valida que el layout cumpla las restricciones y reporta problemas
    """
    
    # Agrupar por filas
    threshold_row = 50
    rows = []
    
    for barcode in ordered_barcodes:
        num = barcode['position']
        cx, cy = barcode['centroid']
        rect = barcode['bbox']
        
        placed = False
        for row in rows:
            row_y_avg = np.mean([item['centroid'][1] for item in row])  # cy promedio
            if abs(row_y_avg - cy) < threshold_row:
                row.append(barcode)
                placed = True
                break
        if not placed:
            rows.append([barcode])
    
    rows.sort(key=lambda row: np.mean([item['centroid'][1] for item in row]))
    
    print(f"\nVALIDACIÓN DEL LAYOUT:")
    print(f"   Total filas: {len(rows)}")
    
    layout_valid = True
    
    for i, row in enumerate(rows):
        max_allowed = max_codes_per_row[i] if i < len(max_codes_per_row) else 2
        
        if len(row) > max_allowed:
            print(f"Fila {i+1}: {len(row)} códigos (máximo: {max_allowed})")
            layout_valid = False
        else:
            print(f"Fila {i+1}: {len(row)} códigos (máximo: {max_allowed})")
    
    return layout_valid, rows

# ===== CLASE GRADIENT DETECTOR MEJORADA =====

class GradientDetector:
    """Detector de códigos de barras usando gradientes - EXACTAMENTE COMO PASTE.TXT CON MEJORAS"""
    
    def __init__(self):
        self.debug = True
        # AÑADIR: Detector de folio como en paste.txt
        self.folio_detector = FolioDetector()
        self.preprocessor = BarcodePreprocessor()
    
    def is_image_valid(self, image):
        """Verifica si la imagen tiene calidad suficiente para procesar códigos de barras"""
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
    
    def determine_grid_layout(self, num_detected_codes):
        """Determina el layout de la cuadrícula según el número de códigos detectados"""
        if num_detected_codes <= 24:
            rows, cols = 6, 4  # Con header: 6x4 = 24 cupones
            max_codes = 24
        else:
            rows, cols = 7, 4  # Sin header: 7x4 = 28 posiciones, pero solo 26 códigos útiles
            max_codes = 26
        
        return rows, cols, max_codes
    
    def detect_and_order_barcodes_like_paste(self, image_path):
        """Detecta y ordena códigos de barras en una imagen, siguiendo el enfoque de paste.txt pero con mejoras."""
        
        # Leer imagen (como en paste.txt)
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        else:
            img = image_path
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calcular gradientes (EXACTO DE PASTE.TXT)
        Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        # USAR PARÁMETROS FIJOS DE PASTE.TXT (NO ADAPTATIVOS)
        energy_factor = 0.7  # FIJO como en paste.txt
        threshold_percentile = 92  # FIJO como en paste.txt
        
        # Mapa de energía (EXACTO DE PASTE.TXT)
        energy_map = np.abs(Ix) - energy_factor * np.abs(Iy)
        energy_map = np.clip(energy_map, 0, 255)
        
        # Suavizado (EXACTO DE PASTE.TXT)
        kernel_size = max(15, min(gray.shape) // 60)
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        smoothed = cv2.filter2D(energy_map, -1, kernel)
        
        # Umbralización (EXACTO DE PASTE.TXT)
        heatmap_data = smoothed / np.max(smoothed) if np.max(smoothed) > 0 else smoothed
        threshold = np.percentile(heatmap_data, threshold_percentile)
        heatmap_selective = np.where(heatmap_data > threshold, heatmap_data, 0)
        
        # Convertir a binario
        binary_heatmap = (heatmap_selective > 0).astype(np.uint8) * 255
        
        # Encontrar contornos
        contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # FILTRAR CON CRITERIOS FIJOS DE PASTE.TXT (NO ADAPTATIVOS)
        valid_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            area = cv2.contourArea(cnt)
            
            # CRITERIOS EXACTOS DE PASTE.TXT
            if (w > 30 and h > 15 and 
                1.2 < aspect_ratio < 5.0 and 
                area > 400):
                valid_regions.append((x, y, w, h))
        
        print(f"Regiones detectadas inicialmente: {len(valid_regions)}")
        
        
        # ===== ORDENAMIENTO SIMPLE PRIMERO =====
        ordered_barcodes = self.simple_grid_ordering(valid_regions)
        
        print(f"Después del ordenamiento simple: {len(ordered_barcodes)} códigos")
        
        # ===== VERIFICAR SI NECESITA MERGE =====
        needs_merge = self.check_if_merge_needed(ordered_barcodes)

        if needs_merge:
            print("Detectadas inconsistencias, aplicando merge...")
            # SOLO AHORA aplicar merge
            valid_regions_for_merge = [barcode['bbox'] for barcode in ordered_barcodes]
            merged_regions = merge_rectangles_by_layout_constraints(
                valid_regions_for_merge,
                max_codes_per_row=[4, 4, 4, 4, 4, 4, 2]
            )
            # Reordenar después del merge
            ordered_barcodes = self.simple_grid_ordering(merged_regions)
            print(f"Después del merge: {len(ordered_barcodes)} códigos")
        else:
            print("Ordenamiento simple suficiente, no se necesita merge")

        # Convertir ordered_barcodes al formato que espera el resto del código
        valid_regions_merged = [barcode['bbox'] for barcode in ordered_barcodes]
        
        return ordered_barcodes
    
    def group_by_rows(self, ordered_barcodes):
        """Agrupa códigos por filas"""
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



    def check_if_merge_needed(self, ordered_barcodes):
        """Verifica si se necesita merge basado en inconsistencias"""
        
        # Criterio 1: Demasiados códigos (más de 26)
        if len(ordered_barcodes) > 26:
            print(f"\nCriterio 1: Demasiados códigos ({len(ordered_barcodes)} > 26)")
            return True
        
        # Criterio 2: Filas con demasiados códigos
        rows = self.group_by_rows(ordered_barcodes)
        max_codes_per_row = [4, 4, 4, 4, 4, 4, 2]
        
        for i, row in enumerate(rows):
            expected_count = max_codes_per_row[i] if i < len(max_codes_per_row) else 4
            if len(row) > expected_count:
                print(f"\nCriterio 2: Fila {i+1} tiene {len(row)} códigos (esperado: {expected_count})")
                return True
        
        # Criterio 3: Códigos muy pequeños (posibles fragmentos)
        small_codes = [b for b in ordered_barcodes if b['bbox'][2] * b['bbox'][3] < 300]
        if len(small_codes) > len(ordered_barcodes) * 0.3:  # Más del 30% son pequeños
            print(f"\nCriterio 3: Muchos códigos pequeños ({len(small_codes)}/{len(ordered_barcodes)})")
            return True
        
        print(" No se detectaron inconsistencias")
        return False

    def simple_grid_ordering(self, regions):
        """Ordenamiento simple en grid 6x4 sin merge"""
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
        
        # print("DEBUG - Orden asignado:")
        for barcode in ordered_barcodes[:10]:  # Primeros 10
            x, y, w, h = barcode['bbox']
            # print(f"Pos {barcode['position']}: centro=({barcode['centroid'][0]:.0f}, {barcode['centroid'][1]:.0f})")
        return ordered_barcodes
    
    def decode_with_preprocessing(self, roi_gray):
        """Decodifica usando múltiples técnicas de preprocesamiento (DE PASTE.TXT)"""
        
        preprocessing_techniques = self.preprocessor.get_all_preprocessing_techniques(roi_gray)
        
        results = {}
        
        for technique_name, processed_image in preprocessing_techniques:
            # Intentar con pyzbar
            pyzbar_result = None
            try:
                decoded_objects = pyzbar.decode(processed_image)
                if decoded_objects:
                    pyzbar_result = {
                        'data': decoded_objects[0].data.decode('utf-8'),
                        'type': decoded_objects[0].type,
                        'technique': technique_name
                    }
            except Exception:
                pass
            
            # Intentar con zxing-cpp
            zxingcpp_result = None
            try:
                zxing_results = zxingcpp.read_barcodes(processed_image)
                if zxing_results:
                    result = zxing_results[0]
                    zxingcpp_result = {
                        'data': result.text,
                        'type': result.format.name,
                        'technique': technique_name
                    }
            except Exception:
                pass
            
            # Guardar resultados si hay éxito
            if pyzbar_result or zxingcpp_result:
                results[technique_name] = {
                    'pyzbar': pyzbar_result,
                    'zxingcpp': zxingcpp_result,
                    'processed_image': processed_image
                }
        
        return results
    
    def process_image(self, image_path):
        """Función principal que procesa una imagen COMO PASTE.TXT"""
        
        if isinstance(image_path, str):
            original_image = cv2.imread(image_path)
            image_name = Path(image_path).name
        else:
            original_image = image_path
            image_name = "unknown_image"
        
        if original_image is None:
            return None, "Error: No se pudo cargar la imagen"
        
        # PASO 1: CORRECCIÓN DE PERSPECTIVA (COMO EN PASTE.TXT)
        result = self.folio_detector.detect_and_correct_folio(original_image)
        
        if result is not None:
            corrected_folio, M, folio_points, dst_points = result
            print("Folio corregido")
            image = corrected_folio  # USAR IMAGEN CORREGIDA
        else:
            print(" No se detectó folio, usando imagen original")
            image = original_image
        
        # PASO 2: VALIDAR IMAGEN
        valid, message = self.is_image_valid(image)
        if not valid:
            return None, f"Error de validación: {message}"
        
        # PASO 3: DETECTAR CÓDIGOS USANDO LÓGICA EXACTA DE PASTE.TXT CON MEJORAS
        detected_codes = self.detect_and_order_barcodes_like_paste(image)
        
        if not detected_codes:
            return None, "No se detectaron códigos de barras"
        
        # PASO 4: DETERMINAR LAYOUT
        rows, cols, max_codes = self.determine_grid_layout(len(detected_codes))
        
        # PASO 5: DECODIFICAR CÓDIGOS (COMO EN PASTE.TXT)
        decoded_results = {}
        
        # DICCIONARIO COMPLETO DE ESTADÍSTICAS (IGUAL QUE PASTE.TXT)
        decoding_stats = {
            "original": 0, "contrast": 0, "brightness_up": 0, "brightness_down": 0,
            "clahe": 0, "gaussian_blur": 0, "median_blur": 0, "bilateral": 0,
            "sharpen": 0, "adaptive_thresh": 0, "otsu_thresh": 0, "morphology": 0,
            "resize_2x": 0, "resize_3x": 0, "resize_4x": 0, "resize_5x": 0, "resize_6x": 0,
            "none": 0
        }
        
        # Procesar TODAS las posiciones de 1 a max_codes
        for position in range(1, max_codes + 1):
            # Buscar si hay una detección para esta posición
            detection = next((d for d in detected_codes if d['position'] == position), None)
            
            if detection:
                x, y, w, h = detection['bbox']
                margin = 10
                roi = image[max(0, y-margin):min(image.shape[0], y+h+margin),
                           max(0, x-margin):min(image.shape[1], x+w+margin)]
                
                if roi.size > 0:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()
                    preprocessing_results = self.decode_with_preprocessing(roi_gray)
                    
                    if preprocessing_results:
                        # Tomar el primer resultado exitoso
                        first_technique = list(preprocessing_results.keys())[0]
                        res = preprocessing_results[first_technique]
                        
                        if res['pyzbar']:
                            code_value = res['pyzbar']['data']
                            method = first_technique  # USAR NOMBRE DE TÉCNICA DIRECTAMENTE
                        elif res['zxingcpp']:
                            code_value = res['zxingcpp']['data']
                            method = first_technique  # USAR NOMBRE DE TÉCNICA DIRECTAMENTE
                        else:
                            code_value = "No detectado"
                            method = "none"
                    else:
                        code_value = "No detectado"
                        method = "none"
                    
                    decoded_results[position] = {
                        'code': code_value,
                        'method': method,
                        'bbox': detection['bbox'],
                        'confidence': float(detection['confidence'])
                    }
                    
                    # INCREMENTAR ESTADÍSTICA CORRECTA
                    if method in decoding_stats:
                        decoding_stats[method] += 1
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
                # Posición sin detección
                decoded_results[position] = {
                    'code': "No detectado",
                    'method': "none",
                    'bbox': None,
                    'confidence': 0.0
                }
                decoding_stats["none"] += 1
        
        valid_codes = sum(1 for r in decoded_results.values() if r['code'] != "No detectado")
        
        # PASO 6: CREAR IMAGEN ANOTADA
        annotated_image = image.copy()
        for detection in detected_codes:
            x, y, w, h = detection['bbox']
            cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(annotated_image, str(detection['position']), 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return {
            'image_name': image_name,
            'original_image': original_image,
            'annotated_image': annotated_image,
            'detections': {'code': detected_codes, 'header': [], 'barcode': []},
            'header_detected': False,  # Gradient detector no detecta headers
            'grid_layout': (rows, cols),
            'max_codes': max_codes,
            'decoded_results': decoded_results,
            'valid_codes': valid_codes,
            'decoding_stats': decoding_stats,
            'success_rate': float(valid_codes / max_codes)
        }, None
    
    def save_results_to_json(self, results, base_dir='data/predictions/gradient'):
        """Guarda los resultados en formato JSON completo y simple"""
        
        # Crear directorio si no existe
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        
        image_name_clean = results['image_name'].replace('.jpg', '')
        
        # JSON COMPLETO
        json_data_complete = {
            'image_name': results['image_name'],
            'header_detected': bool(results['header_detected']),
            'grid_layout': {
                'rows': int(results['grid_layout'][0]),
                'cols': int(results['grid_layout'][1])
            },
            'max_codes': int(results['max_codes']),
            'valid_codes': int(results['valid_codes']),
            'success_rate': float(results['success_rate']),
            'decoding_stats': results['decoding_stats'],
            'codes': {}
        }
        
        # JSON SIMPLE
        codes_list = []
        
        # Llenar ambos formatos manteniendo TODAS las posiciones
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
    
    def validate_and_correct_ean13(self, code):
        """Valida y corrige códigos EAN-13 usando patrones conocidos"""
        if len(code) != 13:
            return code, False
        

def apply_spanish_ean13_patterns(self, code):
    """Aplica patrones conocidos de EAN-13 españoles"""
    if len(code) != 13:
        return code
    
    # Patrones españoles conocidos
    spanish_patterns = ['847', '840', '841', '842', '843', '844', '845', '846', '848', '849']
    
    # Si no empieza por patrón español, intentar corregir
    if not any(code.startswith(pattern) for pattern in spanish_patterns):
        # Intentar con 847 (más común)
        if code.startswith('641') or code.startswith('647'):
            # Posible confusión 6→8, 41→47
            candidate = '847' + code[3:]
            corrected, valid = self.validate_and_correct_ean13(candidate)
            if valid:
                print(f"   Patrón español aplicado: {code} → {corrected}")
                return corrected
    
    return code


# ===== FUNCIONES AUXILIARES (IGUALES A YOLO) =====

def process_multiple_images(detector, image_dir, max_images=None, save_json=True):
    """Procesa múltiples imágenes y devuelve resultados"""
    image_paths = list(Path(image_dir).glob("*.jpg"))
    if max_images:
        image_paths = image_paths[:max_images]
    
    results_list = []
    for img_path in image_paths:
        print(f"\nProcesando: {img_path.name}")
        result, error = detector.process_image(str(img_path))
        
        if result:
            results_list.append(result)
            if save_json:
                json_complete, json_simple = detector.save_results_to_json(result)
                print(f"  → JSON completo: {json_complete}")
                print(f"  → JSON simple: {json_simple}")
        else:
            print(f"  → Error: {error}")
    
    return results_list

def create_summary_dataframe(results_list):
    """Crea un DataFrame resumen de los resultados"""
    summary_data = []
    
    for result in results_list:
        # Agrupar estadísticas de decodificación CORRECTAMENTE
        pyzbar_total = sum(result['decoding_stats'].get(k, 0) for k in result['decoding_stats'] 
                          if k != 'none')  # Todo excepto 'none'
        zxingcpp_total = 0  # Gradient detector usa técnicas, no separación pyzbar/zxingcpp
        
        summary_data.append({
            'image_name': result['image_name'],
            'header_detected': bool(result['header_detected']),
            'valid_codes': int(result['valid_codes']),
            'max_codes': int(result['max_codes']),
            'success_rate': float(result['success_rate']),
            'pyzbar_codes': pyzbar_total,
            'zxingcpp_codes': zxingcpp_total,
            'failed_codes': int(result['decoding_stats'].get('none', 0))
        })
    
    return pd.DataFrame(summary_data)

def compare_gradient_with_ground_truth(image_name, use_simple_format=True):
    """Compara resultados Gradient con ground truth"""
    image_name_clean = image_name.replace('.jpg', '')
    
    if use_simple_format:
        gradient_path = Path(f"data/predictions/gradient/{image_name_clean}_simple.json")
        gt_path = Path(f"data/ground_truth/{image_name_clean}_simple.json")
        
        if not gradient_path.exists() or not gt_path.exists():
            return None, f"Archivos simples no encontrados: {gradient_path} o {gt_path}"
        
        with open(gradient_path, 'r') as f:
            gradient_codes = json.load(f)
        
        with open(gt_path, 'r') as f:
            gt_codes = json.load(f)
        
        comparisons = []
        exact_matches = 0
        total_positions = min(len(gradient_codes), len(gt_codes))
        
        for i in range(total_positions):
            gt_code = str(gt_codes[i]).strip()
            gradient_code = str(gradient_codes[i]).strip()
            match = gt_code == gradient_code
            
            if match:
                exact_matches += 1
            
            comparisons.append({
                'position': i + 1,
                'ground_truth': gt_code,
                'gradient_detected': gradient_code,
                'match': match
            })
        
        accuracy = exact_matches / total_positions if total_positions > 0 else 0
        
        return {
            'image_name': image_name,
            'total_positions': total_positions,
            'exact_matches': exact_matches,
            'accuracy': accuracy,
            'comparisons': comparisons
        }, None
