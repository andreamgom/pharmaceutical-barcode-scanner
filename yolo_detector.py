import cv2
import numpy as np
import pandas as pd
from pyzbar import pyzbar
import zxingcpp
from ultralytics import YOLO
from pathlib import Path
import json

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

class YOLODetector:
    def __init__(self, model_path="runs/detect/yolov10_train6/weights/best.pt"):
        """Inicializa el detector YOLO"""
        self.model = YOLO(model_path)
        self.class_names = {0: 'code', 1: 'header', 2: 'barcode'}
        
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

    def determine_grid_layout(self, header_detected):
        """Determina el layout de la cuadrícula según si hay header"""
        if header_detected:
            rows, cols = 6, 4  # Con header: 6x4 = 24 cupones
            max_codes = 24
        else:
            rows, cols = 7, 4  # Sin header: 7x4 = 28 posiciones, pero solo 26 códigos útiles
            max_codes = 26  # Los 2 últimos espacios no se usan
        
        return rows, cols, max_codes

    def get_centroid(self, bbox):
        """Calcula el centroide de un bounding box"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return cx, cy
    
    def get_barcode_bbox_corners(self, detections):
        """Devuelve las 4 esquinas del bounding box de la clase 'barcode'."""
        if not detections['barcode']:
            return None
        bbox = detections['barcode'][0]['bbox']
        x1, y1, x2, y2 = bbox
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]  # TL, TR, BR, BL
        return corners

    def correct_perspective(self, image, pts_src, pts_dst):
        """Corrige la perspectiva de la imagen usando los puntos de origen y destino"""
        M = cv2.getPerspectiveTransform(np.float32(pts_src), np.float32(pts_dst))
        h, w = image.shape[:2]
        corrected = cv2.warpPerspective(image, M, (w, h))
        return corrected

    def sort_detections_by_grid(self, code_detections, rows, cols, max_codes):
        """Ordena las detecciones manteniendo todas las posiciones"""
        if not code_detections:
            return {}
        
        # Calcular centroides
        detections_with_centroids = []
        for det in code_detections:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            detections_with_centroids.append({**det, 'cx': cx, 'cy': cy})
        
        # Ordenar por Y primero
        detections_with_centroids.sort(key=lambda d: d['cy'])
        
        # Agrupación de filas más inteligente
        filas = []
        if len(detections_with_centroids) > 0:
            # Calcular umbral dinámico basado en la distribución real
            y_coords = [d['cy'] for d in detections_with_centroids]
            if len(y_coords) > 1:
                y_diffs = np.diff(sorted(y_coords))
                y_threshold = np.median(y_diffs) * 1.5 if len(y_diffs) > 0 else 50
            else:
                y_threshold = 50
            
            current_row = [detections_with_centroids[0]]
            last_y = detections_with_centroids[0]['cy']

            for det in detections_with_centroids[1:]:
                if abs(det['cy'] - last_y) < y_threshold:
                    current_row.append(det)
                else:
                    filas.append(current_row)
                    current_row = [det]
                last_y = det['cy']
            if current_row:
                filas.append(current_row)

        # Ordenar cada fila por X (columnas)
        for fila in filas:
            fila.sort(key=lambda d: d['cx'])
        
        # Asignación que respeta posiciones vacías
        grid = {}
        
        # Crear todas las posiciones posibles (1 a max_codes)
        for pos in range(1, max_codes + 1):
            grid[pos] = None
        
        # Asignar detecciones a posiciones
        idx = 1
        for fila in filas:
            for det in fila:
                if idx <= max_codes:
                    grid[idx] = det
                    idx += 1
        
        # Filtrar solo posiciones con detecciones reales
        final_grid = {}
        for pos, det in grid.items():
            if det is not None:
                final_grid[pos] = det
        
        return final_grid

    def decode_barcode_hybrid(self, roi):
        """Métodos de decodificación híbridos para códigos de barras"""
        results = []

        # Preprocesado suave
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        normalized = cv2.normalize(enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Variaciones preprocesadas
        preprocessed_images = [
            ("original", gray),
            ("enhanced", enhanced),
            ("normalized", normalized)
        ]
        
        # Binarización adaptativa
        try:
            binary = cv2.adaptiveThreshold(normalized, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
            preprocessed_images.append(("binary", binary))
        except:
            pass
        
        # Imagen invertida
        try:
            inverted = cv2.bitwise_not(normalized)
            preprocessed_images.append(("inverted", inverted))
        except:
            pass

        # Probar pyzbar en todas las variaciones
        for name, img in preprocessed_images:
            try:
                barcodes = pyzbar.decode(img, symbols=[
                    pyzbar.ZBarSymbol.EAN13, 
                    pyzbar.ZBarSymbol.CODE128, 
                    pyzbar.ZBarSymbol.UPCA
                ])
                for barcode in barcodes:
                    data = barcode.data.decode('utf-8')
                    if data.isdigit() and len(data) >= 8:
                        results.append((data, f"pyzbar_{name}", len(data)))
            except:
                pass

        # Probar zxingcpp en todas las variaciones
        for name, img in preprocessed_images:
            try:
                zxing_results = zxingcpp.read_barcodes(img)
                for result in zxing_results:
                    if result.text.isdigit() and len(result.text) >= 8:
                        results.append((result.text, f"zxingcpp_{name}", len(result.text)))
            except:
                pass

        # Mejor selección de resultados
        if results:
            # Priorizar códigos de 13 dígitos
            thirteen_digit_codes = [r for r in results if r[2] == 13]
            if thirteen_digit_codes:
                # Si hay múltiples de 13 dígitos, tomar el más común
                from collections import Counter
                code_counts = Counter([r[0] for r in thirteen_digit_codes])
                most_common_code = code_counts.most_common(1)[0][0]
                best_result = next(r for r in thirteen_digit_codes if r[0] == most_common_code)
                return best_result[0], best_result[1]
            else:
                # Tomar el más largo
                best_result = max(results, key=lambda x: x[2])
                return best_result[0], best_result[1]

        return "No detectado", "none"

    def analyze_yolo_detections(self, results):
        """Analiza las detecciones de YOLO"""
        detections = {'header': [], 'code': [], 'barcode': []}
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.class_names[int(cls)]
                
                detections[class_name].append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(conf),
                    'area': (x2-x1) * (y2-y1)
                })
        
        return detections

    def process_image(self, image_path):
        """Función principal"""

        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image_name = Path(image_path).name
        else:
            image = image_path
            image_name = "unknown_image"
        if image is None:
            return None, "Error: No se pudo cargar la imagen"

        # 1. Detección YOLO
        results = self.model(image, conf=0.25, verbose=False)
        detections = self.analyze_yolo_detections(results)

        # 2. Corrección de perspectiva automática usando bounding box de barcode
        barcode_corners = self.get_barcode_bbox_corners(detections)
        if barcode_corners:
            w, h = image.shape[1], image.shape[0]
            pts_dst = [(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)]
            image = self.correct_perspective(image, barcode_corners, pts_dst)

        # 3. Validar imagen
        valid, message = self.is_image_valid(image)
        if not valid:
            return None, f"Error de validación: {message}"

        # 4. Detección YOLO (de nuevo, sobre la imagen corregida)
        results = self.model(image, conf=0.25, verbose=False)
        detections = self.analyze_yolo_detections(results)

        # 5. Determinar layout (LÓGICA ORIGINAL)
        header_detected = len(detections['header']) > 0
        rows, cols, max_codes = self.determine_grid_layout(header_detected)

        # 6. Organizar códigos en cuadrícula (CON MEJORAS)
        grid = self.sort_detections_by_grid(detections['code'], rows, cols, max_codes)

        # 7. Decodificar códigos
        decoded_results = {}
        decoding_stats = {"pyzbar_original": 0, "pyzbar_enhanced": 0, "pyzbar_normalized": 0, 
                         "pyzbar_binary": 0, "pyzbar_inverted": 0,
                         "zxingcpp_original": 0, "zxingcpp_enhanced": 0, "zxingcpp_normalized": 0,
                         "zxingcpp_binary": 0, "zxingcpp_inverted": 0, "none": 0}
        
        # Procesa TODAS las posiciones de 1 a max_codes
        for position in range(1, max_codes + 1):
            if position in grid:
                detection = grid[position]
                x1, y1, x2, y2 = detection['bbox']
                margin = 20
                roi = image[max(0, y1-margin):min(image.shape[0], y2+margin),
                            max(0, x1-margin):min(image.shape[1], x2+margin)]
                code_value, method = self.decode_barcode_hybrid(roi)
                decoded_results[position] = {
                    'code': code_value,
                    'method': method,
                    'bbox': detection['bbox'],
                    'confidence': float(detection['confidence'])
                }
                decoding_stats[method] += 1
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
        
        return {
            'image_name': image_name,
            'original_image': image,
            'annotated_image': results[0].plot(),
            'detections': detections,
            'header_detected': bool(header_detected),
            'grid_layout': (rows, cols),
            'max_codes': max_codes,
            'decoded_results': decoded_results,
            'valid_codes': valid_codes,
            'decoding_stats': decoding_stats,
            'success_rate': float(valid_codes / max_codes)
        }, None

    def save_results_to_json(self, results, base_dir='data/predictions/yolo'):
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

# ===== FUNCIONES AUXILIARES =====

def process_multiple_images(detector, image_dir, max_images=None, save_json=True):
    """Procesa múltiples imágenes y devuelve resultados"""
    image_paths = list(Path(image_dir).glob("*.jpg"))
    if max_images:
        image_paths = image_paths[:max_images]
    
    results_list = []
    for img_path in image_paths:
        print(f"Procesando: {img_path.name}")
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
        # Agrupar estadísticas de decodificación
        pyzbar_total = sum(result['decoding_stats'].get(k, 0) for k in result['decoding_stats'] if k.startswith('pyzbar'))
        zxingcpp_total = sum(result['decoding_stats'].get(k, 0) for k in result['decoding_stats'] if k.startswith('zxingcpp'))
        
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

def compare_yolo_with_ground_truth(image_name, use_simple_format=True):
    """Compara resultados YOLO con ground truth"""
    
    image_name_clean = image_name.replace('.jpg', '')
    
    if use_simple_format:
        yolo_path = Path(f"data/predictions/yolo/{image_name_clean}_simple.json")
        gt_path = Path(f"data/ground_truth/{image_name_clean}_simple.json")
        
        if not yolo_path.exists() or not gt_path.exists():
            return None, f"Archivos simples no encontrados: {yolo_path} o {gt_path}"
        
        with open(yolo_path, 'r') as f:
            yolo_codes = json.load(f)
        
        with open(gt_path, 'r') as f:
            gt_codes = json.load(f)
        
        comparisons = []
        exact_matches = 0
        total_positions = min(len(yolo_codes), len(gt_codes))
        
        for i in range(total_positions):
            gt_code = str(gt_codes[i]).strip()
            yolo_code = str(yolo_codes[i]).strip()
            
            match = gt_code == yolo_code
            if match:
                exact_matches += 1
            
            comparisons.append({
                'position': i + 1,
                'ground_truth': gt_code,
                'yolo_detected': yolo_code,
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

def create_ground_truth_json(image_name, codes_list, base_dir='data/ground_truth'):
    """Crea un JSON de ground truth"""
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    ground_truth_data = {
        'image_name': image_name,
        'codes': {}
    }
    
    for i, code in enumerate(codes_list, 1):
        ground_truth_data['codes'][str(i)] = str(code).strip()
    
    filename_complete = f"{image_name.replace('.jpg', '_complete.json')}"
    filepath_complete = Path(base_dir) / filename_complete
    
    with open(filepath_complete, 'w') as f:
        json.dump(ground_truth_data, f, indent=2)
    
    filename_simple = f"{image_name.replace('.jpg', '_simple.json')}"
    filepath_simple = Path(base_dir) / filename_simple
    
    with open(filepath_simple, 'w') as f:
        json.dump(codes_list, f, indent=2)
    
    return str(filepath_complete), str(filepath_simple)
