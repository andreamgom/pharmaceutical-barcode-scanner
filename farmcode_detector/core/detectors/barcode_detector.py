# core/detectors/barcode_detector.py
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class BarcodeDetector:
    """Detector de zona barcode con márgenes adaptativos según contexto"""
    
    def __init__(self, model_path=None, debug=False):
        self.debug = debug
        self.class_names = {0: 'code', 1: 'header', 2: 'barcode'}
        
        if model_path is None:
            possible_paths = [
                "runs/detect/yolov10_train7/weights/best.pt",
                "../runs/detect/yolov10_train7/weights/best.pt",
                "../../runs/detect/yolov10_train7/weights/best.pt",
                "../../../runs/detect/yolov10_train7/weights/best.pt"
            ]
            
            model_path = None
            for path in possible_paths:
                if Path(path).exists():
                    model_path = path
                    if self.debug:
                        print(f"✅ Modelo encontrado: {model_path}")
                    break
            
            if model_path is None:
                raise FileNotFoundError("No se encontró el modelo YOLOv10")
        else:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        if self.debug:
            print(f"BarcodeDetector inicializando con: {model_path}")
        
        self.model = YOLO(model_path)
    
    def detect_and_crop_barcode_region(self, image, header_info=None, perspective_info=None):
        """Detecta zona barcode con validación temprana de imagen"""
        if self.debug:
            print("PASO 3: Recortando zona de códigos...")
        
        # Ejecutar YOLO para detectar región barcode
        results = self.model(image, conf=0.3, verbose=self.debug)
        barcode_detections = []
        
        boxes = []
        classes = []
        confidences = []
        
        if len(results[0].boxes) == 0:
            if self.debug:
                print("❌ No se detectaron códigos ni regiones barcode - Imagen inválida")
            
            return {
                'cropped_image': None,
                'crop_info': None,
                'barcode_detections': [],
                'detection_success': False,
                'error': 'Imagen inválida: No contiene códigos de barras detectables'
            }
        
        # Si hay detecciones, procesarlas
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # Buscar detecciones de barcode (clase 2)
        for box, cls, conf in zip(boxes, classes, confidences):
            class_name = self.class_names.get(int(cls), f"clase_{int(cls)}")
            if class_name == 'barcode':
                x1, y1, x2, y2 = box.astype(int)
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'area': (x2 - x1) * (y2 - y1)
                }
                barcode_detections.append(detection)
        
        # Si no hay detecciones de barcode, buscar códigos individuales
        if not barcode_detections:
            if self.debug:
                print("⚠️ No se detectó región barcode, buscando códigos individuales...")
            
            # Buscar códigos individuales (clase 0: 'code')
            code_boxes = []
            for box, cls, conf in zip(boxes, classes, confidences):
                if int(cls) == 0:  # Clase 'code'
                    code_boxes.append(box)
            
            if not code_boxes:
                if self.debug:
                    print("❌ No se detectaron códigos individuales - Imagen inválida")
                
                return {
                    'cropped_image': None,
                    'crop_info': None,
                    'barcode_detections': [],
                    'detection_success': False,
                    'error': 'Imagen inválida: No contiene códigos de barras ni códigos individuales'
                }
            
            # Crear región barcode desde códigos individuales
            code_boxes = np.array(code_boxes)
            x1 = int(np.min(code_boxes[:, 0]))
            y1 = int(np.min(code_boxes[:, 1]))
            x2 = int(np.max(code_boxes[:, 2]))
            y2 = int(np.max(code_boxes[:, 3]))
            
            barcode_detections = [{
                'bbox': [x1, y1, x2, y2],
                'confidence': 0.8,
                'area': (x2 - x1) * (y2 - y1)
            }]
            
            if self.debug:
                print(f"✅ Región barcode creada desde {len(code_boxes)} códigos: bbox=({x1},{y1},{x2},{y2})")
        
        # Continuar con procesamiento normal
        if barcode_detections:
            cropped_image, crop_info = self._adaptive_crop_with_context(
                image, barcode_detections, header_info, perspective_info
            )
            
            return {
                'cropped_image': cropped_image,
                'crop_info': crop_info,
                'barcode_detections': barcode_detections,
                'detection_success': True
            }
        else:
            # Este caso ya no debería ocurrir debido a las validaciones anteriores
            return {
                'cropped_image': None,
                'crop_info': None,
                'barcode_detections': [],
                'detection_success': False,
                'error': 'Imagen inválida: Error inesperado en procesamiento'
            }


    def _adaptive_crop_with_context(self, image, barcode_detections, header_info, perspective_info):
        """Recorte con márgenes adaptativos según header y perspectiva"""
        
        # Tomar el barcode con mayor área
        best_barcode = max(barcode_detections, key=lambda x: x['area'])
        x1, y1, x2, y2 = best_barcode['bbox']
        
        if self.debug:
            print(f"Recorte original YOLO: bbox=({x1}, {y1}, {x2}, {y2})")
        
        h, w = image.shape[:2]
        
        # Determinar contexto
        has_header = header_info.get('header_detected', False) if header_info else False
        perspective_corrected = perspective_info.get('correction_applied', False) if perspective_info else False
        
        # LÓGICA DE MÁRGENES ADAPTATIVOS
        if has_header:
            # CON HEADER: márgenes mínimos siempre
            margin_left = 3
            margin_right = 3
            margin_top = 3
            margin_bottom = 8
            layout_type = "con_header"
            reason = "Header detectado - márgenes mínimos"
            
        elif perspective_corrected:
            # SIN HEADER + PERSPECTIVA CORREGIDA: márgenes grandes
            margin_left = 10
            margin_right = 5
            margin_top = 5
            margin_bottom = 65  # MARGEN GRANDE - perspectiva puede haber cortado
            layout_type = "sin_header_perspectiva_corregida"
            reason = "Sin header + perspectiva corregida - márgenes grandes"
            
        else:
            # SIN HEADER + SIN CORRECCIÓN: márgenes moderados
            margin_left = 10
            margin_right = 5
            margin_top = 5
            margin_bottom = 65  # Margen moderado
            layout_type = "sin_header_sin_perspectiva"
            reason = "Sin header + sin corrección - márgenes moderados"
        
        # Aplicar márgenes con límites de imagen
        crop_x1 = max(0, x1 - margin_left)
        crop_y1 = max(0, y1 - margin_top)
        crop_x2 = min(w, x2 + margin_right)
        crop_y2 = min(h, y2 + margin_bottom)
        
        # Recortar imagen
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        crop_info = {
            'offset_x': crop_x1,
            'offset_y': crop_y1,
            'original_bbox': [x1, y1, x2, y2],
            'crop_bbox': [crop_x1, crop_y1, crop_x2, crop_y2],
            'margins': {
                'left': margin_left, 
                'right': margin_right, 
                'top': margin_top, 
                'bottom': margin_bottom
            },
            'layout_type': layout_type,
            'context': {
                'has_header': has_header,
                'perspective_corrected': perspective_corrected,
                'reason': reason
            }
        }
        
        if self.debug:
            print(f"Recorte ({layout_type}): bbox=({crop_x1}, {crop_y1}, {crop_x2}, {crop_y2})")
            print(f"Razón: {reason}")
            print(f"Márgenes: L={margin_left}, R={margin_right}, T={margin_top}, B={margin_bottom}")
        
        return cropped, crop_info
    
    def get_individual_codes_for_validation(self, processed_image):
        """Obtiene códigos YOLO individuales para validación cruzada"""
        
        if self.debug:
            print("   Obteniendo códigos YOLO para validación...")
        
        try:
            results = self.model(processed_image, conf=0.5, verbose=False)
            validation_codes = []
            
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, cls, conf in zip(boxes, classes, confidences):
                    if int(cls) == 0:  # Solo códigos individuales (no header, no barcode region)
                        x1, y1, x2, y2 = box.astype(int)
                        validation_codes.append({
                            'bbox': [x1, y1, x2-x1, y2-y1],  # Convertir a (x, y, w, h)
                            'confidence': float(conf),
                            'area': (x2 - x1) * (y2 - y1)
                        })
            
            if self.debug:
                print(f"   Códigos YOLO para validación: {len(validation_codes)}")
            
            return validation_codes
            
        except Exception as e:
            if self.debug:
                print(f"   Error obteniendo códigos YOLO: {e}")
            return []
