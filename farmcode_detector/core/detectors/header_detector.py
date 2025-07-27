# core/detectors/header_detector.py
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class HeaderDetector:
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
            
            for path in possible_paths:
                if Path(path).exists():
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError("No se encontró el modelo YOLOv10")
        
        self.model = YOLO(model_path)
    
    def detect_header(self, image):
        """Detección solo YOLO con alta confianza"""
        # Solo YOLO con confianza MUY alta
        results = self.model(image, conf=0.87, verbose=False)
        
        header_detected = False
        header_confidence = 0.0
        header_bbox = None
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confidences):
                if self.class_names.get(int(cls)) == 'header':
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Solo validar posición
                    h, w = image.shape[:2]
                    if y1 < h * 0.25:  # Solo en el 25% superior
                        header_detected = True
                        header_confidence = float(conf)
                        header_bbox = [x1, y1, x2, y2]
                        break
        
        return {
            'header_detected': header_detected,
            'header_confidence': header_confidence,
            'header_bbox': header_bbox,
            'grid_config': self._get_grid_config(header_detected)
        }
    
    def _get_grid_config(self, header_detected):
        if header_detected:
            return {'rows': 6, 'cols': 4, 'max_codes': 24, 'layout_type': 'with_header'}
        else:
            return {'rows': 7, 'cols': 4, 'max_codes': 26, 'layout_type': 'without_header'}
