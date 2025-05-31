# core/image_processor.py
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

class ImageProcessor:
    """Procesador de imágenes para farmacia"""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def validate_image(self, image_path):
        """Valida que la imagen sea procesable"""
        path = Path(image_path)
        
        if not path.exists():
            return False, "Archivo no existe"
        
        if path.suffix.lower() not in self.supported_formats:
            return False, f"Formato no soportado: {path.suffix}"
        
        try:
            image = cv2.imread(str(path))
            if image is None:
                return False, "No se pudo cargar la imagen"
            
            h, w = image.shape[:2]
            if w < 100 or h < 100:
                return False, "Imagen demasiado pequeña"
            
            return True, "Imagen válida"
            
        except Exception as e:
            return False, f"Error procesando imagen: {e}"
    
    def resize_if_needed(self, image, max_size=2000):
        """Redimensiona imagen si es muy grande"""
        h, w = image.shape[:2]
        
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image
    
    def enhance_for_barcodes(self, image):
        """Mejora imagen específicamente para códigos de barras"""
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Convertir de vuelta a color si es necesario
        if len(image.shape) == 3:
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced
