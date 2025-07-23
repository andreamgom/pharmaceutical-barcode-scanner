# core/processors/decoding/barcode_preprocessor.py
import cv2
import numpy as np
from skimage.filters import threshold_otsu

class BarcodePreprocessor:
    """Clase para aplicar diferentes técnicas de preprocesamiento robustas"""
    
    def __init__(self, debug=False):
        self.debug = debug
    
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
    
    def gamma_correction(self, image, gamma=1.2):
        """Aplica corrección gamma"""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def unsharp_mask(self, image, sigma=1.0, strength=1.5):
        """Aplica máscara de desenfoque"""
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = float(strength + 1) * image - float(strength) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        return sharpened.astype(np.uint8)
    
    def histogram_equalization(self, image):
        """Aplica ecualización de histograma"""
        return cv2.equalizeHist(image)
    
    def noise_reduction(self, image):
        """Reduce ruido usando filtro bilateral"""
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def edge_enhancement(self, image):
        """Realza bordes para mejorar detección de códigos"""
        # Filtro Laplaciano para realzar bordes
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Combinar con imagen original
        enhanced = cv2.addWeighted(image, 0.8, laplacian, 0.2, 0)
        return enhanced
    
    def contrast_stretching(self, image):
        """Aplica estiramiento de contraste"""
        # Encontrar valores mínimo y máximo
        min_val = np.min(image)
        max_val = np.max(image)
        
        # Evitar división por cero
        if max_val == min_val:
            return image
        
        # Estirar contraste
        stretched = ((image - min_val) / (max_val - min_val)) * 255
        return stretched.astype(np.uint8)
    
    def apply_gaussian_pyramid(self, image, levels=2):
        """Aplica pirámide gaussiana para suavizado multi-escala"""
        current = image.copy()
        for i in range(levels):
            current = cv2.pyrDown(current)
        
        # Reconstruir a tamaño original
        for i in range(levels):
            current = cv2.pyrUp(current)
        
        # Redimensionar exactamente al tamaño original si es necesario
        if current.shape != image.shape:
            current = cv2.resize(current, (image.shape[1], image.shape[0]))
        
        return current
    
    def apply_top_hat_transform(self, image, kernel_size=15):
        """Aplica transformación top-hat para realzar estructuras claras"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        return cv2.add(image, tophat)
    
    def apply_black_hat_transform(self, image, kernel_size=15):
        """Aplica transformación black-hat para realzar estructuras oscuras"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        return cv2.subtract(image, blackhat)

    def get_all_preprocessing_techniques(self, roi_gray):
        """Retorna TODAS las técnicas de preprocesamiento aplicadas (EXACTO DE TU CÓDIGO)"""
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
            ("gamma_12", self.gamma_correction(roi_gray, 1.2)),
            ("gamma_08", self.gamma_correction(roi_gray, 0.8)),
            ("unsharp", self.unsharp_mask(roi_gray)),
            ("hist_eq", self.histogram_equalization(roi_gray)),
            ("noise_reduction", self.noise_reduction(roi_gray)),
            ("edge_enhancement", self.edge_enhancement(roi_gray)),
            ("contrast_stretch", self.contrast_stretching(roi_gray)),
            ("gaussian_pyramid", self.apply_gaussian_pyramid(roi_gray)),
            ("top_hat", self.apply_top_hat_transform(roi_gray)),
            ("black_hat", self.apply_black_hat_transform(roi_gray)),
            ("resize_2x", cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
            ("resize_3x", cv2.resize(roi_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)),
            ("resize_4x", cv2.resize(roi_gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)),
            ("resize_5x", cv2.resize(roi_gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)),
            ("resize_6x", cv2.resize(roi_gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC))
        ]
        
        if self.debug:
            print(f"Aplicando {len(preprocessing_techniques)} técnicas de preprocesamiento")
        
        return preprocessing_techniques
    
    def get_basic_preprocessing_techniques(self, roi_gray):
        """Retorna solo las técnicas básicas más efectivas"""
        basic_techniques = [
            ("original", roi_gray),
            ("clahe", self.apply_clahe(roi_gray)),
            ("contrast", self.enhance_contrast(roi_gray)),
            ("sharpen", self.sharpen_image(roi_gray)),
            ("adaptive_thresh", self.adaptive_threshold(roi_gray)),
            ("resize_2x", cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
            ("resize_3x", cv2.resize(roi_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
        ]
        
        return basic_techniques
    
    def get_advanced_preprocessing_techniques(self, roi_gray):
        """Retorna técnicas avanzadas para casos difíciles"""
        advanced_techniques = [
            ("gamma_12", self.gamma_correction(roi_gray, 1.2)),
            ("gamma_08", self.gamma_correction(roi_gray, 0.8)),
            ("unsharp", self.unsharp_mask(roi_gray)),
            ("edge_enhancement", self.edge_enhancement(roi_gray)),
            ("top_hat", self.apply_top_hat_transform(roi_gray)),
            ("black_hat", self.apply_black_hat_transform(roi_gray)),
            ("gaussian_pyramid", self.apply_gaussian_pyramid(roi_gray)),
            ("resize_4x", cv2.resize(roi_gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)),
            ("resize_5x", cv2.resize(roi_gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)),
            ("resize_6x", cv2.resize(roi_gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC))
        ]
        
        return advanced_techniques
    
    def validate_preprocessed_image(self, original, processed, technique_name):
        """Valida que la imagen preprocesada sea válida"""
        if processed is None:
            return False, f"Técnica {technique_name} devolvió None"
        
        if processed.size == 0:
            return False, f"Técnica {technique_name} devolvió imagen vacía"
        
        # Verificar que las dimensiones sean razonables
        if len(processed.shape) != 2:  # Debe ser escala de grises
            return False, f"Técnica {technique_name} no devolvió imagen en escala de grises"
        
        # Verificar que no sea completamente negra o blanca
        if np.all(processed == 0) or np.all(processed == 255):
            return False, f"Técnica {technique_name} devolvió imagen uniforme"
        
        return True, "Imagen válida"
    
    def create_preprocessing_visualization(self, roi_gray, max_techniques=9):
        """Crea visualización de las técnicas de preprocesamiento"""
        techniques = self.get_all_preprocessing_techniques(roi_gray)[:max_techniques]
        
        # Calcular grid para visualización
        rows = int(np.ceil(np.sqrt(len(techniques))))
        cols = int(np.ceil(len(techniques) / rows))
        
        # Crear imagen combinada
        h, w = roi_gray.shape
        combined = np.zeros((h * rows, w * cols), dtype=np.uint8)
        
        for i, (name, img) in enumerate(techniques):
            row = i // cols
            col = i % cols
            
            # Redimensionar si es necesario
            if img.shape != roi_gray.shape:
                img = cv2.resize(img, (w, h))
            
            # Colocar en grid
            y_start = row * h
            y_end = y_start + h
            x_start = col * w
            x_end = x_start + w
            
            combined[y_start:y_end, x_start:x_end] = img
        
        return combined, [name for name, _ in techniques]
