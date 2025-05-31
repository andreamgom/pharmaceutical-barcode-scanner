# core/barcode_preprocessor.py
import cv2
import numpy as np
from skimage.filters import threshold_otsu

class BarcodePreprocessor:
    """Clase para aplicar diferentes técnicas de preprocesamiento robustas"""
    
    def __init__(self):
        self.debug = False
    
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
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
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

    def get_all_preprocessing_techniques(self, roi_gray):
        """Retorna TODAS las técnicas de preprocesamiento aplicadas"""
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
            ("resize_2x", cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
            ("resize_3x", cv2.resize(roi_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)),
            ("resize_4x", cv2.resize(roi_gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)),
            ("resize_5x", cv2.resize(roi_gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)),
            ("resize_6x", cv2.resize(roi_gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC))
        ]
        
        return preprocessing_techniques
