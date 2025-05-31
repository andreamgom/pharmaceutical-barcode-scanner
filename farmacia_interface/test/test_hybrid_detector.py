# tests/test_hybrid_detector.py
import pytest
import cv2
import numpy as np
from pathlib import Path
from core.hybrid_detector import HybridDetector

class TestHybridDetector:
    
    @pytest.fixture
    def detector(self):
        return HybridDetector(debug=False)
    
    @pytest.fixture
    def sample_image(self):
        # Crear imagen de prueba o usar una real
        return "dataset/yolo_format/images/test/codigos10.jpg"
    
    def test_detector_initialization(self, detector):
        assert detector is not None
        assert hasattr(detector, 'yolo_detector')
        assert hasattr(detector, 'gradient_detector')
    
    def test_process_image_success(self, detector, sample_image):
        if Path(sample_image).exists():
            results, error = detector.process_image(sample_image)
            assert error is None
            assert results is not None
            assert 'valid_codes' in results
            assert 'success_rate' in results
    
    def test_process_invalid_image(self, detector):
        results, error = detector.process_image("nonexistent.jpg")
        assert error is not None
        assert results is None
    
    def test_crop_barcode_region(self, detector):
        # Test con imagen sintética
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = {
            'barcode': [{'bbox': (10, 10, 80, 80), 'area': 4900}]
        }
        
        cropped, crop_info = detector._crop_barcode_region(test_image, detections)
        assert cropped is not None
        assert 'offset_x' in crop_info
        assert 'offset_y' in crop_info
