# test_components.py
from farmacia_interface.core.barcode_preprocessor import BarcodePreprocessor
from farmacia_interface.core.barcode_decoder import BarcodeDecoder
from farmacia_interface.core.rectangle_merger import RectangleMerger
import cv2

# Test preprocessor
preprocessor = BarcodePreprocessor()
image = cv2.imread("dataset/yolo_format/images/test/codigos10.jpg", 0)
techniques = preprocessor.get_all_preprocessing_techniques(image)
print(f"✅ Preprocessor: {len(techniques)} técnicas disponibles")

# Test decoder
decoder = BarcodeDecoder(debug=True)
decoder.set_preprocessor(preprocessor)
code, method = decoder.decode_barcode_hybrid(image)
print(f"✅ Decoder: {code} usando {method}")

# Test merger
merger = RectangleMerger(debug=True)
test_regions = [(10, 10, 50, 30), (70, 10, 50, 30), (130, 10, 50, 30)]
merged = merger.merge_rectangles_by_layout_constraints(test_regions)
print(f"✅ Merger: {len(test_regions)} → {len(merged)} regiones")
