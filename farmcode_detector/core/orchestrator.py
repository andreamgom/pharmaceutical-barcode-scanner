# core/orchestrator.py

import cv2
import time
import numpy as np
from pathlib import Path

# Imports directos como en el notebook
from .detectors.header_detector import HeaderDetector
from .detectors.barcode_detector import BarcodeDetector
from .detectors.grid_detector import GridDetector
from .processors.barcode_decoder import BarcodeDecoder
from .processors.barcode_preprocessor import BarcodePreprocessor
from .processors.rectangle_merger import RectangleMerger
from .processors.grid_position_corrector import GridPositionCorrector

class Orchestrator:
    """Orquestador simple"""
    
    def __init__(self, yolo_model_path, debug=True):
        self.debug = debug
        
        # Inicializar componentes 
        self.header_detector = HeaderDetector(model_path=yolo_model_path, debug=debug)
        self.barcode_detector = BarcodeDetector(model_path=yolo_model_path, debug=debug)
        self.grid_detector = GridDetector(debug=debug)
        self.decoder = BarcodeDecoder(debug=debug)
        self.preprocessor = BarcodePreprocessor(debug=debug)
        self.decoder.set_preprocessor(self.preprocessor)
        self.merger = RectangleMerger(debug=debug)
        self.position_corrector = GridPositionCorrector(debug=debug)
        
        if self.debug:
            print("✅ Orquestador simple inicializado")
    
    def process_image(self, image_input):
        """Procesa imagen EXACTAMENTE como en el notebook"""
        start_time = time.time()
        
        try:
            # PASO 1: Cargar imagen
            if isinstance(image_input, str):
                imagen_original = cv2.imread(image_input)
                image_name = Path(image_input).name
            else:
                # Array de Streamlit - convertir RGB a BGR
                if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                    imagen_original = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
                else:
                    imagen_original = image_input
                image_name = "streamlit_upload"
            
            if imagen_original is None:
                return None, "Error cargando imagen"
            
            if self.debug:
                print(f"🔍 PROCESANDO: {image_name}")
                print(f"   Dimensiones: {imagen_original.shape}")
            
            # PASO 2: Detección header
            header_result = self.header_detector.detect_header(imagen_original)
            
            if self.debug:
                print(f"   Header detectado: {header_result['header_detected']}")
                print(f"   Grid config: {header_result['grid_config']}")
            
            # PASO 3: Detección región barcode 
            barcode_result = self.barcode_detector.detect_and_crop_barcode_region(
            imagen_original, header_result, {}
            )
            
            if not barcode_result['detection_success']:
                error_msg = barcode_result.get('error', 'No se detectó región de códigos')
                if self.debug:
                    print(f"❌ {error_msg}")
                return None, error_msg

            # Continuar solo si la detección fue exitosa...
            if self.debug:
                print(f"   Región barcode detectada exitosamente")
                print(f"   Crop bbox: {barcode_result['crop_info'].get('crop_bbox', 'N/A')}")
                
            # PASO 4: Detección por gradientes 
            gradient_result = self.grid_detector.detect_codes_in_grid(
                barcode_result['cropped_image'],
                header_result['grid_config']
            )
            
            rectangles_detectados = len(gradient_result['decoded_results'])
            if self.debug:
                print(f"   Rectángulos por gradientes: {rectangles_detectados}")
            
            # PASO 5: Merge de rectángulos 
            rectangles_for_merge = [d['bbox'] for d in gradient_result['decoded_results'].values()]
            
            # Configuración según header
            if header_result['header_detected']:
                max_codes_per_row = [4, 4, 4, 4, 4, 4]
                grid_shape = (6, 4)
                max_codes = 24
            else:
                max_codes_per_row = [4, 4, 4, 4, 4, 4, 2]
                grid_shape = (7, 4)
                max_codes = 26
            
            merged_rectangles = self.merger.merge_rectangles_by_layout_constraints(
                rectangles_for_merge,
                max_codes_per_row=max_codes_per_row
            )
            
            if self.debug:
                print(f"   Rectángulos después merge: {len(merged_rectangles)}")
            
            

            # PASO 6: Decodificación
            detection_result_for_expansion = {'decoded_results': {}}
            for idx, rect in enumerate(merged_rectangles):
                detection_result_for_expansion['decoded_results'][idx+1] = {
                    'bbox': rect, 
                    'confidence': 1.0
                }
            
            # Decodificación con expansión
            decoded_results, decoding_stats = self.decoder.decode_grid_complete(
                barcode_result['cropped_image'], 
                detection_result_for_expansion, 
                len(merged_rectangles)
            )

            # # PASO 7: Corrección de posiciones post-decodificación
            if hasattr(self, 'position_corrector'):
                corrected_results = self.position_corrector.correct_grid_positions(
                    decoded_results, 
                    grid_shape, 
                    barcode_result['cropped_image'].shape,
                    cropped_image=barcode_result['cropped_image']
                )
                
                # Comparar resultados antes y después
                original_valid = sum(1 for r in decoded_results.values() if r['code'] != "No detectado")
                corrected_valid = sum(1 for r in corrected_results.values() if r['code'] != "No detectado")
                
                if self.debug and corrected_valid != original_valid:
                    print(f"   🔄 CORRECCIÓN POSICIONAL:")
                    print(f"      Antes: {original_valid} códigos")
                    print(f"      Después: {corrected_valid} códigos")
                
                decoded_results = corrected_results
            
            # PASO 8: Ensamblar resultados finales
            valid_codes = sum(1 for r in decoded_results.values() if r['code'] != "No detectado")
            
            resultado_final = {
                'valid_codes': valid_codes,
                'max_codes': len(decoded_results),
                'success_rate': valid_codes / len(decoded_results) if decoded_results else 0,
                'decoded_results': decoded_results,
                'processing_time': time.time() - start_time,
                'image_name': image_name,
                'header_detected': header_result['header_detected'],
                'grid_layout': grid_shape,
                'barcode_region': cv2.cvtColor(barcode_result['cropped_image'], cv2.COLOR_BGR2RGB),
                'decoding_stats': decoding_stats,
                'merge_stats': {
                    'rectangles_detected': rectangles_detectados,
                    'rectangles_merged': len(merged_rectangles)
                }
            }
            
            if self.debug:
                print(f"✅ COMPLETADO: {valid_codes}/{len(decoded_results)} códigos válidos")
                print(f"   Tasa éxito: {resultado_final['success_rate']:.1%}")
                print(f"   Tiempo: {resultado_final['processing_time']:.3f}s")
            
            return resultado_final, None
            
        except Exception as e:
            error_msg = f"Error en orquestador simple: {str(e)}"
            if self.debug:
                print(f"❌ {error_msg}")
                import traceback
                traceback.print_exc()
            return None, error_msg