# gemini_detector.py
import google.generativeai as genai
import streamlit as st
import re
import numpy as np
from PIL import Image
import cv2
import json
import time
from pathlib import Path
from typing import Dict, List

class GeminiDetector:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.name = "Gemini"
        self._configure_api()
        self.model = genai.GenerativeModel('gemini-1.5-flash') 

        
        self.prompt_template = """
        Analiza la imagen de medicamentos hospitalarios y realiza:

        1️⃣ **Detección de Header**:
        - Busca texto manuscrito con:
          • "SOE de Farmacia"
          • "Día"
          • "Número de la orden"
        - Si existe → Cuadrícula 4x6 (con header)
        - Si no existe → Cuadrícula 4x7 (sin header)

        2️⃣ **Extracción de Códigos**:
        - Devuelve tabla Markdown estricta:
          - 4 columnas
          - 6/7 filas según header
          - Celdas vacías = "Código no encontrado"
          - Mantener posición original

        3️⃣ **Información Header (si aplica)**:
        - Extraer:
          • SOE de Farmacia: [valor] 
          • Día: [valor]
          • Orden: [entero 1-9]
        
        **Formato de respuesta**:
        🔍 **Header detectado**: [Sí/No]
        
        📊 **Códigos detectados**:
        | Col1 | Col2 | Col3 | Col4 |
        |------|------|------|------|
        | ... | ... | ... | ... |

        📝 **Información header** (solo si aplica):
        ```
        SOE: ...
        Día: ...
        Orden: ...
        ```
        """

    def _configure_api(self):
        """Configura la API usando la misma estructura que tu interfaz"""
        try:
            # Usar la misma clave que en tu interfaz
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        except Exception as e:
            raise ValueError(f"Error configurando Gemini: {str(e)}")



    def normalize_ean13(self, code):
        """Normaliza códigos EAN-13"""
        if not code or "código no encontrado" in str(code).lower():
            return "Código no encontrado"
        
        # Limpiar y validar
        clean_code = ''.join(filter(str.isdigit, str(code)))
        if len(clean_code) == 13:
            return clean_code
        return "Código no encontrado"

    def save_results_json(self, results, output_path):
        """Guarda resultados en JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    def load_ground_truth(self, image_name, gt_dir="data/ground_truth"):
        """Carga ground truth para una imagen"""
        gt_path = Path(gt_dir) / f"{Path(image_name).stem}_simple.json"
        if gt_path.exists():
            with open(gt_path, 'r') as f:
                return json.load(f)
        return None

    def _parse_response(self, text: str) -> Dict:
        """Parsea la respuesta estructurada de Gemini"""
        result = {
            "has_header": False,
            "grid": [],
            "soe_info": {}
        }

        try:
            # Detectar presencia de header
            header_match = re.search(r'🔍 \*\*Header detectado\*\*: (\w+)', text)
            if header_match:
                result["has_header"] = header_match.group(1).strip().lower() == "sí"

            # Extraer tabla de códigos
            table_match = re.search(r'📊 \*\*Códigos detectados\*\*:\n(.*?)(?=\n📝|\n🔍|\Z)', text, re.DOTALL)
            if table_match:
                table_text = table_match.group(1).strip()
                rows = [row.split('|')[1:-1] for row in table_text.split('\n') if '|' in row][1:]  # Saltar encabezado
                result["grid"] = [[cell.strip() for cell in row] for row in rows]

            # Extraer información del header
            info_match = re.search(r'📝 \*\*Información header\*\*.*?``````', text, re.DOTALL)
            if info_match:
                info_text = info_match.group(1).strip()
                for line in info_text.split('\n'):
                    if 'SOE:' in line:
                        result["soe_info"]["soe"] = line.split(':')[-1].strip()
                    elif 'Día:' in line:
                        result["soe_info"]["dia"] = line.split(':')[-1].strip()
                    elif 'Orden:' in line:
                        result["soe_info"]["orden"] = line.split(':')[-1].strip()

        except Exception as e:
            if self.debug:
                print(f"Error parsing response: {str(e)}")

        return result

    def process_image_with_gemini(self, image: np.ndarray) -> Dict:
        """Procesa imagen con Gemini con timeout y reintentos"""
        
        max_retries = 3
        base_timeout = 120  # Empezar con 2 minutos
        
        for attempt in range(max_retries):
            try:
                # Convertir a PIL Image
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                # Generar respuesta con timeout personalizado
                response = self.model.generate_content(
                    [self.prompt_template, pil_image],
                    request_options={"timeout": base_timeout * (attempt + 1)}  # Timeout incremental
                )
                
                if not response.parts:
                    raise ValueError("Respuesta vacía de Gemini")
                
                return self._parse_response(response.text)
                
            except Exception as e:
                error_msg = str(e)
                
                if "503" in error_msg or "timeout" in error_msg.lower() or "deadline" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 30  # 30s, 60s, 90s
                        if self.debug:
                            print(f"Intento {attempt + 1} falló: {error_msg}")
                            print(f"Esperando {wait_time}s antes del siguiente intento...")
                        time.sleep(wait_time)
                        continue
                    else:
                        if self.debug:
                            print(f"Todos los intentos fallaron: {error_msg}")
                        return {"error": f"Timeout después de {max_retries} intentos"}
                else:
                    if self.debug:
                        print(f"Error no relacionado con timeout: {error_msg}")
                    return {"error": error_msg}
        
        return {"error": "Error desconocido después de reintentos"}


        def _count_valid_codes(self, grid):
            """Cuenta códigos válidos en la grid"""
            if not grid:
                return 0
            return sum(1 for row in grid for cell in row 
                    if cell != "Código no encontrado")

    def calculate_accuracy(self, detected_grid, ground_truth):
        """Calcula accuracy comparando con ground truth"""
        if not detected_grid or not ground_truth:
            return 0.0
        
        # Convertir grid a lista plana
        detected_flat = []
        for row in detected_grid:
            detected_flat.extend(row)
        
        # Comparar posición por posición
        matches = 0
        total = min(len(detected_flat), len(ground_truth))
        
        for i in range(total):
            if i < len(detected_flat) and detected_flat[i] == ground_truth[i]:
                matches += 1
        
        return matches / len(ground_truth) if ground_truth else 0.0

    def process_image(self, image_path):
        """Procesa imagen y devuelve resultados estándar"""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
            
        start_time = time.time()
        
        try:
            # Procesar con Gemini
            result = self.process_image_with_gemini(image)
            
            # Normalizar códigos
            normalized_grid = []
            for row in result.get('grid', []):
                new_row = [self.normalize_ean13(cell) for cell in row]
                normalized_grid.append(new_row)
            
            # Formato estándar de salida
            return {
                'image_name': Path(image_path).name,
                'detector': self.name,
                'success': bool(result.get('grid')),
                'grid': normalized_grid,
                'has_header': result.get('has_header', False),
                'soe_info': result.get('soe_info', {}),
                'processing_time': time.time() - start_time,
                'total_codes': self._count_valid_codes(normalized_grid)
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error: {e}")
            return {
                'image_name': Path(image_path).name,
                'detector': self.name,
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }

