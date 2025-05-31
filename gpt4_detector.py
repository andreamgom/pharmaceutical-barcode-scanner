# azure_gpt4_detector.py
import base64
import json
import requests
import streamlit as st
from typing import Dict, List
import numpy as np
import cv2
import time
from pathlib import Path

class GPT4Detector:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.name = "GPT-4V"
        
        # Usar las mismas claves que en tu interfaz
        self.api_key = st.secrets["AZURE_OPENAI_KEY"]
        self.endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
        self.deployment_name = st.secrets["AZURE_DEPLOYMENT_NAME"]
        self.api_version = "2024-06-01"  # Versión estable
        
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }



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

    def _encode_image(self, image: np.ndarray) -> str:
        """Convierte imagen numpy a base64"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def _build_payload(self, image_b64: str) -> Dict:
        """Construye payload para GPT-4 Vision con prompt optimizado"""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                            Eres un experto en análisis de imágenes médicas farmacéuticas. Analiza esta imagen de cupones de medicamentos hospitalarios paso a paso:

                            **PASO 1: Examinar la estructura**
                            - Busca texto manuscrito en la parte superior: "SOE de Farmacia", "Día", "Número de la orden"
                            - Si existe header → cuadrícula 4 columnas x 6 filas (24 códigos)
                            - Si no hay header → cuadrícula 4 columnas x 7 filas (28 códigos)

                            **PASO 2: Localizar códigos de barras**
                            - Busca códigos de barras EAN-13 (13 dígitos que empiecen con 847)
                            - Están organizados en una cuadrícula rectangular
                            - Cada código está debajo de su respectivo código de barras visual
                            - Lee fila por fila, de izquierda a derecha

                            **PASO 3: Extraer códigos sistemáticamente**
                            - Examina cada posición de la cuadrícula cuidadosamente
                            - Si no hay código visible, marca como "Código no encontrado"
                            - Mantén la posición exacta en la cuadrícula

                            **FORMATO DE RESPUESTA JSON:**
                            {
                                "has_header": boolean,
                                "grid": [
                                    ["código_fila1_col1", "código_fila1_col2", "código_fila1_col3", "código_fila1_col4"],
                                    ["código_fila2_col1", "código_fila2_col2", "código_fila2_col3", "código_fila2_col4"],
                                    ...
                                ],
                                "soe_info": {
                                    "soe": "valor_si_existe",
                                    "dia": "valor_si_existe", 
                                    "orden": "valor_si_existe"
                                }
                            }

                            **IMPORTANTE:**
                            - Los códigos EAN-13 españoles empiezan con "847"
                            - Tienen exactamente 13 dígitos
                            - Si no puedes leer un código claramente, usa "Código no encontrado"
                            - Mantén la estructura de cuadrícula exacta
                            """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 3000,  # Aumentado para más detalle
            "temperature": 0.0,  # Más determinístico
            "response_format": {"type": "json_object"}
        }


    def process_image_with_azure(self, image: np.ndarray) -> Dict:
        """Procesa imagen con GPT-4 Vision con reintentos y timeout mejorado"""        
        max_retries = 3
        base_timeout = 60  # Timeout base de 60 segundos
        
        for attempt in range(max_retries):
            try:
                # Codificar imagen
                image_b64 = self._encode_image(image)
                
                # Construir solicitud
                payload = self._build_payload(image_b64)
                
                # URL corregida (sin duplicar deployments)
                url = f"{self.endpoint}openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
                
                # Headers con timeout
                headers = {
                    "Content-Type": "application/json",
                    "api-key": self.api_key
                }
                
                # Request con timeout incremental
                timeout = base_timeout * (attempt + 1)  # 60s, 120s, 180s
                
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=payload,
                    timeout=timeout  # Timeout explícito
                )
                response.raise_for_status()
                
                # Procesar respuesta
                result = response.json()
                content = json.loads(result['choices'][0]['message']['content'])
                
                # Normalizar códigos
                normalized_grid = []
                for row in content.get('grid', []):
                    new_row = []
                    for cell in row:
                        if "código no encontrado" in str(cell).lower():
                            new_row.append("Código no encontrado")
                        else:
                            new_row.append(str(cell))
                    normalized_grid.append(new_row)
                
                return {
                    "has_header": content.get("has_header", False),
                    "grid": normalized_grid,
                    "soe_info": content.get("soe_info", {})
                }
                
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout,
                    requests.exceptions.HTTPError) as e:
                
                error_msg = str(e)
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30  # 30s, 60s, 90s
                    if self.debug:
                        print(f"GPT-4 intento {attempt + 1} falló: {error_msg}")
                        print(f"Esperando {wait_time}s antes del siguiente intento...")
                    time.sleep(wait_time)
                    continue
                else:
                    if self.debug:
                        print(f"GPT-4 todos los intentos fallaron: {error_msg}")
                    return {
                        "has_header": False,
                        "grid": [],
                        "soe_info": {},
                        "error": f"Connection error después de {max_retries} intentos"
                    }
                    
            except Exception as e:
                if self.debug:
                    print(f"Error GPT-4 Vision: {str(e)}")
                return {
                    "has_header": False,
                    "grid": [],
                    "soe_info": {},
                    "error": str(e)
                }
        
        return {
            "has_header": False,
            "grid": [],
            "soe_info": {},
            "error": "Error desconocido después de reintentos"
        }


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
            # Procesar con Azure GPT-4V
            result = self.process_image_with_azure(image)
            
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
