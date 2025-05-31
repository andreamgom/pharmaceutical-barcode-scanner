# api/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import json
import time
from pathlib import Path
import sys
import io
from PIL import Image

# Añadir path para importar detectores
sys.path.append('..')
from core.hybrid_detector import HybridDetector
from core.cima_validator import CIMAValidator

app = FastAPI(
    title="FarmaScan API",
    description="API para detección de códigos farmacéuticos vía WhatsApp",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar detectores globales
hybrid_detector = None
cima_validator = None

@app.on_event("startup")
async def startup_event():
    """Inicializar detectores al arrancar la API"""
    global hybrid_detector, cima_validator
    
    print("Inicializando detectores...")
    hybrid_detector = HybridDetector(debug=False)
    cima_validator = CIMAValidator(rate_limit=0.5)
    print("Detectores inicializados correctamente")

@app.get("/")
async def root():
    """Endpoint de estado de la API"""
    return {
        "status": "active",
        "service": "FarmaScan API",
        "version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/health")
async def health_check():
    """Verificación de salud de la API"""
    return {
        "status": "healthy",
        "detectors": {
            "hybrid": hybrid_detector is not None,
            "cima": cima_validator is not None
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/detect-codes")
async def detect_codes(
    file: UploadFile = File(...),
    validate_cima: bool = True,
    include_details: bool = False
):
    """
    Endpoint principal para detección de códigos farmacéuticos
    """
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    try:
        # Leer imagen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="No se pudo procesar la imagen")
        
        # Guardar imagen temporal
        temp_dir = Path("data/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"temp_{int(time.time())}.jpg"
        cv2.imwrite(str(temp_path), image)
        
        # Procesar con detector híbrido
        results, error = hybrid_detector.process_image(str(temp_path))
        
        # Limpiar archivo temporal
        if temp_path.exists():
            temp_path.unlink()
        
        if error:
            raise HTTPException(status_code=500, detail=f"Error en detección: {error}")
        
        # Extraer códigos detectados
        detected_codes = []
        for position, result in results['decoded_results'].items():
            if result['code'] != "No detectado":
                detected_codes.append(result['code'])
        
        response_data = {
            "success": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detection_summary": {
                "total_codes_detected": len(detected_codes),
                "max_possible_codes": results['max_codes'],
                "success_rate": results['success_rate'],
                "header_detected": results['header_detected'],
                "processing_time": results.get('processing_time', 0)
            },
            "codes": detected_codes
        }
        
        # Validación con CIMA si se solicita
        if validate_cima and detected_codes:
            print(f"Validando {len(detected_codes)} códigos con CIMA...")
            
            validation_results = []
            for code in detected_codes:
                validation = cima_validator.validar_medicamento(code)
                validation_results.append({
                    "code": code,
                    "valid": validation.get('valido', False),
                    "name": validation.get('nombre', 'No disponible'),
                    "commercialized": validation.get('comercializado', False),
                    "supply_problem": validation.get('problema_suministro', False)
                })
            
            response_data["cima_validation"] = {
                "validated": True,
                "valid_codes": sum(1 for v in validation_results if v['valid']),
                "invalid_codes": sum(1 for v in validation_results if not v['valid']),
                "results": validation_results
            }
        
        # Incluir detalles técnicos si se solicita
        if include_details:
            response_data["technical_details"] = {
                "method": results.get('method', 'Híbrido'),
                "grid_layout": {
                    "rows": results['grid_layout'][0],
                    "cols": results['grid_layout'][1]
                },
                "decoding_stats": results.get('decoding_stats', {}),
                "hybrid_info": results.get('hybrid_info', {})
            }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request_data: dict, background_tasks: BackgroundTasks):
    """
    Webhook para recibir mensajes de WhatsApp con imágenes
    """
    
    try:
        # Verificar que es un mensaje con imagen
        if not _is_image_message(request_data):
            return {"status": "ignored", "reason": "No es un mensaje con imagen"}
        
        # Extraer información del mensaje
        message_info = _extract_message_info(request_data)
        
        # Procesar imagen en background
        background_tasks.add_task(
            process_whatsapp_image,
            message_info['image_url'],
            message_info['phone_number'],
            message_info['message_id']
        )
        
        return {"status": "received", "message": "Procesando imagen..."}
        
    except Exception as e:
        print(f"Error en webhook WhatsApp: {e}")
        return {"status": "error", "message": str(e)}

def _is_image_message(data: dict) -> bool:
    """Verifica si el mensaje contiene una imagen"""
    try:
        messages = data.get('entry', [{}])[0].get('changes', [{}])[0].get('value', {}).get('messages', [])
        if messages:
            message = messages[0]
            return message.get('type') == 'image'
        return False
    except:
        return False

def _extract_message_info(data: dict) -> dict:
    """Extrae información relevante del mensaje de WhatsApp"""
    try:
        message = data['entry'][0]['changes'][0]['value']['messages'][0]
        
        return {
            'phone_number': message['from'],
            'message_id': message['id'],
            'image_url': message['image']['id'],  # ID de la imagen en WhatsApp
            'timestamp': message['timestamp']
        }
    except Exception as e:
        raise ValueError(f"Error extrayendo información del mensaje: {e}")

async def process_whatsapp_image(image_id: str, phone_number: str, message_id: str):
    """Procesa imagen de WhatsApp y envía respuesta"""
    
    try:
        # Descargar imagen de WhatsApp (implementar según API de WhatsApp)
        image_path = await _download_whatsapp_image(image_id)
        
        # Procesar con detector híbrido
        results, error = hybrid_detector.process_image(image_path)
        
        if error:
            await _send_whatsapp_message(phone_number, f"Error procesando imagen: {error}")
            return
        
        # Extraer códigos
        detected_codes = []
        for position, result in results['decoded_results'].items():
            if result['code'] != "No detectado":
                detected_codes.append(result['code'])
        
        # Crear mensaje de respuesta
        if detected_codes:
            message = f"Códigos detectados ({len(detected_codes)}):\n"
            for i, code in enumerate(detected_codes, 1):
                message += f"{i}. {code}\n"
            
            message += f"\nTasa de éxito: {results['success_rate']*100:.1f}%"
            
            # Validar con CIMA
            valid_count = 0
            for code in detected_codes[:5]:  # Limitar a 5 para evitar spam
                validation = cima_validator.validar_medicamento(code)
                if validation.get('valido'):
                    valid_count += 1
            
            message += f"\nCódigos válidos en CIMA: {valid_count}/{min(len(detected_codes), 5)}"
        else:
            message = "No se detectaron códigos de barras en la imagen. Verifica que la imagen sea clara y contenga códigos farmacéuticos."
        
        # Enviar respuesta
        await _send_whatsapp_message(phone_number, message)
        
        # Limpiar archivo temporal
        if Path(image_path).exists():
            Path(image_path).unlink()
            
    except Exception as e:
        print(f"Error procesando imagen WhatsApp: {e}")
        await _send_whatsapp_message(phone_number, "Error interno procesando la imagen.")

async def _download_whatsapp_image(image_id: str) -> str:
    """Descarga imagen de WhatsApp (implementar según API específica)"""
    # Implementar descarga real según API de WhatsApp Business
    # Por ahora, placeholder
    temp_path = f"data/temp/whatsapp_{image_id}_{int(time.time())}.jpg"
    return temp_path

async def _send_whatsapp_message(phone_number: str, message: str):
    """Envía mensaje de respuesta por WhatsApp (implementar según API específica)"""
    # Implementar envío real según API de WhatsApp Business
    print(f"Enviando a {phone_number}: {message}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
