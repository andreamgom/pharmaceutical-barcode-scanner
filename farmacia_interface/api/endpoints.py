# api/endpoints.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from .whatsapp_handler import WhatsAppHandler

router = APIRouter()
whatsapp_handler = WhatsAppHandler()

@router.post("/detect-bulk")
async def detect_codes_bulk(files: list[UploadFile] = File(...)):
    """Endpoint para procesar múltiples imágenes"""
    results = []
    
    for file in files:
        # Procesar cada archivo
        # ... lógica de procesamiento
        pass
    
    return {"results": results}

@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Obtiene estado de un trabajo de procesamiento"""
    # Lógica para obtener estado
    return {"job_id": job_id, "status": "completed"}

@router.post("/webhook/whatsapp")
async def whatsapp_webhook(request_data: dict):
    """Webhook específico para WhatsApp"""
    return await whatsapp_handler.handle_message(request_data)
