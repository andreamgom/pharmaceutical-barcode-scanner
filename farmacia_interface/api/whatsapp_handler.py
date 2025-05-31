# api/whatsapp_handler.py
import requests
import asyncio
from pathlib import Path

class WhatsAppHandler:
    def __init__(self):
        self.api_url = "https://graph.facebook.com/v17.0"
        self.phone_number_id = "tu_phone_number_id"
        self.access_token = "tu_access_token"
    
    async def handle_message(self, message_data):
        """Maneja mensajes entrantes de WhatsApp"""
        if self._is_image_message(message_data):
            return await self._process_image_message(message_data)
        else:
            return {"status": "ignored", "reason": "No es imagen"}
    
    def _is_image_message(self, data):
        """Verifica si es mensaje con imagen"""
        try:
            messages = data.get('entry', [{}])[0].get('changes', [{}])[0].get('value', {}).get('messages', [])
            return messages and messages[0].get('type') == 'image'
        except:
            return False
    
    async def _process_image_message(self, data):
        """Procesa mensaje con imagen"""
        # Lógica para descargar y procesar imagen
        return {"status": "processing"}
    
    async def send_message(self, phone_number, message):
        """Envía mensaje de respuesta"""
        url = f"{self.api_url}/{self.phone_number_id}/messages"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        payload = {
            "messaging_product": "whatsapp",
            "to": phone_number,
            "text": {"body": message}
        }
        
        response = requests.post(url, json=payload, headers=headers)
        return response.json()
