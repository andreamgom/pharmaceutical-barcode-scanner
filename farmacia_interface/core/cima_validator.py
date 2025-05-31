# core/cima_validator.py
import requests
import pandas as pd
import time
from typing import List, Dict
import json
from pathlib import Path

class CIMAValidator:
    """Validador robusto para códigos farmacéuticos usando API CIMA"""
    
    def __init__(self, rate_limit: float = 0.5, debug: bool = False):
        self.base_url = "https://cima.aemps.es/cima/rest/"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "FarmaScan-TFG/1.0 (Detección de códigos farmacéuticos)",
            "Accept": "application/json",
            "Accept-Language": "es-ES,es;q=0.9"
        })
        self.rate_limit = rate_limit
        self.last_request = 0
        self.debug = debug
        
        # Cache para evitar consultas repetidas
        self.cache = {}
        self.cache_file = Path("data/cache/cima_cache.json")
        self._load_cache()
    
    def _load_cache(self):
        """Carga cache de consultas anteriores"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                if self.debug:
                    print(f"Cache CIMA cargado: {len(self.cache)} entradas")
            except:
                self.cache = {}
    
    def _save_cache(self):
        """Guarda cache de consultas"""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            if self.debug:
                print(f"Error guardando cache: {e}")
    
    def _make_request(self, endpoint: str, params: Dict = None):
        """Maneja solicitudes con límite de tasa y reintentos"""
        
        # Verificar cache primero
        cache_key = f"{endpoint}_{json.dumps(params, sort_keys=True) if params else ''}"
        if cache_key in self.cache:
            if self.debug:
                print(f"Usando cache para {endpoint}")
            return self.cache[cache_key]
        
        # Respetar rate limit
        time_since_last = time.time() - self.last_request
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        
        try:
            response = self.session.get(
                f"{self.base_url}{endpoint}", 
                params=params, 
                timeout=10
            )
            response.raise_for_status()
            self.last_request = time.time()
            
            result = response.json()
            
            # Guardar en cache
            self.cache[cache_key] = result
            if len(self.cache) % 10 == 0:  # Guardar cada 10 consultas
                self._save_cache()
            
            return result
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                result = {"error": "No encontrado", "codigo_error": 404}
                self.cache[cache_key] = result
                return result
            elif e.response.status_code == 429:
                if self.debug:
                    print("Rate limit excedido, esperando...")
                time.sleep(2)
                return {"error": "Rate limit excedido", "codigo_error": 429}
            else:
                return {"error": f"Error HTTP {e.response.status_code}", "codigo_error": e.response.status_code}
        except requests.exceptions.RequestException as e:
            return {"error": f"Error de conexión: {str(e)}", "codigo_error": 0}
    
    def validar_medicamento(self, codigo: str) -> Dict:
        """Valida un código nacional y obtiene información detallada"""
        
        if not codigo or not str(codigo).isdigit():
            return {
                "codigo": codigo,
                "valido": False,
                "error": "Código no numérico o vacío"
            }
        
        codigo = str(codigo).strip()
        
        if self.debug:
            print(f"Validando código: {codigo}")
        
        # Obtener información básica del medicamento
        data = self._make_request("medicamento", {"cn": codigo})
        
        if "error" in data:
            return {
                "codigo": codigo,
                "valido": False,
                "error": data["error"],
                "codigo_error": data.get("codigo_error", 0)
            }
        
        # Obtener problemas de suministro
        suministro = self._make_request(f"psuministro/v2/cn/{codigo}")
        
        # Obtener información adicional si está disponible
        ficha_tecnica_url = None
        if data.get("nregistro"):
            ficha_tecnica_url = f"https://cima.aemps.es/cima/dochtml/ft/{data['nregistro']}/FT_{data['nregistro']}.html"
        
        return self._parse_response(codigo, data, suministro, ficha_tecnica_url)
    
    def _parse_response(self, codigo: str, data: Dict, suministro: Dict, ficha_url: str) -> Dict:
        """Procesa la respuesta de la API de forma robusta"""
        
        try:
            # Información básica
            nombre = data.get("nombre", "No disponible")
            comercializado = data.get("estaComercializado", False)
            
            # Estado de autorización
            estado = data.get("estado", {})
            autorizado = "aut" in estado if estado else False
            fecha_autorizacion = estado.get("aut") if estado else None
            
            # Laboratorio
            laboratorio = data.get("labtitular", "No disponible")
            
            # Principio activo
            principio_activo = data.get("practiv1", "No disponible")
            
            # Problemas de suministro
            problemas = suministro.get("problemas", []) if isinstance(suministro, dict) else []
            problema_suministro = any(p.get("activo", False) for p in problemas) if problemas else False
            
            # Información de problemas de suministro
            problema_info = None
            if problema_suministro and problemas:
                problema_activo = next((p for p in problemas if p.get("activo")), {})
                problema_info = {
                    "tipo": problema_activo.get("tipo", "No especificado"),
                    "fecha_inicio": problema_activo.get("fini"),
                    "fecha_fin": problema_activo.get("ffin"),
                    "observaciones": problema_activo.get("observaciones", "")
                }
            
            return {
                "codigo": codigo,
                "valido": True,
                "nombre": nombre,
                "comercializado": bool(comercializado),
                "autorizado": bool(autorizado),
                "fecha_autorizacion": fecha_autorizacion,
                "laboratorio": laboratorio,
                "principio_activo": principio_activo,
                "problema_suministro": bool(problema_suministro),
                "problema_info": problema_info,
                "ficha_tecnica": ficha_url,
                "ultima_actualizacion": data.get("ultimaActualizacion"),
                "timestamp_consulta": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error parseando respuesta para {codigo}: {e}")
            
            return {
                "codigo": codigo,
                "valido": False,
                "error": f"Error procesando respuesta: {str(e)}"
            }
    
    def validar_lista(self, codigos: List[str], progress_callback=None) -> pd.DataFrame:
        """Valida una lista de códigos y devuelve DataFrame"""
        
        resultados = []
        total = len(codigos)
        
        if self.debug:
            print(f"Validando {total} códigos con CIMA...")
        
        for i, codigo in enumerate(codigos, 1):
            if progress_callback:
                progress_callback(i, total)
            
            resultado = self.validar_medicamento(codigo)
            resultados.append(resultado)
            
            if self.debug and i % 5 == 0:
                print(f"Progreso: {i}/{total} códigos procesados")
        
        # Guardar cache final
        self._save_cache()
        
        # Crear DataFrame
        df = pd.DataFrame(resultados)
        
        # Convertir fechas si están presentes
        if 'fecha_autorizacion' in df.columns:
            df["fecha_autorizacion"] = pd.to_datetime(df["fecha_autorizacion"], unit='ms', errors='coerce')
        if 'ultima_actualizacion' in df.columns:
            df["ultima_actualizacion"] = pd.to_datetime(df["ultima_actualizacion"], unit='ms', errors='coerce')
        
        return df
    
    def limpiar_cache(self, dias_antiguedad: int = 30):
        """Limpia entradas de cache antiguas"""
        
        if not self.cache:
            return
        
        # Por simplicidad, limpiar todo el cache si es muy grande
        if len(self.cache) > 1000:
            self.cache = {}
            self._save_cache()
            if self.debug:
                print("Cache limpiado por tamaño excesivo")
        
        if self.debug:
            print(f"Cache actual: {len(self.cache)} entradas")
