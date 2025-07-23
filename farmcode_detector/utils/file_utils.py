import streamlit as st
from pathlib import Path
import shutil
import time
import hashlib

def save_uploaded_file(uploaded_file, upload_dir="data/uploads"):
    """Guarda archivo subido de forma segura"""
    
    # Crear directorio si no existe
    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    
    # Generar nombre único basado en hash y timestamp
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
    timestamp = int(time.time())
    file_extension = Path(uploaded_file.name).suffix
    
    unique_filename = f"{timestamp}_{file_hash}{file_extension}"
    file_path = upload_path / unique_filename
    
    # Guardar archivo
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)

def cleanup_temp_files(max_age_hours=24):
    """Limpia archivos temporales antiguos"""
    
    temp_dirs = ["data/uploads", "data/temp", "data/processed"]
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for temp_dir in temp_dirs:
        temp_path = Path(temp_dir)
        if temp_path.exists():
            for file_path in temp_path.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        try:
                            file_path.unlink()
                        except:
                            pass  # Ignorar errores de permisos

def get_file_info(file_path):
    """Obtiene información detallada de un archivo"""
    
    path = Path(file_path)
    
    if not path.exists():
        return None
    
    stat = path.stat()
    
    return {
        'name': path.name,
        'size': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified': stat.st_mtime,
        'extension': path.suffix.lower(),
        'is_image': path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    }

def validate_image_file(file_path):
    """Valida que el archivo sea una imagen válida"""
    
    try:
        from PIL import Image
        
        with Image.open(file_path) as img:
            # Verificar que se puede abrir
            img.verify()
            
            # Reabrir para obtener información
            with Image.open(file_path) as img:
                width, height = img.size
                format_name = img.format
                
                # Validaciones básicas
                if width < 100 or height < 100:
                    return False, "Imagen demasiado pequeña (mínimo 100x100)"
                
                if width * height > 50_000_000:  # 50MP
                    return False, "Imagen demasiado grande (máximo 50MP)"
                
                if format_name not in ['JPEG', 'PNG', 'BMP', 'TIFF']:
                    return False, f"Formato no soportado: {format_name}"
                
                return True, "Imagen válida"
                
    except Exception as e:
        return False, f"Error al validar imagen: {str(e)}"

def create_results_directory(base_dir="data/results"):
    """Crea estructura de directorios para resultados"""
    
    dirs_to_create = [
        f"{base_dir}/images",
        f"{base_dir}/json",
        f"{base_dir}/csv",
        f"{base_dir}/reports"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return dirs_to_create
