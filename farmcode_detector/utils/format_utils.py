# utils/format_utils.py
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

def format_codes_for_display(codes_list):
    """Formatea códigos para mostrar en interfaz"""
    formatted = []
    for i, code in enumerate(codes_list, 1):
        if code != "Código no encontrado":
            formatted.append(f"{i:2d}. {code}")
        else:
            formatted.append(f"{i:2d}. ---")
    return formatted

def create_detection_summary(results):
    """Crea resumen formateado de detección"""
    return {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'image': results['image_name'],
        'method': results.get('method', 'Desconocido'),
        'codes_detected': results['valid_codes'],
        'total_positions': results['max_codes'],
        'success_rate': f"{results['success_rate']*100:.1f}%",
        'header_detected': "Sí" if results['header_detected'] else "No"
    }

def export_to_excel(results, output_path):
    """Exporta resultados a Excel"""
    data = []
    for pos, result in results['decoded_results'].items():
        data.append({
            'Posición': pos,
            'Código': result['code'],
            'Método': result['method'],
            'Confianza': result['confidence']
        })
    
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)
    return output_path
