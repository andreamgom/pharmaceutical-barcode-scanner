# core/rectangle_merger.py
import numpy as np
import cv2

class RectangleMerger:
    """Clase para merge inteligente de rectángulos (COPIADO EXACTO DE GRADIENT_DETECTOR.PY)"""
    
    def __init__(self, debug=False):
        self.debug = debug
    
    def merge_rectangles_by_layout_constraints(self, valid_regions, max_codes_per_row=[4,4,4,4,4,4,2]):
        """
        Une rectángulos basándose en las restricciones del layout:
        - Filas 1-6: máximo 4 códigos cada una
        - Fila 7: máximo 2 códigos
        """
        if len(valid_regions) <= 1:
            return valid_regions

        if self.debug:
            print(f"Aplicando merge inteligente basado en layout...")
            print(f" Regiones iniciales: {len(valid_regions)}")

        # Calcular centroides y agrupar por filas
        centroids = []
        for x, y, w, h in valid_regions:
            cx = x + w/2
            cy = y + h/2
            centroids.append((cx, cy, (x, y, w, h)))

        # Agrupar por filas (threshold_row = 35 como en tu código)
        threshold_row = 35
        rows = []
        for cx, cy, rect in sorted(centroids, key=lambda c: c[1]):
            placed = False
            for row in rows:
                row_y_avg = np.mean([c[1] for c in row])
                if abs(row_y_avg - cy) < threshold_row:
                    row.append((cx, cy, rect))
                    placed = True
                    break
            if not placed:
                rows.append([(cx, cy, rect)])

        # Ordenar filas por Y
        rows.sort(key=lambda row: np.mean([c[1] for c in row]))

        if self.debug:
            print(f" Filas detectadas: {len(rows)}")
            for i, row in enumerate(rows, 1):
                print(f" Fila {i}: {len(row)} códigos")

        # APLICAR MERGE INTELIGENTE FILA POR FILA
        merged_regions = []
        for row_idx, row in enumerate(rows):
            if row_idx >= len(max_codes_per_row):
                max_codes_in_row = 2  # Por defecto para filas extra
            else:
                max_codes_in_row = max_codes_per_row[row_idx]

            if self.debug:
                print(f" Procesando fila {row_idx + 1}: {len(row)} códigos → max permitido: {max_codes_in_row}")

            if len(row) <= max_codes_in_row:
                # No hay exceso, mantener como está
                for cx, cy, rect in row:
                    merged_regions.append(rect)
                if self.debug:
                    print(f" Fila {row_idx + 1}: Sin merge necesario")
            else:
                # HAY EXCESO: Aplicar merge inteligente
                if self.debug:
                    print(f" Fila {row_idx + 1}: Aplicando merge ({len(row)} → {max_codes_in_row})")
                
                # Ordenar por X dentro de la fila
                row.sort(key=lambda c: c[0])
                
                # Aplicar merge agresivo para reducir a max_codes_in_row
                merged_row = self.merge_row_to_target_count(row, max_codes_in_row)
                for rect in merged_row:
                    merged_regions.append(rect)
                
                if self.debug:
                    print(f" Fila {row_idx + 1}: Merge completado ({len(row)} → {len(merged_row)})")

        if self.debug:
            print(f"Merge inteligente completado: {len(valid_regions)} → {len(merged_regions)} regiones")
        
        return merged_regions

    def merge_row_to_target_count(self, row, target_count):
        """Reduce una fila a exactamente target_count códigos mediante merge inteligente"""
        if len(row) <= target_count:
            return [rect for cx, cy, rect in row]

        # Extraer solo los rectángulos
        rectangles = [rect for cx, cy, rect in row]

        # Mientras tengamos más códigos de los permitidos, hacer merge
        while len(rectangles) > target_count:
            # Encontrar el par de rectángulos más cercanos horizontalmente
            min_distance = float('inf')
            merge_idx = 0

            for i in range(len(rectangles) - 1):
                x1, y1, w1, h1 = rectangles[i]
                x2, y2, w2, h2 = rectangles[i + 1]
                
                # Calcular distancia entre bordes, no centros
                distance = x2 - (x1 + w1)  # Distancia horizontal entre rectángulos
                
                # Verificar que estén realmente alineados
                center_y1 = y1 + h1/2
                center_y2 = y2 + h2/2
                y_alignment = abs(center_y1 - center_y2)
                max_height = max(h1, h2)

                # Solo considerar para merge si están bien alineados
                if y_alignment < max_height * 0.4 and distance < min_distance:
                    min_distance = distance
                    merge_idx = i

            # Merge los dos rectángulos más cercanos y alineados
            rect1 = rectangles[merge_idx]
            rect2 = rectangles[merge_idx + 1]
            merged_rect = self.merge_two_rectangles(rect1, rect2)

            # Reemplazar los dos rectángulos por el merged
            rectangles = (rectangles[:merge_idx] +
                         [merged_rect] +
                         rectangles[merge_idx + 2:])

            if self.debug:
                print(f"Merged rectángulos en posiciones {merge_idx} y {merge_idx + 1} (distancia: {min_distance:.1f}px)")

        return rectangles

    def merge_two_rectangles(self, rect1, rect2):
        """Une dos rectángulos de manera controlada para cupones farmacéuticos"""
        return self.merge_rectangles_horizontal_aware(rect1, rect2)

    def validate_merge_quality(self, original_rects, merged_rect):
        """Valida que el merge no sea demasiado agresivo"""
        mx, my, mw, mh = merged_rect
        merged_area = mw * mh

        # Calcular área total de rectángulos originales
        total_original_area = sum(w * h for x, y, w, h in original_rects)

        # El merge no debería ser más de 2.5x el área original
        if merged_area > total_original_area * 2.5:
            return False, "Merge demasiado grande"

        # Verificar que el aspect ratio no sea extremo
        aspect_ratio = mw / mh if mh > 0 else 0
        if aspect_ratio > 8.0 or aspect_ratio < 0.5:
            return False, "Aspect ratio extremo"

        return True, "Merge válido"

    def merge_rectangles_horizontal_aware(self, rect1, rect2):
        """Merge que considera si los rectángulos están alineados horizontalmente"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # Calcular centros Y
        center_y1 = y1 + h1/2
        center_y2 = y2 + h2/2

        # Verificar si están en la misma fila (alineados horizontalmente)
        max_height = max(h1, h2)
        y_difference = abs(center_y1 - center_y2)

        if y_difference < max_height * 0.5:  # Están en la misma fila
            # Merge horizontal conservador
            min_x = min(x1, x2)
            max_x = max(x1 + w1, x2 + w2)
            
            # Para Y, usar el rango que cubra ambos pero sin exceso
            min_y = min(y1, y2)
            max_y = max(y1 + h1, y2 + h2)
            
            # Limitar la altura para evitar rectángulos demasiado altos
            height = max_y - min_y
            if height > max_height * 1.3:  # Máximo 30% más alto
                # Usar altura promedio centrada
                avg_height = (h1 + h2) / 2
                center_y = (center_y1 + center_y2) / 2
                min_y = int(center_y - avg_height/2)
                height = int(avg_height)

            result_rect = (min_x, min_y, max_x - min_x, height)
        else:
            # No están alineados horizontalmente, merge muy conservador
            padding = 5
            min_x = min(x1 - padding, x2 - padding)
            min_y = min(y1 - padding, y2 - padding)
            max_x = max(x1 + w1 + padding, x2 + w2 + padding)
            max_y = max(y1 + h1 + padding, y2 + h2 + padding)
            result_rect = (min_x, min_y, max_x - min_x, max_y - min_y)

        # VALIDAR EL RESULTADO ANTES DE DEVOLVERLO
        valid, message = self.validate_merge_quality([rect1, rect2], result_rect)
        if not valid:
            if self.debug:
                print(f"Merge rechazado: {message}, usando merge conservador")
            # Si el merge no es válido, usar merge conservador
            padding = 3
            return (min(rect1[0], rect2[0]) - padding,
                   min(rect1[1], rect2[1]) - padding,
                   max(rect1[0] + rect1[2], rect2[0] + rect2[2]) - min(rect1[0], rect2[0]) + 2*padding,
                   max(rect1[1] + rect1[3], rect2[1] + rect2[3]) - min(rect1[1], rect2[1]) + 2*padding)

        return result_rect

    def validate_layout_constraints(self, ordered_barcodes, max_codes_per_row=[4,4,4,4,4,4,2]):
        """Valida que el layout cumpla las restricciones y reporta problemas"""
        
        # Agrupar por filas
        threshold_row = 50
        rows = []
        for barcode in ordered_barcodes:
            num = barcode['position']
            cx, cy = barcode['centroid']
            rect = barcode['bbox']
            placed = False
            for row in rows:
                row_y_avg = np.mean([item['centroid'][1] for item in row])  # cy promedio
                if abs(row_y_avg - cy) < threshold_row:
                    row.append(barcode)
                    placed = True
                    break
            if not placed:
                rows.append([barcode])

        rows.sort(key=lambda row: np.mean([item['centroid'][1] for item in row]))

        if self.debug:
            print(f"\nVALIDACIÓN DEL LAYOUT:")
            print(f" Total filas: {len(rows)}")

        layout_valid = True
        for i, row in enumerate(rows):
            max_allowed = max_codes_per_row[i] if i < len(max_codes_per_row) else 2
            if len(row) > max_allowed:
                if self.debug:
                    print(f"Fila {i+1}: {len(row)} códigos (máximo: {max_allowed})")
                layout_valid = False
            else:
                if self.debug:
                    print(f"Fila {i+1}: {len(row)} códigos (máximo: {max_allowed})")

        return layout_valid, rows
