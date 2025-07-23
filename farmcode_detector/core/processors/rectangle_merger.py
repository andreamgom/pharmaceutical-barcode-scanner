# core/processors/layout/rectangle_merger.py - VERSIÓN COMPLETA CON MAPEO A CUADRÍCULA VIRTUAL

import cv2
import numpy as np
from typing import List, Tuple, Optional

class RectangleMerger:
    """Merger mejorado con múltiples rondas y mapeo a cuadrícula virtual"""
    
    def __init__(self, debug=False):
        self.debug = debug
    
    def merge_rectangles_by_layout_constraints(self, rectangles: List[Tuple],
                                             max_codes_per_row: List[int] = None,
                                             yolo_validation_codes: List = None) -> List[Tuple]:
        """Merge con MÚLTIPLES RONDAS hasta cumplir restricciones"""
        if not rectangles:
            return []
        
        if max_codes_per_row is None:
            max_codes_per_row = [4, 4, 4, 4, 4, 4, 2]  # Layout por defecto
        
        if self.debug:
            print(f"🔧 MERGE CON MÚLTIPLES RONDAS: {len(rectangles)} rectángulos")
        
        # Agrupar por filas
        rows = self._group_rectangles_by_rows(rectangles)
        
        if self.debug:
            print(f"  Agrupados en {len(rows)} filas")
        
        # Merge fila por fila CON MÚLTIPLES RONDAS
        merged_rows = []
        for i, row in enumerate(rows):
            max_codes_in_row = max_codes_per_row[i] if i < len(max_codes_per_row) else 4
            
            if self.debug:
                print(f"  Fila {i+1}: {len(row)} códigos (máx: {max_codes_in_row})")
            
            # APLICAR MÚLTIPLES RONDAS DE MERGE
            current_row = row
            round_num = 1
            
            while len(current_row) > max_codes_in_row:
                if self.debug:
                    print(f"    Ronda {round_num}: {len(current_row)} > {max_codes_in_row}, aplicando merge...")
                
                # Aplicar una ronda de merge
                merged_row = self._merge_row_by_distance(current_row, max_codes_in_row)
                
                # Verificar si se hizo progreso
                if len(merged_row) >= len(current_row):
                    if self.debug:
                        print(f"    ⚠️ No se pudo reducir más - forzando merge agresivo")
                    # Merge agresivo: combinar los 2 más cercanos
                    merged_row = self._force_aggressive_merge(current_row)
                
                current_row = merged_row
                round_num += 1
                
                if self.debug:
                    print(f"    Después ronda {round_num-1}: {len(current_row)} códigos")
                
                # Prevenir bucle infinito
                if round_num > 10:
                    if self.debug:
                        print(f"    ⚠️ Máximo de rondas alcanzado")
                    break
            
            if self.debug:
                if len(current_row) <= max_codes_in_row:
                    print(f"  ✅ Fila {i+1} completada: {len(row)} → {len(current_row)} (objetivo: {max_codes_in_row})")
                else:
                    print(f"  ⚠️ Fila {i+1} no completada: {len(row)} → {len(current_row)} (objetivo: {max_codes_in_row})")
            
            merged_rows.append(current_row)
        
        # Combinar filas
        final_rectangles = []
        for row in merged_rows:
            final_rectangles.extend(row)
        
        if self.debug:
            print(f"✅ MERGE MÚLTIPLES RONDAS COMPLETADO: {len(rectangles)} → {len(final_rectangles)}")
        
        return final_rectangles
    
    def map_rectangles_to_virtual_grid(self, merged_rectangles: List[Tuple], 
                                     grid_shape: Tuple[int, int], 
                                     image_shape: Tuple[int, int]) -> List[Optional[Tuple]]:
        """Mapea rectángulos a cuadrícula virtual respetando posiciones reales"""
        rows, cols = grid_shape
        h_img, w_img = image_shape[:2]
        
        if self.debug:
            print(f"🗂️ MAPEO A CUADRÍCULA VIRTUAL {rows}x{cols}")
            print(f"  Rectángulos a mapear: {len(merged_rectangles)}")
        
        # Crear cuadrícula vacía
        virtual_grid = [None] * (rows * cols)
        
        # Calcular dimensiones de celda teórica
        cell_width = w_img / cols
        cell_height = h_img / rows
        
        # Mapear cada rectángulo a su posición más probable
        for rect in merged_rectangles:
            x, y, w, h = rect
            
            # Calcular centro del rectángulo
            center_x = x + w / 2
            center_y = y + h / 2
            
            # Determinar posición en cuadrícula
            grid_col = int(center_x / cell_width)
            grid_row = int(center_y / cell_height)
            
            # Asegurar límites
            grid_col = max(0, min(cols - 1, grid_col))
            grid_row = max(0, min(rows - 1, grid_row))
            
            # Calcular índice lineal
            grid_index = grid_row * cols + grid_col
            
            if self.debug:
                print(f"    Rect ({x},{y},{w},{h}) -> centro ({center_x:.1f},{center_y:.1f}) -> grid[{grid_row},{grid_col}] = pos {grid_index}")
            
            # Resolver conflictos manteniendo el más centrado
            if virtual_grid[grid_index] is None:
                virtual_grid[grid_index] = rect
            else:
                # Mantener el más cercano al centro teórico de la celda
                expected_center_x = (grid_col + 0.5) * cell_width
                expected_center_y = (grid_row + 0.5) * cell_height
                
                current_distance = ((center_x - expected_center_x) ** 2 + 
                                  (center_y - expected_center_y) ** 2) ** 0.5
                
                existing_rect = virtual_grid[grid_index]
                existing_center_x = existing_rect[0] + existing_rect[2] / 2
                existing_center_y = existing_rect[1] + existing_rect[3] / 2
                existing_distance = ((existing_center_x - expected_center_x) ** 2 + 
                                   (existing_center_y - expected_center_y) ** 2) ** 0.5
                
                if current_distance < existing_distance:
                    virtual_grid[grid_index] = rect
                    if self.debug:
                        print(f"      Reemplazado por estar más centrado")
        
        # Mostrar visualización de la cuadrícula
        if self.debug:
            self._print_grid_visualization(virtual_grid, grid_shape)
        
        return virtual_grid
    
    def _print_grid_visualization(self, virtual_grid: List[Optional[Tuple]], grid_shape: Tuple[int, int]):
        """Visualización clara de la cuadrícula con huecos"""
        rows, cols = grid_shape
        
        print(f"  📊 CUADRÍCULA VIRTUAL:")
        for row in range(rows):
            row_str = f"    Fila {row+1}: "
            codes_in_row = 0
            
            for col in range(cols):
                index = row * cols + col
                if virtual_grid[index] is not None:
                    row_str += "🟩 "
                    codes_in_row += 1
                else:
                    row_str += "⬜ "
            
            row_str += f"({codes_in_row}/{cols} códigos)"
            print(row_str)
        
        total_codes = sum(1 for pos in virtual_grid if pos is not None)
        total_empty = len(virtual_grid) - total_codes
        print(f"  📈 RESUMEN: {total_codes} códigos, {total_empty} huecos")
    
    def optimize_grid_positioning(self, virtual_grid: List[Optional[Tuple]], 
                                 grid_shape: Tuple[int, int], 
                                 image_shape: Tuple[int, int]) -> List[Optional[Tuple]]:
        """Optimiza el posicionamiento en la cuadrícula para llenar huecos"""
        rows, cols = grid_shape
        h_img, w_img = image_shape[:2]
        
        if self.debug:
            print(f"🔧 OPTIMIZACIÓN DE POSICIONAMIENTO")
        
        optimized_grid = virtual_grid.copy()
        
        # Buscar huecos y códigos mal posicionados
        for row in range(rows):
            row_start = row * cols
            row_end = row_start + cols
            row_cells = optimized_grid[row_start:row_end]
            
            # Encontrar huecos en esta fila
            empty_positions = [i for i, cell in enumerate(row_cells) if cell is None]
            filled_positions = [i for i, cell in enumerate(row_cells) if cell is not None]
            
            if empty_positions and len(filled_positions) > 0:
                if self.debug:
                    print(f"  Fila {row+1}: {len(empty_positions)} huecos, {len(filled_positions)} códigos")
                
                # Intentar redistribuir códigos para llenar huecos
                self._redistribute_codes_in_row(optimized_grid, row, cols, empty_positions, filled_positions)
        
        return optimized_grid
    
    def _redistribute_codes_in_row(self, grid: List[Optional[Tuple]], row: int, cols: int,
                                  empty_positions: List[int], filled_positions: List[int]):
        """Redistribuye códigos en una fila para optimizar posicionamiento"""
        row_start = row * cols
        
        # Por ahora, implementación simple: no redistribuir
        # En el futuro se puede implementar lógica más sofisticada
        if self.debug:
            print(f"    Redistribución en fila {row+1}: manteniendo posiciones actuales")
    
    def _force_aggressive_merge(self, row_rectangles: List[Tuple]) -> List[Tuple]:
        """Merge agresivo: combina los 2 rectángulos más cercanos"""
        if len(row_rectangles) <= 1:
            return row_rectangles
        
        # Ordenar por X
        sorted_row = sorted(row_rectangles, key=lambda r: r[0])
        
        # Encontrar el par más cercano
        min_distance = float('inf')
        best_pair_idx = 0
        
        for i in range(len(sorted_row) - 1):
            x1, y1, w1, h1 = sorted_row[i]
            x2, y2, w2, h2 = sorted_row[i + 1]
            distance = x2 - (x1 + w1)  # Distancia horizontal
            
            if distance < min_distance:
                min_distance = distance
                best_pair_idx = i
        
        # Merge el par más cercano
        rect1 = sorted_row[best_pair_idx]
        rect2 = sorted_row[best_pair_idx + 1]
        merged_rect = self._merge_two_rectangles(rect1, rect2)
        
        # Crear nueva lista sin los 2 originales + el mergeado
        result = (sorted_row[:best_pair_idx] +
                 [merged_rect] +
                 sorted_row[best_pair_idx + 2:])
        
        if self.debug:
            print(f"    Merge agresivo: {rect1} + {rect2} → {merged_rect}")
        
        return result
    
    def _group_rectangles_by_rows(self, rectangles: List[Tuple]) -> List[List[Tuple]]:
        """Agrupa rectángulos por filas con tolerancia mejorada"""
        if not rectangles:
            return []
        
        # Ordenar por Y
        sorted_rects = sorted(rectangles, key=lambda r: r[1])
        
        rows = []
        current_row = [sorted_rects[0]]
        
        for i in range(1, len(sorted_rects)):
            current_rect = sorted_rects[i]
            last_rect = current_row[-1]
            
            # Calcular centros Y
            current_y = current_rect[1] + current_rect[3] / 2
            last_y = last_rect[1] + last_rect[3] / 2
            
            # Tolerancia adaptativa según altura promedio de la fila actual
            avg_height = np.mean([r[3] for r in current_row])
            tolerance = max(50, avg_height * 0.8)  # Mínimo 50px o 80% de altura promedio
            
            # Si están cerca en Y, misma fila
            if abs(current_y - last_y) < tolerance:
                current_row.append(current_rect)
            else:
                rows.append(current_row)
                current_row = [current_rect]
        
        if current_row:
            rows.append(current_row)
        
        # Ordenar cada fila por X
        for row in rows:
            row.sort(key=lambda r: r[0])
        
        return rows
    
    def _merge_row_by_distance(self, row_rectangles: List[Tuple], max_codes: int) -> List[Tuple]:
        """Merge por distancia (UNA RONDA) con criterios mejorados"""
        if len(row_rectangles) <= max_codes:
            return row_rectangles
        
        # Ordenar por X
        sorted_row = sorted(row_rectangles, key=lambda r: r[0])
        
        # Calcular distancias entre rectángulos consecutivos
        distances = []
        for i in range(len(sorted_row) - 1):
            x1, y1, w1, h1 = sorted_row[i]
            x2, y2, w2, h2 = sorted_row[i + 1]
            
            # Distancia horizontal
            horizontal_distance = x2 - (x1 + w1)
            
            # Factor de penalización por diferencia de tamaño
            size_diff = abs(w1 * h1 - w2 * h2) / max(w1 * h1, w2 * h2)
            
            # Puntuación combinada (menor = mejor para merge)
            combined_score = horizontal_distance + (size_diff * 100)
            
            distances.append((i, combined_score))
        
        # Ordenar por puntuación (menor = más probable de mergear)
        distances.sort(key=lambda x: x[1])
        
        # Merge los más cercanos (SOLO UNA RONDA)
        merged_rectangles = sorted_row.copy()
        merges_needed = min(len(sorted_row) - max_codes, len(distances))
        
        # Aplicar merges de más cercano a más lejano
        merges_applied = 0
        i = 0
        
        while merges_applied < merges_needed and i < len(distances):
            pair_idx, score = distances[i]
            
            # Verificar que el índice aún sea válido después de merges anteriores
            if pair_idx < len(merged_rectangles) - 1:
                rect1 = merged_rectangles[pair_idx]
                rect2 = merged_rectangles[pair_idx + 1]
                
                # Merge
                merged_rect = self._merge_two_rectangles(rect1, rect2)
                
                # Reemplazar en la lista
                merged_rectangles = (merged_rectangles[:pair_idx] +
                                   [merged_rect] +
                                   merged_rectangles[pair_idx + 2:])
                
                merges_applied += 1
                
                # Recalcular índices para los merges restantes
                for j in range(i + 1, len(distances)):
                    if distances[j][0] > pair_idx:
                        distances[j] = (distances[j][0] - 1, distances[j][1])
            
            i += 1
        
        return merged_rectangles
    
    def _merge_two_rectangles(self, rect1: Tuple, rect2: Tuple) -> Tuple:
        """Merge dos rectángulos creando bounding box que los englobe"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # Bounding box que englobe ambos
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1 + w1, x2 + w2)
        bottom = max(y1 + h1, y2 + h2)
        
        return (left, top, right - left, bottom - top)
    
    def validate_grid_layout(self, virtual_grid: List[Optional[Tuple]], 
                           grid_shape: Tuple[int, int],
                           expected_codes_per_row: List[int] = None) -> dict:
        """Valida que el layout de la cuadrícula sea correcto"""
        rows, cols = grid_shape
        
        if expected_codes_per_row is None:
            expected_codes_per_row = [4] * rows
        
        validation_results = {
            'total_codes': sum(1 for pos in virtual_grid if pos is not None),
            'total_empty': sum(1 for pos in virtual_grid if pos is None),
            'row_analysis': [],
            'layout_valid': True,
            'issues': []
        }
        
        for row in range(rows):
            row_start = row * cols
            row_end = row_start + cols
            row_cells = virtual_grid[row_start:row_end]
            
            codes_in_row = sum(1 for cell in row_cells if cell is not None)
            expected_in_row = expected_codes_per_row[row] if row < len(expected_codes_per_row) else 4
            
            row_info = {
                'row': row + 1,
                'codes_found': codes_in_row,
                'codes_expected': expected_in_row,
                'valid': codes_in_row <= expected_in_row
            }
            
            if not row_info['valid']:
                validation_results['layout_valid'] = False
                validation_results['issues'].append(f"Fila {row+1}: {codes_in_row} códigos > {expected_in_row} esperados")
            
            validation_results['row_analysis'].append(row_info)
        
        if self.debug:
            print(f"📋 VALIDACIÓN DE LAYOUT:")
            print(f"  Total códigos: {validation_results['total_codes']}")
            print(f"  Total huecos: {validation_results['total_empty']}")
            print(f"  Layout válido: {validation_results['layout_valid']}")
            if validation_results['issues']:
                for issue in validation_results['issues']:
                    print(f"  ⚠️ {issue}")
        
        return validation_results
