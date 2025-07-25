from sklearn.cluster import KMeans
import numpy as np

class VirtualGridDetector:
    """Detecta posiciones virtuales en cuadrícula farmacéutica usando análisis espacial"""
    
    def __init__(self, debug=False):
        self.debug = debug
    
    def detect_virtual_positions(self, detected_rectangles, grid_shape, image_shape, max_codes_per_row):
        """
        Detecta todas las posiciones (códigos + vacías) en la cuadrícula farmacéutica
        Basado en análisis de distribución espacial y interpolación geométrica
        """
        rows, cols = grid_shape
        h, w = image_shape[:2]
        
        if self.debug:
            print(f"🔍 DETECCIÓN CUADRÍCULA VIRTUAL:")
            print(f"   - Grid shape: {grid_shape}")
            print(f"   - Rectángulos detectados: {len(detected_rectangles)}")
            print(f"   - Códigos por fila: {max_codes_per_row}")
        
        if len(detected_rectangles) < 3:
            return self._fallback_uniform_grid(grid_shape, image_shape, max_codes_per_row)
        
        # PASO 1: Extraer centros de rectángulos detectados
        centers = []
        for rect in detected_rectangles:
            center_x = rect[0] + rect[2] // 2
            center_y = rect[1] + rect[3] // 2
            centers.append((center_x, center_y))
        
        # PASO 2: Agrupar por filas usando clustering
        y_coords = [center[1] for center in centers]
        row_clusters = self._cluster_y_coordinates(y_coords, rows)
        
        # PASO 3: Generar cuadrícula completa
        grid_positions = {}
        position_counter = 1
        
        for row_idx in range(rows):
            max_codes_this_row = max_codes_per_row[row_idx]
            
            # Encontrar puntos en esta fila
            target_y = row_clusters[row_idx] if row_idx < len(row_clusters) else None
            row_centers = self._get_centers_for_row(centers, target_y, tolerance=40)
            
            if len(row_centers) >= 2:
                # Interpolación basada en detecciones reales
                row_positions = self._interpolate_row_positions(
                    row_centers, max_codes_this_row, target_y
                )
            else:
                # Estimación uniforme para fila
                row_positions = self._estimate_uniform_row(
                    w, max_codes_this_row, target_y or (h * (row_idx + 1) / (rows + 1))
                )
            
            # Añadir posiciones de esta fila al grid
            for col_idx, (pos_x, pos_y) in enumerate(row_positions):
                if col_idx < max_codes_this_row:
                    bbox = self._estimate_bbox_from_position(pos_x, pos_y, detected_rectangles)
                    confidence = self._calculate_confidence(pos_x, pos_y, centers)
                    
                    grid_positions[position_counter] = {
                        'center': (int(pos_x), int(pos_y)),
                        'estimated_bbox': bbox,
                        'confidence': confidence,
                        'row': row_idx,
                        'col': col_idx
                    }
                    position_counter += 1
        
        if self.debug:
            real_detections = sum(1 for pos in grid_positions.values() if pos['confidence'] > 0.6)
            virtual_detections = len(grid_positions) - real_detections
            print(f"   - Posiciones reales (alta confianza): {real_detections}")
            print(f"   - Posiciones virtuales (estimadas): {virtual_detections}")
            print(f"   - Total posiciones: {len(grid_positions)}")
        
        return grid_positions
    
    def _cluster_y_coordinates(self, y_coords, n_clusters):
        """Agrupa coordenadas Y en clusters para identificar filas"""
        if len(y_coords) <= n_clusters:
            return sorted(y_coords)
        
        try:
            # Usar KMeans para clustering robusto
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            y_array = np.array(y_coords).reshape(-1, 1)
            clusters = kmeans.fit(y_array)
            centers = clusters.cluster_centers_.flatten()
            return sorted(centers)
        except:
            # Fallback: división uniforme
            return [min(y_coords) + i * (max(y_coords) - min(y_coords)) / (n_clusters - 1) 
                   for i in range(n_clusters)]
    
    def _get_centers_for_row(self, centers, target_y, tolerance=40):
        """Obtiene centros que pertenecen a una fila específica"""
        if target_y is None:
            return []
        
        row_centers = []
        for x, y in centers:
            if abs(y - target_y) <= tolerance:
                row_centers.append((x, y))
        
        return sorted(row_centers, key=lambda p: p[0])  # Ordenar por X
    
    def _interpolate_row_positions(self, row_centers, max_codes, target_y):
        """Interpola posiciones completas de una fila basándose en detecciones"""
        if len(row_centers) < 2:
            return []
        
        x_coords = [center[0] for center in row_centers]
        
        # Calcular espaciado promedio
        spacings = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
        avg_spacing = sum(spacings) / len(spacings)
        
        # Estimar posición inicial
        start_x = min(x_coords) - avg_spacing * ((max_codes - len(x_coords)) // 2)
        
        # Generar posiciones uniformes
        positions = []
        for i in range(max_codes):
            pos_x = start_x + i * avg_spacing
            positions.append((pos_x, target_y))
        
        return positions
    
    def _estimate_uniform_row(self, image_width, max_codes, y_position):
        """Estima posiciones uniformes para una fila"""
        positions = []
        for i in range(max_codes):
            pos_x = image_width * (i + 1) / (max_codes + 1)
            positions.append((pos_x, y_position))
        return positions
    
    def _estimate_bbox_from_position(self, pos_x, pos_y, detected_rectangles):
        """Estima bbox basándose en rectángulos detectados cercanos"""
        if not detected_rectangles:
            return [int(pos_x-40), int(pos_y-20), 80, 40]
        
        # Calcular tamaño promedio de rectángulos detectados
        avg_width = sum(rect[2] for rect in detected_rectangles) / len(detected_rectangles)
        avg_height = sum(rect[3] for rect in detected_rectangles) / len(detected_rectangles)
        
        x = int(pos_x - avg_width // 2)
        y = int(pos_y - avg_height // 2)
        
        return [x, y, int(avg_width), int(avg_height)]
    
    def _calculate_confidence(self, pos_x, pos_y, detected_centers):
        """Calcula confianza basada en proximidad a detecciones reales"""
        if not detected_centers:
            return 0.3
        
        min_distance = min([
            ((pos_x - dx)**2 + (pos_y - dy)**2)**0.5 
            for dx, dy in detected_centers
        ])
        
        # Confianza alta si está cerca de una detección real
        if min_distance < 50:
            return 0.9
        elif min_distance < 100:
            return 0.7
        elif min_distance < 150:
            return 0.5
        else:
            return 0.3
    
    def _fallback_uniform_grid(self, grid_shape, image_shape, max_codes_per_row):
        """Grid uniforme cuando hay muy pocas detecciones"""
        rows, cols = grid_shape
        h, w = image_shape[:2]
        
        grid_positions = {}
        position_counter = 1
        
        for row in range(rows):
            max_codes_this_row = max_codes_per_row[row]
            row_y = h * (row + 1) / (rows + 1)
            
            for col in range(max_codes_this_row):
                col_x = w * (col + 1) / (max_codes_this_row + 1)
                
                grid_positions[position_counter] = {
                    'center': (int(col_x), int(row_y)),
                    'estimated_bbox': [int(col_x-40), int(row_y-20), 80, 40],
                    'confidence': 0.2,  # Baja confianza para grid uniforme
                    'row': row,
                    'col': col
                }
                position_counter += 1
        
        return grid_positions