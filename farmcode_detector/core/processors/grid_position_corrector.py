import numpy as np

class GridPositionCorrector:
    """Corrector avanzado de posiciones con detección de huecos espaciales"""
    
    def __init__(self, debug=False):
        self.debug = debug
    
    def correct_grid_positions(self, decoded_results, grid_shape, image_shape, cropped_image=None):
        """
        Genera cuadrícula completa con detección inteligente de huecos
        """
        if self.debug:
            print(f"\n🔧 CORRECCIÓN AVANZADA DE POSICIONES")
            print("-" * 50)
        
        rows, cols = grid_shape
        if rows == 6:  # Con header
            total_positions = 24  # 6 filas × 4 columnas
        elif rows == 7:  # Sin header
            total_positions = 26  # 6 filas completas (24) + 2 en última fila
        else:
            total_positions = rows * cols  # Otros casos
        
        # PASO 1: Extraer códigos válidos con sus bboxes
        valid_detections = {}
        for pos, result in decoded_results.items():
            if result['code'] != "No detectado" and result['bbox'] is not None:
                valid_detections[pos] = {
                    'code': result['code'],
                    'bbox': result['bbox'],
                    'center': self._calculate_center(result['bbox']),
                    'confidence': result.get('confidence', 1.0),
                    'method': result['method']
                }
        
        if self.debug:
            print(f"   Detecciones válidas con bbox: {len(valid_detections)}")
            print(f"   Total posiciones esperadas: {total_positions}")
        
        # DECIDIR MÉTODO BASADO EN PORCENTAJE DE DETECCIÓN
        detection_ratio = len(valid_detections) / total_positions
        
        if detection_ratio < 0.5 and cropped_image is not None:
            # MENOS DEL 50%: Usar cuadrícula estática ajustada a líneas de la hoja
            virtual_grid = self._generate_static_grid_adjusted(cropped_image, grid_shape, image_shape)
            if self.debug:
                print(f"   Ratio detección: {detection_ratio:.2%} - Usando grid estático ajustado")
        else:
            # MÁS DEL 50%: Usar método actual sin cambios
            virtual_grid = self._generate_virtual_grid(valid_detections, grid_shape, image_shape)
            if self.debug:
                print(f"   Ratio detección: {detection_ratio:.2%} - Usando grid dinámico actual")
        
        # PASO 3: Mapear códigos a posiciones virtuales
        mapped_results = self._map_codes_to_virtual_positions(
            valid_detections, virtual_grid, total_positions
        )
        
        if self.debug:
            filled_positions = sum(1 for r in mapped_results.values() 
                                 if r['code'] != "No detectado")
            print(f"   Posiciones finales llenas: {filled_positions}/{total_positions}")
        
        return mapped_results
    
    def _generate_static_grid_adjusted(self, cropped_image, grid_shape, image_shape):
        """
        Genera cuadrícula estática ajustada a las líneas visibles de la hoja
        """
        rows, cols = grid_shape
        h, w = image_shape[:2]
        
        try:
            # Intentar detectar líneas de cuadrícula usando OpenCV
            import cv2
            
            # Preprocesamiento para detectar líneas
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            
            # Mejorar contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Detectar bordes
            edges = cv2.Canny(enhanced, 30, 100, apertureSize=3)
            
            # Detectar líneas con Hough Transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=int(min(h, w) * 0.2))
            
            if lines is not None and len(lines) >= 6:
                horizontal_lines, vertical_lines = self._process_detected_lines(lines, h, w)
                
                if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
                    return self._create_grid_from_lines(horizontal_lines, vertical_lines, grid_shape)
            
            if self.debug:
                print("   No se pudieron detectar líneas suficientes, usando grid uniforme mejorado")
                
        except ImportError:
            if self.debug:
                print("   OpenCV no disponible, usando grid uniforme mejorado")
        except Exception as e:
            if self.debug:
                print(f"   Error en detección de líneas: {e}, usando grid uniforme mejorado")
        
        # Fallback: Grid uniforme mejorado
        return self._create_enhanced_uniform_grid(grid_shape, image_shape)
    
    def _process_detected_lines(self, lines, height, width):
        """
        Procesa líneas detectadas y las clasifica en horizontales y verticales
        """
        horizontal_rhos = []
        vertical_rhos = []
        
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta)
            
            # Clasificar líneas por ángulo
            if abs(angle_deg) < 15 or abs(angle_deg - 180) < 15:
                # Líneas horizontales
                y_intercept = abs(rho / np.sin(theta)) if abs(np.sin(theta)) > 0.1 else rho
                if 0 <= y_intercept <= height:
                    horizontal_rhos.append(y_intercept)
                    
            elif abs(angle_deg - 90) < 15:
                # Líneas verticales  
                x_intercept = abs(rho / np.cos(theta)) if abs(np.cos(theta)) > 0.1 else rho
                if 0 <= x_intercept <= width:
                    vertical_rhos.append(x_intercept)
        
        # Agrupar líneas similares
        h_lines = self._cluster_similar_lines(horizontal_rhos, tolerance=height*0.02)
        v_lines = self._cluster_similar_lines(vertical_rhos, tolerance=width*0.02)
        
        return sorted(h_lines), sorted(v_lines)
    
    def _cluster_similar_lines(self, line_positions, tolerance=10):
        """
        Agrupa líneas que están muy cerca entre sí
        """
        if not line_positions:
            return []
            
        sorted_lines = sorted(line_positions)
        clustered = [sorted_lines[0]]
        
        for line_pos in sorted_lines[1:]:
            if abs(line_pos - clustered[-1]) <= tolerance:
                # Promedio ponderado
                clustered[-1] = (clustered[-1] + line_pos) / 2
            else:
                clustered.append(line_pos)
        
        return clustered
    
    def _create_grid_from_lines(self, h_lines, v_lines, grid_shape):
        """
        Crea cuadrícula basada en las intersecciones de líneas detectadas
        """
        rows, cols = grid_shape
        virtual_grid = {}
        
        # Asegurar que tenemos suficientes líneas
        if len(h_lines) < rows + 1:
            h_lines = self._interpolate_missing_lines(h_lines, rows + 1)
        if len(v_lines) < cols + 1:
            v_lines = self._interpolate_missing_lines(v_lines, cols + 1)
        
        position = 1
        for row_idx in range(rows):
            codes_in_row = cols if row_idx < 6 else (2 if rows == 7 else cols)
            
            for col_idx in range(codes_in_row):
                if row_idx < len(h_lines) - 1 and col_idx < len(v_lines) - 1:
                    # Calcular centro de celda
                    y1, y2 = h_lines[row_idx], h_lines[row_idx + 1]
                    x1, x2 = v_lines[col_idx], v_lines[col_idx + 1]
                    
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Dimensiones de celda (80% del espacio disponible)
                    cell_w = int(abs(x2 - x1) * 0.8)
                    cell_h = int(abs(y2 - y1) * 0.8)
                    
                    virtual_grid[position] = {
                        'center': (center_x, center_y),
                        'bbox': [center_x - cell_w//2, center_y - cell_h//2, cell_w, cell_h],
                        'confidence': 0.8,
                        'source': 'static_lines_adjusted'
                    }
                
                position += 1
        
        return virtual_grid
    
    def _interpolate_missing_lines(self, existing_lines, target_count):
        """
        Interpola líneas faltantes para completar la cuadrícula
        """
        if len(existing_lines) >= target_count:
            return existing_lines[:target_count]
        
        if len(existing_lines) < 2:
            # No hay suficientes líneas para interpolar
            return existing_lines
        
        # Interpolar líneas entre las existentes
        result = existing_lines.copy()
        while len(result) < target_count:
            # Encontrar el espacio más grande entre líneas consecutivas
            max_gap = 0
            insert_idx = 0
            
            for i in range(len(result) - 1):
                gap = result[i + 1] - result[i]
                if gap > max_gap:
                    max_gap = gap
                    insert_idx = i
            
            # Insertar línea en el medio del espacio más grande
            new_line = (result[insert_idx] + result[insert_idx + 1]) / 2
            result.insert(insert_idx + 1, new_line)
        
        return sorted(result)
    
    def _create_enhanced_uniform_grid(self, grid_shape, image_shape):
        """
        Grid uniforme mejorado para cuando no se pueden detectar líneas
        """
        rows, cols = grid_shape
        h, w = image_shape[:2]
        
        grid = {}
        
        # Márgenes más realistas
        margin_x = w * 0.08  # 8% de margen horizontal
        margin_y = h * 0.08  # 8% de margen vertical
        
        usable_width = w - 2 * margin_x
        usable_height = h - 2 * margin_y
        
        cell_width = usable_width / cols
        cell_height = usable_height / rows
        
        position = 1
        for row_idx in range(rows):
            codes_in_row = cols if row_idx < 6 else (2 if rows == 7 else cols)
            
            for col_idx in range(codes_in_row):
                # Centro de la celda
                center_x = int(margin_x + (col_idx + 0.5) * cell_width)
                center_y = int(margin_y + (row_idx + 0.5) * cell_height)
                
                # Dimensiones (75% del espacio disponible)
                bbox_w = int(cell_width * 0.75)
                bbox_h = int(cell_height * 0.75)
                
                grid[position] = {
                    'center': (center_x, center_y),
                    'bbox': [center_x - bbox_w//2, center_y - bbox_h//2, bbox_w, bbox_h],
                    'confidence': 0.6,
                    'source': 'enhanced_uniform'
                }
                
                position += 1
        
        return grid
    
    # MANTENER TODOS LOS MÉTODOS EXISTENTES SIN CAMBIOS
    def _generate_virtual_grid(self, valid_detections, grid_shape, image_shape):
        """
        Genera cuadrícula virtual basada en análisis espacial de códigos detectados
        """
        rows, cols = grid_shape
        h, w = image_shape[:2]
        
        if len(valid_detections) < 3:
            # Fallback: grid uniforme
            return self._create_uniform_grid(grid_shape, image_shape)
        
        # Extraer centros de códigos detectados
        centers = [det['center'] for det in valid_detections.values()]
        x_coords = [center[0] for center in centers]
        y_coords = [center[1] for center in centers]
        
        # ANÁLISIS DE FILAS
        y_clusters = self._cluster_coordinates(y_coords, max_clusters=rows)
        y_centers = sorted(y_clusters)
        
        # ANÁLISIS DE COLUMNAS
        x_clusters = self._cluster_coordinates(x_coords, max_clusters=cols)
        x_centers = sorted(x_clusters)
        
        # Generar cuadrícula virtual completa
        virtual_grid = {}
        
        for row_idx in range(rows):
            # Determinar número de códigos en esta fila
            codes_in_row = cols if row_idx < 6 else (2 if rows == 7 else cols)
            
            for col_idx in range(codes_in_row):
                position = row_idx * cols + col_idx + 1
                
                # Calcular posición estimada
                if row_idx < len(y_centers):
                    est_y = y_centers[row_idx]
                else:
                    # Interpolar para filas faltantes
                    est_y = self._interpolate_y_position(row_idx, y_centers, h)
                
                if col_idx < len(x_centers):
                    est_x = x_centers[col_idx]
                else:
                    # Interpolar para columnas faltantes
                    est_x = self._interpolate_x_position(col_idx, x_centers, w)
                
                # Estimar bbox para esta posición
                est_bbox = self._estimate_bbox_from_position(
                    est_x, est_y, valid_detections
                )
                
                virtual_grid[position] = {
                    'center': (est_x, est_y),
                    'bbox': est_bbox,
                    'confidence': self._calculate_position_confidence(
                        est_x, est_y, centers
                    )
                }
        
        return virtual_grid
    
    def _cluster_coordinates(self, coords, max_clusters):
        """
        Agrupa coordenadas usando clustering por distancia
        """
        if len(coords) <= max_clusters:
            return coords
        
        # Clustering simple mejorado
        sorted_coords = sorted(coords)
        clusters = [sorted_coords[0]]
        
        # Calcular umbral dinámico basado en distribución
        coord_range = max(sorted_coords) - min(sorted_coords)
        threshold = coord_range / (max_clusters * 2)  # Umbral adaptativo
        
        for coord in sorted_coords[1:]:
            if coord - clusters[-1] > threshold:
                clusters.append(coord)
            else:
                # Promedio ponderado para actualizar cluster
                clusters[-1] = (clusters[-1] + coord) / 2
        
        return clusters[:max_clusters]
    
    def _interpolate_y_position(self, row_idx, y_centers, image_height):
        """
        Interpola posición Y para filas no detectadas
        """
        if len(y_centers) >= 2:
            # Detectar si los códigos están en la parte inferior
            first_y = y_centers[0]
            last_y = y_centers[-1]
            
            # Si el primer código está en la mitad inferior de la imagen
            if first_y > image_height * 0.5:
                # Los códigos están concentrados abajo - extrapolar hacia arriba
                avg_spacing = (last_y - first_y) / (len(y_centers) - 1) if len(y_centers) > 1 else image_height / 8
                # Calcular posición extrapolando desde el primer código detectado
                return first_y - (len(y_centers) - 1 - row_idx) * avg_spacing
            else:
                # Comportamiento original - interpolación normal
                spacing = (y_centers[-1] - y_centers[0]) / (len(y_centers) - 1)
                return y_centers[0] + row_idx * spacing
        else:
            # Distribución uniforme (sin cambios)
            return int(image_height * (row_idx + 1) / 8)
    
    def _interpolate_x_position(self, col_idx, x_centers, image_width):
        """
        Interpola posición X para columnas no detectadas
        """
        if len(x_centers) >= 2:
            spacing = (x_centers[-1] - x_centers[0]) / (len(x_centers) - 1)
            return x_centers[0] + col_idx * spacing
        else:
            return int(image_width * (col_idx + 1) / 5)
    
    def _estimate_bbox_from_position(self, est_x, est_y, valid_detections):
        """
        Estima bbox basado en bboxes de códigos detectados
        """
        if not valid_detections:
            return [est_x - 40, est_y - 20, 80, 40]
        
        # Calcular dimensiones promedio de códigos detectados
        widths = [det['bbox'][2] for det in valid_detections.values()]
        heights = [det['bbox'][3] for det in valid_detections.values()]
        
        avg_width = int(np.mean(widths)) if widths else 80
        avg_height = int(np.mean(heights)) if heights else 40
        
        return [
            int(est_x - avg_width // 2),
            int(est_y - avg_height // 2),
            avg_width,
            avg_height
        ]
    
    def _map_codes_to_virtual_positions(self, valid_detections, virtual_grid, total_positions):
        """
        Mapea códigos detectados a posiciones virtuales más cercanas
        """
        mapped_results = {}
        used_positions = set()

        # DEFINIR POSICIONES VÁLIDAS SEGÚN EL LAYOUT
        if total_positions == 24:  # Con header
            valid_positions = set(range(1, 25))  # Posiciones 1-24
        elif total_positions == 26:  # Sin header
            valid_positions = set(range(1, 25)) | {27, 28}  # 1-24 + 27,28 (saltar 25,26)
        else:
            valid_positions = set(range(1, total_positions + 1))  # Otros casos
        
        # Inicializar todas las posiciones como vacías
        for position in range(1, total_positions + 1):
            mapped_results[position] = {
                'code': "No detectado",
                'bbox': virtual_grid.get(position, {}).get('bbox'),
                'confidence': 0.0,
                'method': 'virtual_grid_empty',
                'position': position
            }
        
        # Mapear códigos detectados a posiciones virtuales más cercanas
        for det_pos, detection in valid_detections.items():
            det_center = detection['center']
            
            # Encontrar posición virtual más cercana no utilizada
            best_position = None
            min_distance = float('inf')
            
            for virt_pos, virt_info in virtual_grid.items():
                if virt_pos in used_positions:
                    continue
                
                virt_center = virt_info['center']
                distance = np.sqrt(
                    (det_center[0] - virt_center[0])**2 + 
                    (det_center[1] - virt_center[1])**2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    best_position = virt_pos
            
            # Asignar código a la mejor posición
            if best_position is not None:
                mapped_results[best_position] = {
                    'code': detection['code'],
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'method': f"virtual_mapped_from_{det_pos}",
                    'position': best_position,
                    'mapping_distance': min_distance
                }
                used_positions.add(best_position)
        
        return mapped_results
    
    def _calculate_center(self, bbox):
        """Calcula centro de un bounding box"""
        if bbox is None:
            return (0, 0)
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
    
    def _calculate_position_confidence(self, est_x, est_y, detected_centers):
        """Calcula confianza de posición virtual basada en proximidad"""
        if not detected_centers:
            return 0.3
        
        min_distance = min([
            np.sqrt((est_x - center[0])**2 + (est_y - center[1])**2)
            for center in detected_centers
        ])
        
        # Confianza inversamente proporcional a la distancia
        confidence = max(0.1, 1.0 - min_distance / 200)
        return confidence
    
    def _create_uniform_grid(self, grid_shape, image_shape):
        """Grid uniforme como fallback"""
        rows, cols = grid_shape
        h, w = image_shape[:2]
        
        grid = {}
        
        for row in range(rows):
            codes_in_row = cols if row < 6 else (2 if rows == 7 else cols)
            for col in range(codes_in_row):
                position = row * cols + col + 1
                
                est_x = int(w * (col + 1) / (codes_in_row + 1))
                est_y = int(h * (row + 1) / (rows + 1))
                
                grid[position] = {
                    'center': (est_x, est_y),
                    'bbox': [est_x - 40, est_y - 20, 80, 40],
                    'confidence': 0.3
                }
        
        return grid
