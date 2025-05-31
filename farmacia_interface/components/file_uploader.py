import streamlit as st
from PIL import Image
import io

class CustomFileUploader:
    """Uploader personalizado para imágenes farmacéuticas"""
    
    def __init__(self):
        self.accepted_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        self.max_size_mb = 10
    
    def create_uploader(self):
        """Crea uploader personalizado con validaciones"""
        
        st.markdown("""
        <div style="
            border: 2px dashed #cccccc;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            background-color: #f9f9f9;
            margin: 20px 0;
        ">
            <h4>📤 Arrastra tu imagen aquí o haz clic para seleccionar</h4>
            <p style="color: #666;">Formatos soportados: JPG, PNG, BMP, TIFF (máx. 10MB)</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Seleccionar imagen",
            type=self.accepted_formats,
            help="Sube una imagen de cupones farmacéuticos para procesar",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Validar tamaño
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            
            if file_size_mb > self.max_size_mb:
                st.error(f"❌ Archivo demasiado grande: {file_size_mb:.1f}MB. Máximo permitido: {self.max_size_mb}MB")
                return None
            
            # Validar que sea una imagen válida
            try:
                image = Image.open(uploaded_file)
                image.verify()  # Verificar integridad
                uploaded_file.seek(0)  # Reset para uso posterior
                
                st.success(f"✅ Imagen cargada correctamente: {uploaded_file.name}")
                return uploaded_file
                
            except Exception as e:
                st.error(f"❌ Error al cargar imagen: {str(e)}")
                return None
        
        return None
