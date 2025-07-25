# Procesamiento Automático de Cupones Precinto de Farmacia

_Un sistema de Visión por Computador para la digitalización y análisis de hojas de cupones precinto del Sistema Nacional de Salud español._

Este repositorio contiene el código fuente y los resultados del Trabajo Fin de Grado (TFG) "Procesamiento automático de cupones precinto de farmacia por medio de técnicas de aprendizaje profundo", desarrollado por Andrea Mayor Gómez[1].

## Índice

- [Visión General del Proyecto](#visión-general-del-proyecto)
- [Características Principales](#características-principales)
- [El Enfoque Híbrido: La Clave del Éxito](#el-enfoque-híbrido-la-clave-del-éxito)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Instalación](#instalación)
- [Uso](#uso)
- [Resultados Experimentales](#resultados-experimentales)
- [Licencia](#licencia)
- [Agradecimientos](#agradecimientos)

## Visión General del Proyecto

La gestión de cupones precinto en las farmacias españolas es un proceso manual, lento y susceptible a errores que representa uno de los últimos obstáculos para la digitalización completa del sector[1]. Este proyecto aborda este problema desarrollando una solución integral que automatiza la detección, extracción y validación de los códigos de barras de las hojas de cupones.

Se investigaron y compararon múltiples enfoques tecnológicos[1]:
1.  **Redes Neuronales:** Modelos como YOLOv10.
2.  **Procesamiento Clásico de Imágenes:** Un detector basado en análisis de gradientes.
3.  **Modelos Fundacionales (APIs):** GPT-4 Vision y Google Gemini.
4.  **Segmentación Universal:** SAM2.

La solución final es una **aplicación web interactiva** construida con Streamlit que implementa un **sistema híbrido**, combinando las fortalezas de YOLO y el análisis de gradientes para alcanzar una precisión y robustez superiores a cualquier método individual[1].

## Características Principales

-   **Sistema de Detección Híbrido:** Combina YOLO para el análisis de layout y el detector de gradientes para una localización precisa de códigos, logrando una **precisión media del 85%**[1].
-   **Corrección Posicional Avanzada:** Un algoritmo (`GridPositionCorrector`) que garantiza que cada código de barras se asigne a su celda correcta en la cuadrícula, incluso con detecciones parciales[1].
-   **Análisis Comparativo Exhaustivo:** El repositorio incluye los scripts y resultados de la evaluación de todos los métodos probados[1].
-   **Validación Regulatoria:** Integración con la API de **CIMA** (Centro de Información de Medicamentos de la AEMPS) para verificar la validez de los códigos extraídos[1].
-   **Interfaz Web Intuitiva:** Una aplicación desarrollada con Streamlit que permite la carga de imágenes por lotes, la visualización de resultados y la edición manual de forma sencilla[1].

## El Enfoque Híbrido: La Clave del Éxito

Los experimentos demostraron que ningún método individual era perfecto. La solución óptima fue un **orquestador inteligente** que combina lo mejor de dos mundos[1]:

1.  **Fase 1 (Análisis de Layout con YOLO):** Se utiliza YOLO para un análisis rápido de la imagen completa. Su principal función es detectar si existe una cabecera (`header`) manuscrita y localizar el área general (`barcode`) donde se encuentra la matriz de cupones. Esto determina si la cuadrícula es de 6x4 o 7x4[1].
2.  **Fase 2 (Detección Precisa con Gradientes):** El detector de gradientes, que es más lento pero muy preciso localizando las barras verticales, se aplica únicamente sobre el área de interés recortada por YOLO. Esto enfoca su potencia computacional y evita falsos positivos[1].
3.  **Fase 3 (Orquestación y Corrección):** El `Orchestrator` y el `GridPositionCorrector` fusionan los resultados, aplican la lógica de corrección posicional y ensamblan la cuadrícula final de códigos decodificados[1].

Este enfoque híbrido (implementado en `farmcode_detector.py`) demostró ser significativamente superior, alcanzando una precisión media del **85.0%**, frente al 64.8% de YOLO por sí solo y el 53.7% del detector de gradientes[1].

## Estructura del Repositorio

(```

pharmaceutical-barcode-scanner/
├── README.md
├── farmcode_detector/ # Aplicación web principal con el sistema híbrido
│ ├── streamlit_app.py
│ ├── requirements.txt
│ ├── core/ # Lógica de detección, procesamiento y orquestación
│ └── components/ # Módulos de la interfaz de Streamlit
├── data/ # Resultados de los experimentos (CSV, JSON)
├── datasets/ # Imágenes originales y anotaciones (YOLO, LabelMe)
├── runs/ # Modelos YOLO entrenados
├── sam2/ # Modelos y artefactos de SAM2
└── pruebas/ # Scripts para ejecutar y evaluar cada detector por separado

```)


## Instalación

Para ejecutar la aplicación principal y los scripts de prueba, sigue estos pasos:

1.  **Prerrequisitos:**
    -   Git
    -   Python 3.9 o superior
    -   Un gestor de entornos virtuales como `venv` o `conda`.

2.  **Clonar el repositorio:**
    ```
    git clone https://github.com/tu-usuario/pharmaceutical-barcode-scanner.git
    cd pharmaceutical-barcode-scanner
    ```

3.  **Crear y activar un entorno virtual:**
    ```
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

4.  **Instalar dependencias del sistema (para sistemas basados en Debian/Ubuntu):**
    Algunas librerías como OpenCV y ZBar requieren paquetes del sistema.
    ```
    sudo apt-get update
    sudo apt-get install -y $(cat farmcode_detector/packages.txt)
    ```

5.  **Instalar dependencias de Python:**
    ```
    pip install -r farmcode_detector/requirements.txt
    ```

6.  **(Opcional) Configurar claves de API:**
    Si deseas ejecutar las pruebas con los modelos de LLM (`gemini_detector.py`, `gpt4_detector.py`), necesitarás configurar tus claves de API como variables de entorno[1]:
    ```
    export GOOGLE_API_KEY="tu_clave_de_gemini"
    export OPENAI_API_KEY="tu_clave_de_openai"
    ```

## Uso

### Ejecutar la Aplicación Web

La aplicación principal implementa el sistema híbrido y ofrece una interfaz gráfica completa.


cd farmcode_detector/
python -m streamlit run streamlit_app.py


Navega a la URL local que se muestra en la terminal (normalmente `http://localhost:8501`) para empezar a procesar imágenes de cupones.

### Ejecutar los Scripts de Prueba

Los scripts en la carpeta `pruebas/` permiten ejecutar cada detector de forma individual. Son útiles para reproducir los resultados experimentales.

*Consulta cada script para ver sus argumentos específicos.*

## Resultados Experimentales

La evaluación comparativa de los métodos arrojó los siguientes resultados clave. Gemini demostró la mayor precisión, pero con una latencia y dependencia de API significativas. El sistema híbrido (`FarmCode`) ofrece el mejor equilibrio entre rendimiento, velocidad y autonomía[1].

| Métrica                   | **FarmCode (Híbrido)** | YOLOv10 (Local) | Gradientes (Local) | Gemini (API) | GPT-4 Vision (API) |
| ------------------------- | :--------------------: | :-------------: | :----------------: | :----------: | :----------------: |
| **Precisión Media**       |       **85.0%**[1]     |      64.8%[1]   |       53.7%[1]     |  **98.3%**[1]  |       24.7%[1]       |
| **Tiempo de Proc. (s)**   |       **~3.5s**[1]     |      ~3.0s[1]   |       ~1.5s[1]     |     ~9.5s[1]   |       ~6.6s[1]       |
| **Dependencia Externa**   |           No           |       No        |         No         |      Sí      |         Sí         |
| **Escenario Recomendado** |    **Producción**      |   Base Rápida   |      Respaldo      | Alta Precisión |  No Recomendado    |

*Los resultados detallados, imagen por imagen, se encuentran en `data/comparison/`.*

## Licencia

Este proyecto está distribuido bajo la Licencia MIT.

## Agradecimientos

-   **Autora:** Andrea Mayor Gómez[1]
-   **Tutores:** Javier Sánchez Pérez, Idafen Santana Pérez[1]
-   **Empresa Colaboradora:** [Farmalitics](https://www.farmalitics.com/), por proporcionar los datos y el contexto del problema[1].
