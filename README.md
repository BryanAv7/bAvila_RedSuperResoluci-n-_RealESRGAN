# Real-ESRGAN Video Super Resolution 

---

## Descripción

Este parte del trabajo se implementa un sistema de super resolución en video utilizando el modelo **Real-ESRGAN x2** con OpenCV en C++.  
El sistema procesa un video de entrada, aplica super resolución frame por frame y muestra el video resultante en tiempo real.

Se ha implementado soporte para ejecución tanto en **CPU** como en **GPU (CUDA)** para aprovechar la aceleración por hardware. La configuración permite elegir fácilmente el dispositivo para la inferencia.

---

## Características principales

- Carga y procesamiento de video con OpenCV.  
- Aplicación del modelo Real-ESRGAN para aumentar la resolución de cada frame.  
- Soporte para ejecución en GPU con CUDA y CPU según configuración.  
- Visualización de video con FPS superpuestos para monitorear el rendimiento.   

---

## Requisitos

- C++17 o superior  
- OpenCV (con soporte para video)  
- ONNX Runtime (versiones CPU y GPU)  
- GPU con soporte CUDA (opcional para aceleración)  
- Sistema operativo Linux (probado en Ubuntu)  

---

## Estructura de archivos

├── models/
│ └── RealESRGAN_x2.onnx # Modelo ONNX preentrenado
├── video5.mp4 # Video de prueba
├── principal.cpp # Código fuente principal
├── CMakeLists.txt # Archivo de construcción CMake
├── README.md # Este archivo

---

## Uso

1. Clonar este repositorio.  
2. Asegurar que las dependencias estén instaladas (OpenCV, ONNX Runtime).  
3. Ajustar las rutas en `principal.cpp` para el modelo ONNX y el video de entrada.  
4. Compilar con CMake o tu método preferido.  
5. Ejecutar el programa.

Por defecto, el programa intentará usar GPU. Para usar solo CPU, comenta o desactiva la sección correspondiente en el código.

---

## Resultados y recomendaciones

La ejecución en **GPU** proporciona un rendimiento significativamente superior debido a la alta carga computacional del modelo Real-ESRGAN, logrando un procesamiento más fluido y cercano a tiempo real.

La inferencia en **CPU** es funcional pero mucho más lenta, lo que puede limitar la usabilidad en aplicaciones en vivo.

**Recomendación:** Se recomienda usar una GPU moderna con soporte CUDA para obtener el mejor rendimiento en aplicaciones de super resolución de video en tiempo real.

