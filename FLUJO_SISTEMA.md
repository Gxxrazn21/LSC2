# Sistema de Reconocimiento de Lengua de Señas Colombiana (LSC)

## Objetivo

Detectar señas de saludo (HOLA, BUENAS, DIAS, NOCHES, TARDES) en tiempo real usando cámara, y pronunciarlas en voz alta 3 veces al ser identificadas con suficiente confianza.

---

## Flujo del Sistema

```
Cámara (OpenCV)
      │
      ▼
Voltear frame horizontalmente
      │
      ├──► MediaPipe FaceDetector
      │         Detecta si hay un rostro en escena
      │         Dibuja bounding box (naranja)
      │
      ├──► MediaPipe HandLandmarker
      │         Detecta 21 landmarks de la mano
      │         Dibuja esqueleto (verde)
      │
      ▼
  ¿Hay mano?
  ┌── Sí ──► recortar_crop()  →  imagen cuadrada de la mano
  │              │
  │              ▼
  │         Normalizar con transforms (Resize 128×128, ToTensor, Normalize)
  │              │
  │              ▼
  │         LSC_CNN (PyTorch)
  │              │ softmax → probabilidad por clase
  │              ▼
  │         ¿prob >= UMBRAL_CONFIANZA (70%)?
  │         ┌── Sí ──► Acumular frames estables (FRAMES_ESTABLES = 10)
  │         │              │
  │         │              ▼
  │         │          ¿10 frames consecutivos con misma predicción?
  │         │          ┌── Sí ──► texto_para_voz()  → "hola hola hola"
  │         │          │              │
  │         │          │              ▼
  │         │          │          hablar_async()  → cola TTS (hilo separado)
  │         │          │          pyttsx3 pronuncia la frase
  │         │          └── No ──► Seguir acumulando
  │         └── No ──► label "Analizando..."
  └── No  ──► label "Muestra una mano" / "Esperando seña..."
      │
      ▼
Dibujar panel superior (FPS, estado mano/rostro, nº clases)
Dibujar subtítulo inferior (etiqueta + confianza)
Mostrar frame con cv2.imshow()
      │
      ▼
Tecla [q] → Cerrar cámara y salir
```

---

## Tecnologías y sus Funciones

| Tecnología | Versión recomendada | Función en el sistema |
|---|---|---|
| **Python** | 3.11+ | Lenguaje base del proyecto |
| **PyTorch** | ≥ 2.0 | Entrenamiento e inferencia de la CNN |
| **torchvision** | ≥ 0.15 | Transformaciones de imagen para train/val/inferencia |
| **OpenCV (cv2)** | ≥ 4.8 | Captura de cámara, volteo, dibujo de overlays y bounding boxes |
| **MediaPipe** | ≥ 0.10 (Tasks API) | Detección de manos (HandLandmarker) y rostro (FaceDetector) |
| **pyttsx3** | ≥ 2.90 | Motor TTS (text-to-speech) local, sin internet, en hilo separado |
| **Pillow (PIL)** | ≥ 10.0 | Conversión de arrays NumPy a formato PIL para transforms de torchvision |
| **NumPy** | ≥ 1.24 | Operaciones matriciales, shuffling de índices para train/val split |

---

## Archivos del Proyecto

```
ENTRANAMIENTO/
├── config.py                  Todos los hiperparámetros y rutas (carga .env)
├── requirements.txt           Dependencias del proyecto
├── hand_landmarker.task       Modelo MediaPipe para landmarks de mano
├── face_detector.task         Modelo MediaPipe para detección de rostro
│
├── fase1_cnn/
│   ├── contextos.py           Define las clases de señas y traducciones TTS
│   ├── dataset.py             LSCDataset: carga imágenes del dataset LSC70
│   ├── modelo.py              LSC_CNN: arquitectura de la red convolucional
│   ├── entrenar.py            Bucle de entrenamiento con early stopping
│   ├── evaluar.py             Evaluación con matriz de confusión
│   ├── predecir.py            Inferencia sobre una imagen estática
│   └── predecir_voz.py        Reconocimiento en tiempo real + TTS
│
├── utils/
│   ├── crop_mano.py           recortar_crop(): extrae ROI cuadrada de la mano
│   ├── metricas.py            calcular_accuracy, reporte_clasificacion
│   ├── preprocesar_manos.py   Utilidades de preprocesado offline
│   └── visualizacion.py       graficar_curvas(): curvas loss/accuracy
│
├── modelos_guardados/
│   ├── mejor_modelo_saludos.pth   Pesos entrenados de la CNN
│   └── clases_saludos.json        Lista ordenada de clases ["BUENAS","DIAS",...]
│
└── datasets/
    └── LSC70_HAND_CROPS/      Dataset de crops de manos (necesario para entrenar)
```

---

## Módulos Clave — Funciones Principales

### `config.py`
| Variable/Función | Descripción |
|---|---|
| `FASE1` | Dict con batch_size, epochs, lr, img_size, rutas |
| `UMBRAL_CONFIANZA` | Probabilidad mínima para aceptar predicción (default 0.70) |
| `FRAMES_ESTABLES` | Frames consecutivos requeridos antes de hablar (default 10) |
| `COOLDOWN_TTS_SEG` | Segundos mínimos entre pronunciaciones de la misma palabra (default 1.5) |
| `TTS_RATE` | Velocidad del TTS en palabras por minuto (default 160) |
| `resumen()` | Imprime configuración activa en consola |

### `contextos.py`
| Función | Descripción |
|---|---|
| `clases_de(contexto)` | Devuelve lista de clases: `["HOLA","BUENAS","DIAS","NOCHES","TARDES"]` |
| `rutas_de(contexto)` | Devuelve dict con rutas a modelo .pth y clases .json |
| `texto_para_voz(clase)` | Convierte clase a texto TTS repetido 3 veces: `"hola hola hola"` |

### `modelo.py`
| Clase/Función | Descripción |
|---|---|
| `LSC_CNN` | CNN: 3 bloques Conv2d+BN+ReLU+MaxPool → AdaptiveAvgPool2d(4) → Linear(256) → Linear(n_clases) |
| `crear_modelo(n_clases, img_size, device)` | Instancia y mueve modelo al dispositivo correcto |

### `dataset.py`
| Clase/Método | Descripción |
|---|---|
| `LSCDataset.__init__` | Carga rutas de imágenes filtrando por `target_gestures` |
| `LSCDataset.__getitem__` | Abre imagen PIL, aplica transform, devuelve (tensor, label) |

### `entrenar.py`
| Función | Descripción |
|---|---|
| `entrenar(contexto, epochs, batch_size, lr)` | Loop completo: carga datos, Adam + ReduceLROnPlateau, early stopping (12 épocas), guarda mejor modelo |
| `_build_loaders(...)` | Construye DataLoaders train/val con augmentación en train |

### `predecir_voz.py`
| Función | Descripción |
|---|---|
| `_tts_worker(rate)` | Hilo daemon: consume cola TTS y llama pyttsx3 |
| `hablar_async(texto)` | Encola texto para pronunciación sin bloquear el loop de cámara |
| `cargar_modelo(img_size)` | Carga pesos .pth y lista de clases del modelo saludos |
| `dibujar_landmarks_mano(frame, ...)` | Dibuja 21 puntos y conexiones de la mano en verde |
| `dibujar_panel_superior(frame, ...)` | Overlay superior con estado Mano/Rostro/FPS |
| `dibujar_subtitulo(frame, label)` | Overlay inferior con la predicción activa |
| `predecir_tiempo_real(camera_index)` | Pipeline principal: captura → detección → CNN → TTS |

### `utils/crop_mano.py`
| Función | Descripción |
|---|---|
| `recortar_crop(frame, hand_landmarks, out_size)` | Calcula bounding box de los 21 landmarks, añade margen, devuelve crop cuadrado redimensionado |

---

## Entrenamiento

```bash
# Entrenar el modelo de saludos (HOLA, BUENAS, DIAS, NOCHES, TARDES)
python -m fase1_cnn.entrenar

# Con parámetros personalizados
python -m fase1_cnn.entrenar --epochs 40 --batch-size 16 --lr 0.0005

# Evaluar con matriz de confusión
python -m fase1_cnn.evaluar
```

**Nota:** El dataset `LSC70_HAND_CROPS` debe estar en `datasets/LSC70_HAND_CROPS/` con subcarpetas por clase (HOLA, BUENAS, DIAS, NOCHES, TARDES).

---

## Reconocimiento en Vivo

```bash
python -m fase1_cnn.predecir_voz

# Con cámara específica
python -m fase1_cnn.predecir_voz --camera 1
```

**Comportamiento TTS:** Al detectar una seña con ≥70% de confianza durante 10 frames consecutivos, el sistema pronuncia la palabra **3 veces** (ej: `"hola hola hola"`). No repite hasta que pasen 1.5 segundos o se cambie de seña. Tras 8 frames sin mano, el cooldown se reinicia.

---

## Variables de Entorno (.env)

```env
# Rutas
LSC70_PATH=datasets/LSC70_HAND_CROPS/LSC70/LSC70
LSC_MODELOS_DIR=modelos_guardados
LSC_RESULTADOS_DIR=resultados

# Cámara y TTS
CAMERA_INDEX=0
UMBRAL_CONFIANZA=0.70
FRAMES_ESTABLES=10
COOLDOWN_TTS_SEG=1.5
TTS_RATE=160

# Entrenamiento
FASE1_EPOCHS=30
FASE1_BATCH_SIZE=32
FASE1_LR=0.001
FASE1_IMG_SIZE=128

# Dispositivo
LSC_DEVICE=cpu   # o cuda
```
