
# Guía de Instalación, Entrenamiento y Ejecución del Modelo LSC

Esta guía cubre: entorno, dependencias, datasets, entrenamiento, evaluación y
reconocimiento en tiempo real por cámara con voz y subtítulos.

---

## 1. Requisitos previos

- **Python 3.11** (recomendado). **NO** usar Python 3.13: `mediapipe 0.10.x` no lo soporta.
- Sistema operativo: Windows 10/11, macOS o Linux.
- Opcional: GPU NVIDIA con CUDA 11.8/12.1 para acelerar el entrenamiento.

Para verificar tu versión:
```bash
python --version
```

Si tienes Python 3.13 instalado, instala Python 3.11 desde
https://www.python.org/downloads/release/python-3119/ y úsalo con `py -3.11 ...`.

---

## 2. Configuración del Entorno Virtual

> El `venv_lsc` incluido en el repo fue creado con Python 3.13; **elimínalo** y
> crea uno nuevo con 3.11 si no lo has hecho ya.

### Crear un entorno virtual limpio (Windows PowerShell)

```powershell
# En la raiz del proyecto
py -3.11 -m venv venv_lsc

# Activar
.\venv_lsc\Scripts\Activate.ps1

# Si aparece error de ejecucion de scripts, primero:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
```

### Crear un entorno virtual (CMD)

```cmd
py -3.11 -m venv venv_lsc
.\venv_lsc\Scripts\activate.bat
```

### Crear un entorno virtual (macOS / Linux)

```bash
python3.11 -m venv venv_lsc
source venv_lsc/bin/activate
```

### Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Importante**: si ya tenías instalado `opencv-contrib-python`, desinstálalo antes:
> `pip uninstall -y opencv-contrib-python`. No se pueden tener ambos paquetes.

---

## 3. Datasets esperados

Estructura base:

```
datasets/
├── LSC70/LSC70ANH/...   (Fase 1: imágenes RGB)
├── LSC-54/              (Fase 2: JSON de landmarks temporales)
└── LSC50/               (Fase 3: video + IMU + landmarks)
```

- **Fase 1**: actualmente el código busca recursivamente `LSC70W/Per*/<GESTO>/*.jpg`
  bajo `datasets/LSC70/LSC70ANH`.
- **Fase 2 y 3**: aún sin datos locales. Coloca los JSON/CSV/MP4 conforme
  la estructura descrita en cada `dataset.py`.

---

## 4. Descargar los modelos de MediaPipe (detección de manos y rostro)

Dos archivos `.task` deben estar en la raíz del proyecto:

| Archivo | Función | URL |
|---------|---------|-----|
| `hand_landmarker.task` | Detección de 21 landmarks de manos | https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task |
| `face_detector.task` | Detección de rostro (BlazeFace) | https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite |

Descarga con `curl`:

```bash
curl -L -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
curl -L -o face_detector.task "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
```

---

## 5. Entrenar los modelos

### 5.1. Pre-procesamiento de manos (obligatorio antes de entrenar Fase 1)

El dataset LSC70 trae imágenes completas con **caras censuradas**, ropa y fondos
que cambian de una persona a otra — todo ruido para el modelo. Antes de entrenar
generamos **crops mano+antebrazo** con MediaPipe HandLandmarker para que la CNN
sólo vea la parte informativa de la seña.

```bash
python -m utils.preprocesar_manos
```

Esto recorre `datasets/LSC70/LSC70ANH/**/*.jpg`, detecta las manos y guarda los
crops en `datasets/LSC70_HAND_CROPS/` (misma estructura de subcarpetas). Las
imágenes donde no se detecta mano se registran en
`resultados/preprocesamiento_fallidas.txt`. Sólo se ejecuta una vez (saltea los
crops ya generados). Usa `--force` para reprocesar todo.

### 5.2. Entrenar la Fase 1 (CNN multi-contexto)

Fase 1 puede entrenar **varios modelos independientes**, uno por contexto. Así
puedes tener un modelo de saludos y luego sumar números, letras, frases, etc.,
sin re-entrenar todo (ver `fase1_cnn/contextos.py`).

```bash
# Contexto por defecto: saludos (10 clases: HOLA, BUENAS, DIAS, ...)
python -m fase1_cnn.entrenar

# Otros contextos disponibles:
python -m fase1_cnn.entrenar --contexto numeros
python -m fase1_cnn.entrenar --contexto letras

# Overrides puntuales:
python -m fase1_cnn.entrenar --contexto saludos --epochs 40 --lr 0.0005
```

Cada contexto genera:
- `modelos_guardados/mejor_modelo_<contexto>.pth`
- `modelos_guardados/clases_<contexto>.json`
- `resultados/curvas_<contexto>.png`

> **¿Cómo agregar un contexto nuevo (p. ej. "frases" o "colores")?**
> 1. Edita `fase1_cnn/contextos.py` y añade una clave en `CONTEXTOS` con la
>    lista de clases (los nombres deben coincidir con carpetas del dataset).
> 2. Corre `python -m fase1_cnn.entrenar --contexto <nombre>`.
> 3. Listo: `predecir_voz.py` lo detecta automáticamente y puedes cambiar a él
>    con la tecla **c** en vivo.

### Fase 2 — LSTM sobre landmarks LSC-54
```bash
python -m fase2_lstm.entrenar
```

### Fase 3 — Fusión multimodal LSC50
```bash
python -m fase3_multimodal.entrenar
```

> Nota: `python -m <modulo>` asegura que Python resuelva los imports desde la
> raíz del proyecto. Alternativamente puedes invocar el archivo directo:
> `python fase1_cnn/entrenar.py`.

---

## 6. Evaluar los modelos

Genera matriz de confusión + reporte por clase + accuracy global:

```bash
# Fase 1 (especifica contexto):
python -m fase1_cnn.evaluar --contexto saludos
python -m fase1_cnn.evaluar --contexto numeros
python -m fase1_cnn.evaluar --contexto letras

# Otras fases:
python -m fase2_lstm.evaluar
python -m fase3_multimodal.evaluar
```

Para Fase 1 se generan:
- `resultados/fase1_<contexto>_reporte.txt` → precision/recall/f1 por clase
- `resultados/fase1_<contexto>_matriz_confusion.png`

En consola verás **accuracy global + top 3 clases con peor desempeño** — úsalo
para decidir si necesitas más datos o ajustes de esas clases específicas.

---

## 7. Reconocimiento en vivo por cámara (manos + rostro + voz + subtítulos)

Tras entrenar **al menos un contexto** de la Fase 1:

```bash
python -m fase1_cnn.predecir_voz
python -m fase1_cnn.predecir_voz --camera 1
python -m fase1_cnn.predecir_voz --contexto numeros   # arranca ya en ese contexto
```

Qué hace el script:
- Abre la cámara con OpenCV (`cv2.VideoCapture(CAMERA_INDEX)`).
- **Rostro** (MediaPipe FaceDetector, Tasks API) → bbox naranja, ancla visual.
- **Manos** (MediaPipe HandLandmarker, Tasks API) → hasta 2 manos con 21 puntos.
- Aplica **el mismo crop mano+antebrazo del preprocesamiento** y lo pasa al
  modelo CNN del contexto activo.
- Umbral de confianza y estabilidad (N frames consecutivos misma predicción).
- Sintetiza la seña con `pyttsx3` en un hilo daemon: **cada vez que se
  estabiliza una seña la pronuncia**, con un cooldown mínimo entre repeticiones
  idénticas para no saturar.
- **Auto-descubrimiento multi-contexto**: al arrancar escanea
  `modelos_guardados/mejor_modelo_*.pth` y lista todos los contextos
  disponibles. La tecla **c** cicla entre ellos en vivo.

Controles:
| Tecla | Acción |
|-------|--------|
| `q` | Salir |
| `c` | Ciclar al siguiente contexto cargado (saludos → números → letras → …) |

Parámetros ajustables (vía `.env` o directamente en `config.py`):
| Variable `.env` | Efecto | Default |
|-----------------|--------|---------|
| `UMBRAL_CONFIANZA` | Mínima probabilidad para aceptar la predicción | 0.70 |
| `FRAMES_ESTABLES` | Frames consecutivos antes de pronunciar | 10 |
| `COOLDOWN_TTS_SEG` | Segundos mínimos entre dos locuciones iguales | 1.5 |
| `TTS_RATE` | Velocidad de voz (palabras por minuto aprox.) | 160 |
| `CAMERA_INDEX` | Índice de la cámara | 0 |

---

## 8. Ajustes generales

- Toda la configuración (batch size, epochs, learning rate, rutas, dispositivo)
  vive en `config.py` y puede sobrescribirse vía `.env` (ver `.env.example`).
- Rutas de datasets: `LSC70_PATH`, `LSC54_PATH`, `LSC50_PATH`.
- Ruta de MediaPipe: `HAND_LANDMARKER_PATH`, `FACE_DETECTOR_PATH`.
- El proyecto detecta GPU automáticamente (`LSC_DEVICE` en `.env` puede forzar
  `cpu` o `cuda`).

---

## 8.1. Cómo empujar el modelo hacia 100%

El entrenamiento Fase 1 ya arranca con augmentación (flip, rotación,
color jitter) y Dropout(0.3). Si tras `evaluar` ves clases con accuracy bajo,
estas son las palancas en orden de impacto:

1. **Mira la matriz de confusión** (`resultados/fase1_<ctx>_matriz_confusion.png`).
   Las celdas fuera de la diagonal te dicen **qué se confunde con qué** — si
   HOLA se confunde con BUENAS, no necesitas "más datos en general", sino
   más variedad en esas dos clases concretas.
2. **Más epochs + learning-rate más bajo al final**: `--epochs 50 --lr 0.0005`.
   El scheduler (`ReduceLROnPlateau`) ya baja LR al estancarse val_loss.
3. **Más augmentación** para robustez a cámara real: añade
   `transforms.RandomAffine(translate=(0.1, 0.1))` en `entrenar.py`.
4. **Regularización**: sube Dropout a 0.4-0.5 en `fase1_cnn/modelo.py` si ves
   train_acc >> val_acc (overfitting).
5. **Más datos** para las clases débiles: graba tú mismo más videos y pásalos
   por el pre-procesador. Son los pocos shots que mueven la aguja real en vivo.
6. **Balance**: si tras evaluar hay clases con muchas menos muestras, considera
   `WeightedRandomSampler` o `class_weight` en `CrossEntropyLoss`.

> **Realidad**: 100% en val normalmente significa overfitting a la persona/fondo
> del dataset. El objetivo real es **alta precisión en cámara, con tu rostro y
> tu iluminación**. Prioriza datos propios sobre epochs extra.

---

## 8.2. Varios modelos + unificación (multi-contexto)

¿Se puede entrenar un modelo por dominio e ir sumando con el tiempo? **Sí.** El
proyecto ya lo soporta:

- Cada **contexto** (saludos, números, letras, frases, …) tiene su propio
  modelo CNN entrenado con sólo sus clases. Menos clases por modelo = mayor
  accuracy por clase.
- No hay re-entrenamiento global al añadir un contexto nuevo — solo agregas
  una entrada en `fase1_cnn/contextos.py` y corres `entrenar --contexto X`.
- `predecir_voz.py` **unifica** los modelos en inferencia: los carga todos y
  permite cambiar con la tecla **c**. Puedes empezar en uno concreto con
  `--contexto numeros`.
- Futuro: un "router" automático que escoja el contexto según el gesto inicial
  (por ejemplo, manos cerca de la cabeza → saludos, dedos rectos → números).
  Para eso basta añadir otro modelo pequeño tipo `contexto_router.pth` que
  clasifique entre los 3-4 contextos posibles, y luego delegue al experto.

---

## 9. Solución de problemas comunes

| Problema | Causa | Solución |
|----------|-------|----------|
| `ModuleNotFoundError: mediapipe` | Python 3.13 o venv sin instalar | Usar Python 3.11 + `pip install -r requirements.txt` |
| `Could not find a version that satisfies mediapipe` | Python incompatible | Usar Python 3.9–3.12 |
| `cv2` no tiene atributos | Conflicto `opencv-python` vs `opencv-contrib-python` | Desinstalar ambos, reinstalar solo `opencv-python` |
| Cámara no abre (`cap.isOpened()` = False) | Otra app usa la cámara o falta permiso | Cerrar otras apps; en Windows revisar permisos de cámara |
| TTS silencioso en Windows | Drivers SAPI | Probar `gTTS` (online) como alternativa |
| `ERROR: No se encontró hand_landmarker.task` o `face_detector.task` | Archivo no descargado | Ver sección 4 |
| `ERROR: No se encontró clases_fase1.json` | No se ha entrenado aún | Ejecutar `python -m fase1_cnn.entrenar` |
