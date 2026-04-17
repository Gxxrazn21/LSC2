# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sistema de Reconocimiento de Lengua de Señas Colombiana (LSC) — Phase 1. Detects greeting signs (HOLA, BUENAS, DIAS, NOCHES, TARDES) in real time via camera and pronounces them aloud using TTS.

## Environment Setup

```bash
# Activate virtual environment (Windows)
source venv_lsc/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

Validated on Python 3.13 | torch 2.11 | mediapipe 0.10.33 | opencv 4.13. Do **not** install `opencv-contrib-python` alongside `opencv-python`.

## Common Commands

```bash
# Train the model (default: saludos context)
python -m fase1_cnn.entrenar

# Train with custom hyperparameters
python -m fase1_cnn.entrenar --epochs 40 --batch-size 16 --lr 0.0005

# Evaluate with confusion matrix
python -m fase1_cnn.evaluar

# Run real-time recognition + TTS
python -m fase1_cnn.predecir_voz

# Run with a specific camera index
python -m fase1_cnn.predecir_voz --camera 1

# Predict on a single static image
python -m fase1_cnn.predecir <ruta_imagen>

# Print active configuration
python config.py
```

## Architecture

All hyperparameters and paths live in `config.py`, which loads overrides from `.env` (see `.env.example`). Key env vars: `LSC70_PATH`, `CAMERA_INDEX`, `UMBRAL_CONFIANZA`, `FRAMES_ESTABLES`, `LSC_DEVICE`.

**Real-time pipeline** (`fase1_cnn/predecir_voz.py`):
1. OpenCV captures and flips each frame
2. MediaPipe `FaceDetector` checks for a face in scene
3. MediaPipe `HandLandmarker` extracts 21 hand landmarks
4. `utils/crop_mano.recortar_crop()` computes a padded bounding box → square crop
5. `LSC_CNN` (PyTorch) classifies the crop via softmax
6. Prediction is accepted when `prob ≥ UMBRAL_CONFIANZA` (default 70%) for `FRAMES_ESTABLES` (default 10) consecutive frames
7. `hablar_async()` enqueues the word to a daemon thread running `pyttsx3` (non-blocking)

**CNN architecture** (`fase1_cnn/modelo.py`): 3 conv blocks (Conv2d + BatchNorm + ReLU + MaxPool) → `AdaptiveAvgPool2d(4)` → `Linear(4096→256)` → `Linear(256→n_classes)`. ~1.1M parameters regardless of `img_size`.

**Training** (`fase1_cnn/entrenar.py`): Adam + `ReduceLROnPlateau`, early stopping after 12 epochs without val_acc improvement. Train augmentations: horizontal flip, rotation ±15°, affine translate/scale, color jitter.

## Key Files

| File | Role |
|---|---|
| `config.py` | Central config; all paths and hyperparameters |
| `fase1_cnn/contextos.py` | Defines class list (`CLASES`) and TTS translations |
| `fase1_cnn/modelo.py` | `LSC_CNN` architecture + `crear_modelo()` factory |
| `fase1_cnn/dataset.py` | `LSCDataset` — loads images filtered by `target_gestures` |
| `fase1_cnn/entrenar.py` | Full training loop with early stopping |
| `fase1_cnn/evaluar.py` | Evaluation with confusion matrix |
| `fase1_cnn/predecir.py` | Static image inference |
| `fase1_cnn/predecir_voz.py` | Real-time camera + TTS pipeline |
| `utils/crop_mano.py` | `recortar_crop()` — hand ROI extraction |
| `utils/metricas.py` | `calcular_accuracy`, `reporte_clasificacion`, `evaluar_modelo` |
| `utils/visualizacion.py` | `graficar_curvas()` — loss/accuracy plots |

## Required Assets

- `hand_landmarker.task` — MediaPipe HandLandmarker model (root dir)
- `face_detector.task` — MediaPipe FaceDetector model (root dir)
- `datasets/LSC70_HAND_CROPS/LSC70/LSC70/` — dataset folders named by class (HOLA, BUENAS, etc.)

After training, `modelos_guardados/` will contain `mejor_modelo_saludos.pth` and `clases_saludos.json`. Results (reports, confusion matrix, training curves) go to `resultados/`.

## Extending to New Sign Contexts

To add a new context beyond `saludos`, update `fase1_cnn/contextos.py`: add the class list to `CLASES`, entries to `TRADUCCIONES_TTS`, and register the context name in `listar_contextos()`.
