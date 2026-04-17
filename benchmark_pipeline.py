"""
Benchmark de latencia del pipeline completo de reconocimiento LSC.

Mide por separado:
  1. Captura de frame (camara o frame sintetico)
  2. Deteccion de rostro (MediaPipe FaceDetector)
  3. Deteccion de mano (MediaPipe HandLandmarker)
  4. Extraccion de crop (recortar_crop)
  5. Inferencia CNN (MobileNetV3-Small)
  6. Latencia total por frame
  7. FPS estimado

Uso:
  python benchmark_pipeline.py              # sin camara (usa imagen del dataset)
  python benchmark_pipeline.py --camera 0   # con camara real
  python benchmark_pipeline.py --frames 100 # numero de frames a promediar
"""

import argparse
import json
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    FASE1, DEVICE, HAND_LANDMARKER_PATH, FACE_DETECTOR_PATH, MODELOS_DIR,
)
from fase1_cnn.modelo import crear_modelo
from fase1_cnn.contextos import CONTEXTO, rutas_de
from utils.crop_mano import recortar_crop

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

BAR = "=" * 62


def ms(t: float) -> str:
    return f"{t * 1000:.2f} ms"


def stats(times: list) -> dict:
    arr = np.array(times)
    return {
        "media":  float(arr.mean()),
        "mediana": float(np.median(arr)),
        "p95":    float(np.percentile(arr, 95)),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
    }


def imprimir_fila(nombre, s: dict, unidad="ms"):
    factor = 1000 if unidad == "ms" else 1
    print(
        f"  {nombre:<25s}  "
        f"media={s['media']*factor:7.2f}ms  "
        f"mediana={s['mediana']*factor:7.2f}ms  "
        f"p95={s['p95']*factor:7.2f}ms  "
        f"[{s['min']*factor:.1f}-{s['max']*factor:.1f}]"
    )


def cargar_modelo_inferencia():
    r = rutas_de(CONTEXTO)
    if not os.path.exists(r["modelo"]) or not os.path.exists(r["clases"]):
        print("  ERROR: Modelo no encontrado. Ejecuta primero: python -m fase1_cnn.entrenar")
        sys.exit(1)
    with open(r["clases"], "r", encoding="utf-8") as f:
        clases = json.load(f)
    model = crear_modelo(len(clases), FASE1["img_size"], DEVICE, freeze_backbone=False)
    model.load_state_dict(torch.load(r["modelo"], map_location=DEVICE, weights_only=True))
    model.eval()
    return model, clases


def obtener_frame_base() -> np.ndarray:
    """Frame de referencia: imagen real del dataset si existe, o sintetica."""
    ruta = os.path.join(
        "datasets", "LSC70_HAND_CROPS", "LSC70", "LSC70",
        "LSC70W", "Per01", "HOLA", "Per01_HOLA_0.jpg"
    )
    if os.path.exists(ruta):
        img = cv2.imread(ruta)
        # Escalar a resolucion tipica de webcam
        frame = cv2.resize(img, (640, 480))
        print(f"  Frame base: imagen real del dataset ({ruta})")
        return frame
    else:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("  Frame base: frame sintetico (dataset no disponible en ruta relativa)")
        return frame


def benchmark(n_frames: int = 60, camera_index: int = None):
    print(f"\n{BAR}")
    print("  BENCHMARK PIPELINE LSC — Fase 1")
    print(BAR)
    print(f"  Dispositivo: {DEVICE}  |  Frames: {n_frames}  |  img_size: {FASE1['img_size']}")

    # ── Cargar modelo CNN ──────────────────────────────────────────
    print("\n  Cargando modelo CNN...")
    t0 = time.perf_counter()
    model, clases = cargar_modelo_inferencia()
    t_carga = time.perf_counter() - t0
    print(f"  Modelo cargado en {ms(t_carga)}  ({len(clases)} clases)")

    # Warmup CNN (primera inferencia es mas lenta por JIT/cache)
    dummy = torch.randn(1, 3, FASE1["img_size"], FASE1["img_size"]).to(DEVICE)
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
    print("  Warmup CNN: OK (5 pasadas)")

    transform = transforms.Compose([
        transforms.Resize((FASE1["img_size"], FASE1["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    # ── MediaPipe ─────────────────────────────────────────────────
    if not os.path.exists(HAND_LANDMARKER_PATH):
        print(f"  ERROR: Falta {HAND_LANDMARKER_PATH}")
        sys.exit(1)

    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_opts)

    face_detector = None
    if os.path.exists(FACE_DETECTOR_PATH):
        face_opts = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_DETECTOR_PATH),
            min_detection_confidence=0.5,
        )
        face_detector = vision.FaceDetector.create_from_options(face_opts)
        print("  FaceDetector: OK")
    else:
        print(f"  FaceDetector: NO disponible ({FACE_DETECTOR_PATH})")

    # ── Camara o frame sintetico ───────────────────────────────────
    cap = None
    frame_base = None
    if camera_index is not None:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"  Camara: indice {camera_index} OK")
        else:
            print(f"  Camara: indice {camera_index} NO disponible — usando frame sintetico")
            cap = None
    if cap is None:
        frame_base = obtener_frame_base()

    # ── Loop de medicion ──────────────────────────────────────────
    print(f"\n  Midiendo {n_frames} frames...\n")

    t_captura   = []
    t_face      = []
    t_hand      = []
    t_crop      = []
    t_cnn       = []
    t_total     = []
    n_manos     = 0
    n_rostros   = 0

    for i in range(n_frames):
        t_frame_start = time.perf_counter()

        # 1. Captura
        t0 = time.perf_counter()
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
        else:
            frame = frame_base.copy()
            # Simular variacion minima de camara real
            noise = np.random.randint(-8, 8, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        t_captura.append(time.perf_counter() - t0)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 2. Deteccion de rostro
        if face_detector is not None:
            t0 = time.perf_counter()
            face_res = face_detector.detect(mp_img)
            t_face.append(time.perf_counter() - t0)
            if face_res.detections:
                n_rostros += 1

        # 3. Deteccion de mano
        t0 = time.perf_counter()
        hand_res = hand_detector.detect(mp_img)
        t_hand.append(time.perf_counter() - t0)
        hand_ok = bool(hand_res.hand_landmarks)
        if hand_ok:
            n_manos += 1

        # 4. Crop (solo si hay mano detectada)
        crop_bgr = None
        if hand_ok:
            t0 = time.perf_counter()
            crop_bgr = recortar_crop(frame, hand_res.hand_landmarks,
                                     out_size=FASE1["img_size"])
            t_crop.append(time.perf_counter() - t0)
        else:
            t_crop.append(0.0)

        # 5. Inferencia CNN (siempre — usa crop real o dummy)
        t0 = time.perf_counter()
        if crop_bgr is not None:
            rgb_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb_crop)
            x_t = transform(pil).unsqueeze(0).to(DEVICE)
        else:
            x_t = dummy
        with torch.no_grad():
            out   = model(x_t)
            probs = torch.softmax(out, dim=1)
        t_cnn.append(time.perf_counter() - t0)

        t_total.append(time.perf_counter() - t_frame_start)

        if (i + 1) % 20 == 0 or i == 0:
            fps_inst = 1.0 / max(t_total[-1], 1e-6)
            print(f"  Frame {i+1:4d}/{n_frames}  "
                  f"total={ms(t_total[-1])}  FPS inst.={fps_inst:.1f}  "
                  f"manos={'Si' if hand_ok else 'No'}")

    if cap:
        cap.release()
    try: hand_detector.close()
    except Exception: pass
    if face_detector:
        try: face_detector.close()
        except Exception: pass

    # ── Resultados ────────────────────────────────────────────────
    n = len(t_total)
    print(f"\n{BAR}")
    print(f"  RESULTADOS ({n} frames procesados)")
    print(BAR)

    imprimir_fila("Captura frame", stats(t_captura))
    if t_face:
        imprimir_fila("FaceDetector (MediaPipe)", stats(t_face))
    imprimir_fila("HandLandmarker (MediaPipe)", stats(t_hand))
    imprimir_fila("Crop mano+antebrazo", stats(t_crop))
    imprimir_fila("Inferencia CNN", stats(t_cnn))
    print("  " + "-" * 58)
    imprimir_fila("TOTAL por frame", stats(t_total))

    fps_med = 1.0 / max(stats(t_total)["media"], 1e-6)
    fps_p95 = 1.0 / max(stats(t_total)["p95"],   1e-6)
    print(f"\n  FPS estimado (1/media):   {fps_med:.1f}")
    print(f"  FPS garantizado (1/p95):  {fps_p95:.1f}")
    print(f"\n  Manos detectadas:   {n_manos}/{n} frames ({100*n_manos/n:.0f}%)")
    if t_face:
        print(f"  Rostros detectados: {n_rostros}/{n} frames ({100*n_rostros/n:.0f}%)")

    # Cuello de botella
    etapas = {
        "Captura":       stats(t_captura)["media"],
        "FaceDetector":  stats(t_face)["media"] if t_face else 0,
        "HandLandmarker": stats(t_hand)["media"],
        "Crop":          stats(t_crop)["media"],
        "CNN":           stats(t_cnn)["media"],
    }
    cuello = max(etapas, key=etapas.get)
    print(f"\n  Cuello de botella:  {cuello} ({etapas[cuello]*1000:.1f} ms/frame)")

    if fps_med >= 25:
        nivel = "EXCELENTE (tiempo real fluido)"
    elif fps_med >= 15:
        nivel = "ACEPTABLE (leve latencia visible)"
    else:
        nivel = "LENTO — considerar GPU o reducir img_size"
    print(f"  Rendimiento:        {nivel}")

    print(f"\n{BAR}\n")

    return {
        "fps_media": fps_med,
        "fps_p95":   fps_p95,
        "etapas_ms": {k: v * 1000 for k, v in etapas.items()},
        "cuello":    cuello,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Benchmark pipeline LSC Fase 1")
    p.add_argument("--frames",  type=int, default=60,  help="Frames a medir (default: 60)")
    p.add_argument("--camera",  type=int, default=None, help="Indice de camara (omitir = sintetico)")
    args = p.parse_args()
    benchmark(n_frames=args.frames, camera_index=args.camera)
