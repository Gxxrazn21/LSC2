"""
FASE 1 - RECONOCIMIENTO EN VIVO (MANOS + ROSTRO + VOZ)

Pipeline:
  1. Camara (OpenCV)
  2. Deteccion de rostro (MediaPipe FaceDetector)
  3. Deteccion de manos (MediaPipe HandLandmarker)
  4. CNN clasifica el crop de mano
  5. TTS dice la palabra detectada 3 veces (hilo aparte, no bloquea)

Controles:
  q -> salir
"""
import json
import os
import queue
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from torchvision import transforms

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from config import (
    FASE1, DEVICE, HAND_LANDMARKER_PATH, FACE_DETECTOR_PATH, MODELOS_DIR,
    CAMERA_INDEX, UMBRAL_CONFIANZA, FRAMES_ESTABLES, TTS_RATE,
)
from fase1_cnn.modelo import crear_modelo
from fase1_cnn.contextos import CONTEXTO, rutas_de, texto_para_voz
from utils.crop_mano import recortar_crop

COLOR_MANO   = (0, 255, 0)
COLOR_ROSTRO = (255, 180, 0)
COLOR_TEXTO  = (255, 255, 255)
COLOR_FONDO  = (0, 0, 0)
COLOR_OK     = (60, 220, 60)
COLOR_WARN   = (0, 165, 255)


# ─────────────────────────────────────────────
# TTS NO BLOQUEANTE
# ─────────────────────────────────────────────
_tts_queue: "queue.Queue" = queue.Queue()


def _tts_worker(rate: int):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
    except Exception as e:
        print(f"  [TTS] No se pudo iniciar pyttsx3: {e}")
        return
    while True:
        msg = _tts_queue.get()
        if msg is None:
            _tts_queue.task_done()
            break
        try:
            engine.say(msg)
            engine.runAndWait()
        except Exception as e:
            print(f"  [TTS] Error al sintetizar '{msg}': {e}")
        finally:
            _tts_queue.task_done()


def hablar_async(texto: str):
    _tts_queue.put(texto)


# ─────────────────────────────────────────────
# CARGA DE MODELO
# ─────────────────────────────────────────────
def cargar_modelo(img_size: int):
    r = rutas_de(CONTEXTO)
    if not os.path.exists(r["modelo"]) or not os.path.exists(r["clases"]):
        print(f"  ERROR: Modelo no encontrado en {MODELOS_DIR}")
        print(f"         Ejecuta: python -m fase1_cnn.entrenar")
        return None, None
    with open(r["clases"], "r", encoding="utf-8") as f:
        clases = json.load(f)
    modelo = crear_modelo(len(clases), img_size, DEVICE)
    modelo.load_state_dict(torch.load(r["modelo"], map_location=DEVICE, weights_only=True))
    modelo.eval()
    return modelo, clases


# ─────────────────────────────────────────────
# UTILIDADES DE DIBUJO
# ─────────────────────────────────────────────
def dibujar_landmarks_mano(frame, hand_landmarks, w, h):
    CONEX = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in CONEX:
        cv2.line(frame, pts[a], pts[b], COLOR_MANO, 2)
    for (px, py) in pts:
        cv2.circle(frame, (px, py), 3, COLOR_MANO, -1)


def dibujar_panel_superior(frame, fps, hand_ok, face_ok, n_clases):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), COLOR_FONDO, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    color_mano = COLOR_OK if hand_ok else COLOR_WARN
    color_cara = COLOR_OK if face_ok else COLOR_WARN
    cv2.putText(frame, f"Mano: {'OK' if hand_ok else '--'}",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_mano, 2)
    cv2.putText(frame, f"Rostro: {'OK' if face_ok else '--'}",
                (140, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_cara, 2)
    cv2.putText(frame, f"Clases: {n_clases}",
                (280, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXTO, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (w - 110, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXTO, 2)


def dibujar_subtitulo(frame, label_text):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 70), (w, h), COLOR_FONDO, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, label_text,
                (15, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXTO, 2)
    cv2.putText(frame, "[q] salir",
                (15, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def predecir_tiempo_real(camera_index: int = None):
    camera_index = CAMERA_INDEX if camera_index is None else camera_index

    threading.Thread(target=_tts_worker, kwargs={"rate": TTS_RATE}, daemon=True).start()

    modelo, clases = cargar_modelo(FASE1["img_size"])
    if modelo is None:
        return
    print(f"  Modelo cargado: {len(clases)} clases — {clases}")

    if not os.path.exists(HAND_LANDMARKER_PATH):
        print(f"  ERROR: Falta {HAND_LANDMARKER_PATH}")
        return
    hand_opts = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_opts)

    if not os.path.exists(FACE_DETECTOR_PATH):
        print(f"  ERROR: Falta {FACE_DETECTOR_PATH}")
        hand_detector.close()
        return
    face_opts = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=FACE_DETECTOR_PATH),
        min_detection_confidence=0.5,
    )
    face_detector = vision.FaceDetector.create_from_options(face_opts)

    transform = transforms.Compose([
        transforms.Resize((FASE1["img_size"], FASE1["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"  ERROR: No se pudo abrir camara (indice {camera_index})")
        face_detector.close()
        hand_detector.close()
        return

    print(f"\n  Reconocimiento en vivo activo.")
    print(f"  Umbral: {UMBRAL_CONFIANZA*100:.0f}%  |  Estabilidad: {FRAMES_ESTABLES} frames")
    print("  Tecla: [q] salir\n")

    last_pred = ""
    pred_count = 0
    t_prev = time.time()
    fps_smooth = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            face_result = face_detector.detect(mp_image)
            face_ok = bool(face_result.detections)
            for det in face_result.detections:
                bb = det.bounding_box
                x1 = int(max(0, bb.origin_x))
                y1 = int(max(0, bb.origin_y))
                x2 = int(min(w, bb.origin_x + bb.width))
                y2 = int(min(h, bb.origin_y + bb.height))
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ROSTRO, 2)

            hand_result = hand_detector.detect(mp_image)
            hand_ok = bool(hand_result.hand_landmarks)
            label_text = "Esperando sena..." if face_ok else "Acercate a la camara"

            if hand_ok:
                frames_sin_mano = 0
                crop_bgr = recortar_crop(frame, hand_result.hand_landmarks,
                                         out_size=FASE1["img_size"])
                for hlm in hand_result.hand_landmarks:
                    dibujar_landmarks_mano(frame, hlm, w, h)

                if crop_bgr is not None:
                    rgb_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb_crop)
                    x = transform(pil).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        out = modelo(x)
                        probs = torch.softmax(out, dim=1)
                        prob, pred = torch.max(probs, 1)
                    current_prob = float(prob.item())
                    idx = int(pred.item())

                    if current_prob >= UMBRAL_CONFIANZA:
                        current_pred = clases[idx]
                        label_text = f"{current_pred}  ({current_prob*100:.0f}%)"

                        if current_pred == last_pred:
                            pred_count += 1
                        else:
                            last_pred = current_pred
                            pred_count = 1

                        if pred_count >= FRAMES_ESTABLES:
                            texto_voz = texto_para_voz(current_pred)
                            print(f"  >>> {current_pred}  ({current_prob*100:.1f}%)")
                            hablar_async(texto_voz)
                            pred_count = 0
                    else:
                        label_text = f"Analizando... ({current_prob*100:.0f}%)"
                        pred_count = 0
                        last_pred = ""
            else:
                if face_ok:
                    label_text = "Muestra una mano"
                pred_count = 0
                last_pred = ""

            t_now = time.time()
            dt = max(t_now - t_prev, 1e-6)
            t_prev = t_now
            fps_inst = 1.0 / dt
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps_inst if fps_smooth > 0 else fps_inst

            dibujar_panel_superior(frame, fps_smooth, hand_ok, face_ok, len(clases))
            dibujar_subtitulo(frame, label_text)
            cv2.imshow("LSC - Reconocimiento Fase 1", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try: face_detector.close()
        except Exception: pass
        try: hand_detector.close()
        except Exception: pass
        _tts_queue.put(None)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Reconocimiento LSC en vivo")
    p.add_argument("--camera", type=int, default=None, help="Indice de camara")
    args = p.parse_args()
    predecir_tiempo_real(camera_index=args.camera)
