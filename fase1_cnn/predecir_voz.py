"""
FASE 1 - RECONOCIMIENTO EN VIVO (MANOS + ROSTRO + VOZ)

Pipeline:
  1. Camara (OpenCV)
  2. Deteccion de rostro (MediaPipe FaceDetector) -- solo informativo
  3. Deteccion de manos (MediaPipe HandLandmarker)
  4. CNN clasifica el crop mano+antebrazo
  5. SignDetector: ventana deslizante con voto ponderado
  6. TTS habla cada sena detectada (hilo aparte, no bloquea)
  7. PhraseBuffer: en --modo-frase acumula palabras y habla la frase completa

Controles:
  q -> salir
  c -> limpiar buffer de frase (solo en modo frase)
"""
import json
import os
import queue
import sys
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from torchvision import transforms

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from config import (
    FASE1, DEVICE, HAND_LANDMARKER_PATH, FACE_DETECTOR_PATH, MODELOS_DIR,
    CAMERA_INDEX, UMBRAL_CONFIANZA, COOLDOWN_TTS_SEG, TTS_RATE,
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
COLOR_FRASE  = (255, 220, 50)

# ──────────────────────────────────────────────────────────────────
# TTS NO BLOQUEANTE — win32com SAPI5 (nativo Windows, sin conflictos COM)
# Fallback: pyttsx3 re-init por utterance si win32com no esta disponible
# ──────────────────────────────────────────────────────────────────
_tts_queue: "queue.Queue" = queue.Queue()
_USE_WIN32COM = False
try:
    import win32com.client   # pywin32
    import pythoncom
    _USE_WIN32COM = True
except ImportError:
    try:
        import pyttsx3
    except ImportError:
        pass


def _tts_worker_win32(rate: int):
    """Worker usando SAPI.SpVoice via win32com — mas estable en Windows."""
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    try:
        spk = win32com.client.Dispatch("SAPI.SpVoice")
        # SAPI Rate va de -10 (lento) a 10 (rapido); pyttsx3 usa ~160 wpm
        spk.Rate = max(-5, min(5, (rate - 160) // 20))
        while True:
            msg = _tts_queue.get()
            if msg is None:
                _tts_queue.task_done()
                break
            try:
                spk.Speak(msg)
            except Exception as e:
                print(f"  [TTS] Error win32com '{msg}': {e}")
            finally:
                _tts_queue.task_done()
    finally:
        pythoncom.CoUninitialize()


def _tts_worker_pyttsx3(rate: int):
    """Worker con pyttsx3 re-init por utterance — evita estado corrupto del engine."""
    while True:
        msg = _tts_queue.get()
        if msg is None:
            _tts_queue.task_done()
            break
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.say(msg)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"  [TTS] Error pyttsx3 '{msg}': {e}")
        finally:
            _tts_queue.task_done()


def _tts_worker(rate: int):
    if _USE_WIN32COM:
        print("  [TTS] Motor: SAPI5 via win32com")
        _tts_worker_win32(rate)
    else:
        print("  [TTS] Motor: pyttsx3 (re-init por utterance)")
        _tts_worker_pyttsx3(rate)


def hablar_async(texto: str):
    _tts_queue.put(texto)


# ──────────────────────────────────────────────────────────────────
# SIGN DETECTOR: ventana deslizante con voto ponderado
# ──────────────────────────────────────────────────────────────────
class SignDetector:
    """
    Reemplaza el contador de frames consecutivos.

    Mantiene una deque de los ultimos `window` frames. Cada frame
    aporta un voto pesado por su confianza y su posicion reciente.
    Emite una sena cuando la clase dominante supera `vote_threshold`
    con al menos `min_votes` frames validos en la ventana.
    """

    def __init__(self, window: int = 15, min_votes: int = 8,
                 vote_threshold: float = 0.60):
        self.window        = deque(maxlen=window)
        self.min_votes     = min_votes
        self.vote_threshold = vote_threshold

    def push(self, idx: int, confidence: float):
        """Agrega un frame con prediccion valida (confidence >= umbral)."""
        self.window.append((idx, confidence))

    def push_invalid(self):
        """Frame sin prediccion valida (mano ausente o confianza baja)."""
        self.window.append(None)

    def consensus(self) -> tuple:
        """
        Devuelve (class_idx, score) si hay consenso, o (None, 0.0).
        Score = fraccion del peso total que recibe la clase ganadora.
        """
        valid = [(i, c) for i, entry in enumerate(self.window)
                 if entry is not None for _, c in [entry]]

        # Reconstruir correctamente
        entries = [(i, e[0], e[1]) for i, e in enumerate(self.window) if e is not None]

        if len(entries) < self.min_votes:
            return None, 0.0

        votes: dict = {}
        total_w = 0.0
        n = len(self.window)
        for pos, cls_idx, conf in entries:
            weight = ((pos + 1) / n) * conf   # reciente + alta confianza = mas peso
            votes[cls_idx] = votes.get(cls_idx, 0.0) + weight
            total_w += weight

        if total_w == 0:
            return None, 0.0

        best = max(votes, key=votes.get)
        score = votes[best] / total_w
        return (best, score) if score >= self.vote_threshold else (None, 0.0)

    def clear(self):
        self.window.clear()

    @property
    def top_candidate(self) -> tuple:
        """Clase e confianza promedio del frame mas reciente valido (para UI)."""
        for entry in reversed(self.window):
            if entry is not None:
                return entry
        return (None, 0.0)


# ──────────────────────────────────────────────────────────────────
# PHRASE BUFFER: acumula senas para hablar frases completas
# ──────────────────────────────────────────────────────────────────
class PhraseBuffer:
    """
    Acumula senas detectadas y las une como frase cuando hay
    una pausa (gap_seconds sin nueva sena).
    Solo activo en --modo-frase.
    """

    def __init__(self, gap_seconds: float = 2.0):
        self.gap_seconds = gap_seconds
        self._signs: list = []          # lista de strings
        self._last_time: float = 0.0

    def add(self, sign_name: str) -> bool:
        """
        Agrega una sena. Devuelve True si es una sena nueva
        (evita duplicados consecutivos).
        """
        if self._signs and self._signs[-1] == sign_name:
            return False
        self._signs.append(sign_name)
        self._last_time = time.time()
        return True

    def should_flush(self) -> bool:
        return bool(self._signs) and (time.time() - self._last_time) >= self.gap_seconds

    def flush(self) -> list:
        phrase = list(self._signs)
        self._signs = []
        self._last_time = 0.0
        return phrase

    def clear(self):
        self._signs = []
        self._last_time = 0.0

    @property
    def content(self) -> list:
        return list(self._signs)


# ──────────────────────────────────────────────────────────────────
# CARGA DE MODELO
# ──────────────────────────────────────────────────────────────────
def cargar_modelo(img_size: int):
    r = rutas_de(CONTEXTO)
    if not os.path.exists(r["modelo"]) or not os.path.exists(r["clases"]):
        print(f"  ERROR: Modelo no encontrado en {MODELOS_DIR}")
        print(f"         Ejecuta: python -m fase1_cnn.entrenar")
        return None, None
    with open(r["clases"], "r", encoding="utf-8") as f:
        clases = json.load(f)
    # freeze_backbone=False en inferencia: todos los pesos se cargan correctamente
    modelo = crear_modelo(len(clases), img_size, DEVICE, freeze_backbone=False)
    modelo.load_state_dict(torch.load(r["modelo"], map_location=DEVICE, weights_only=True))
    modelo.eval()
    return modelo, clases


# ──────────────────────────────────────────────────────────────────
# UTILIDADES DE DIBUJO
# ──────────────────────────────────────────────────────────────────
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


def dibujar_panel_superior(frame, fps, hand_ok, face_ok, n_clases, modo_frase):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), COLOR_FONDO, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    color_mano = COLOR_OK if hand_ok else COLOR_WARN
    color_cara = COLOR_OK if face_ok else COLOR_WARN
    cv2.putText(frame, f"Mano: {'OK' if hand_ok else '--'}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_mano, 2)
    cv2.putText(frame, f"Rostro: {'OK' if face_ok else '--'}",
                (140, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_cara, 2)
    modo_str = "FRASE" if modo_frase else "PALABRA"
    cv2.putText(frame, f"Modo: {modo_str}",
                (280, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_FRASE if modo_frase else COLOR_TEXTO, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXTO, 2)


def dibujar_barra_confianza(frame, clase, prob, clases):
    """Mini barra de confianza debajo del panel superior."""
    h, w = frame.shape[:2]
    if clase is None:
        return
    barra_w = int((w - 20) * prob)
    cv2.rectangle(frame, (10, 48), (10 + barra_w, 58), COLOR_OK, -1)
    cv2.rectangle(frame, (10, 48), (w - 10, 58), COLOR_TEXTO, 1)
    cv2.putText(frame, f"{clase} {prob*100:.0f}%",
                (12, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_FONDO, 1)


def dibujar_buffer_frase(frame, phrase_words):
    """Muestra la frase acumulada encima del subtitulo."""
    if not phrase_words:
        return
    h, w = frame.shape[:2]
    texto = " | ".join(phrase_words)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 110), (w, h - 75), COLOR_FONDO, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"Frase: {texto}",
                (10, h - 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_FRASE, 2)


def dibujar_subtitulo(frame, label_text):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 72), (w, h), COLOR_FONDO, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, label_text,
                (15, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXTO, 2)
    cv2.putText(frame, "[q] salir  [c] limpiar frase",
                (15, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)


# ──────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────────
def predecir_tiempo_real(camera_index: int = None, modo_frase: bool = False,
                         phrase_gap: float = 2.0):
    camera_index = CAMERA_INDEX if camera_index is None else camera_index

    threading.Thread(target=_tts_worker, kwargs={"rate": TTS_RATE}, daemon=True).start()

    modelo, clases = cargar_modelo(FASE1["img_size"])
    if modelo is None:
        return
    print(f"  Modelo cargado: {len(clases)} clases - {clases}")

    if not os.path.exists(HAND_LANDMARKER_PATH):
        print(f"  ERROR: Falta {HAND_LANDMARKER_PATH}")
        return
    hand_opts = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_opts)

    face_detector = None
    if os.path.exists(FACE_DETECTOR_PATH):
        face_opts = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=FACE_DETECTOR_PATH),
            min_detection_confidence=0.5,
        )
        face_detector = vision.FaceDetector.create_from_options(face_opts)

    transform = transforms.Compose([
        transforms.Resize((FASE1["img_size"], FASE1["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"  ERROR: No se pudo abrir camara (indice {camera_index})")
        hand_detector.close()
        return

    # ── Estado de deteccion ────────────────────────────────────────
    detector     = SignDetector(window=15, min_votes=8, vote_threshold=0.60)
    cooldowns: dict = {}          # {clase_str: timestamp_ultimo_hablar}
    phrase_buf   = PhraseBuffer(gap_seconds=phrase_gap)

    print(f"\n  Reconocimiento en vivo activo.")
    print(f"  Umbral CNN: {UMBRAL_CONFIANZA*100:.0f}%  |  "
          f"Ventana: 15 frames  |  Cooldown/sena: {COOLDOWN_TTS_SEG}s")
    if modo_frase:
        print(f"  MODO FRASE activo: gap={phrase_gap}s para hablar frase acumulada")
    print("  Teclas: [q] salir  [c] limpiar frase\n")

    t_prev   = time.time()
    fps_smooth = 0.0
    label_text = "Preparando..."

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # ── Deteccion de rostro (solo visual, no bloquea) ──────
            face_ok = False
            if face_detector is not None:
                face_result = face_detector.detect(mp_image)
                face_ok = bool(face_result.detections)
                for det in face_result.detections:
                    bb = det.bounding_box
                    x1 = int(max(0, bb.origin_x))
                    y1 = int(max(0, bb.origin_y))
                    x2 = int(min(w, bb.origin_x + bb.width))
                    y2 = int(min(h, bb.origin_y + bb.height))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ROSTRO, 2)

            # ── Deteccion de manos ────────────────────────────────
            hand_result  = hand_detector.detect(mp_image)
            hand_ok      = bool(hand_result.hand_landmarks)
            ui_clase_top = None
            ui_prob_top  = 0.0

            if hand_ok:
                crop_bgr = recortar_crop(frame, hand_result.hand_landmarks,
                                         out_size=FASE1["img_size"])
                for hlm in hand_result.hand_landmarks:
                    dibujar_landmarks_mano(frame, hlm, w, h)

                if crop_bgr is not None:
                    rgb_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb_crop)
                    x_t = transform(pil).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        out   = modelo(x_t)
                        probs = torch.softmax(out, dim=1)
                        prob, pred = torch.max(probs, 1)
                    current_prob = float(prob.item())
                    idx          = int(pred.item())

                    if current_prob >= UMBRAL_CONFIANZA:
                        detector.push(idx, current_prob)
                        ui_clase_top = clases[idx]
                        ui_prob_top  = current_prob
                        label_text   = f"{ui_clase_top}  ({current_prob*100:.0f}%)"
                    else:
                        detector.push_invalid()
                        label_text = f"Analizando... ({current_prob*100:.0f}%)"
                        ci, cp = detector.top_candidate
                        if ci is not None:
                            ui_clase_top = clases[ci]
                            ui_prob_top  = cp
                else:
                    detector.push_invalid()
                    label_text = "Crop invalido"
            else:
                detector.push_invalid()
                label_text = "Muestra la mano" if face_ok else "Acercate a la camara"

            # ── Consenso y decision de hablar ─────────────────────
            best_idx, score = detector.consensus()
            if best_idx is not None:
                clase_str = clases[best_idx]
                ahora     = time.time()
                ultimo    = cooldowns.get(clase_str, 0.0)

                # Limpiar SIEMPRE al alcanzar consenso.
                # Si no se limpia aqui, el signo anterior bloquea la ventana
                # incluso cuando el usuario ya cambio de sena.
                detector.clear()

                if ahora - ultimo >= COOLDOWN_TTS_SEG:
                    cooldowns[clase_str] = ahora
                    texto_voz = texto_para_voz(clase_str)
                    print(f"  >>> {clase_str}  (score={score:.2f})")

                    if modo_frase:
                        phrase_buf.add(clase_str)
                        label_text = f"{clase_str}  [acumulado]"
                    else:
                        hablar_async(texto_voz)
                        label_text = f">> {clase_str} <<"

            # ── Flush de frase si hay pausa ───────────────────────
            if modo_frase and phrase_buf.should_flush():
                frase = phrase_buf.flush()
                textos_voz = [texto_para_voz(s) for s in frase]
                print(f"  FRASE: {' '.join(frase)}")
                hablar_async(" ".join(textos_voz))
                label_text = " ".join(frase)

            # ── FPS ───────────────────────────────────────────────
            t_now = time.time()
            dt = max(t_now - t_prev, 1e-6)
            t_prev = t_now
            fps_inst   = 1.0 / dt
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps_inst if fps_smooth > 0 else fps_inst

            # ── Dibujo ────────────────────────────────────────────
            dibujar_panel_superior(frame, fps_smooth, hand_ok, face_ok,
                                   len(clases), modo_frase)
            dibujar_barra_confianza(frame, ui_clase_top, ui_prob_top, clases)
            if modo_frase:
                dibujar_buffer_frase(frame, phrase_buf.content)
            dibujar_subtitulo(frame, label_text)
            cv2.imshow("LSC - Reconocimiento Fase 1", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                phrase_buf.clear()
                detector.clear()
                print("  [buffer limpiado]")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if face_detector:
            try: face_detector.close()
            except Exception: pass
        try: hand_detector.close()
        except Exception: pass
        _tts_queue.put(None)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Reconocimiento LSC en vivo")
    p.add_argument("--camera",     type=int,   default=None,  help="Indice de camara")
    p.add_argument("--modo-frase", action="store_true",
                   help="Acumula senas y habla la frase cuando hay pausa")
    p.add_argument("--phrase-gap", type=float, default=2.0,
                   help="Segundos de pausa para emitir la frase (default: 2.0)")
    args = p.parse_args()
    predecir_tiempo_real(camera_index=args.camera,
                         modo_frase=args.modo_frase,
                         phrase_gap=args.phrase_gap)
