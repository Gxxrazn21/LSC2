"""
=============================================================
PREPROCESAMIENTO - GENERACION DE CROPS MANO+ANTEBRAZO
=============================================================

Recorre todo el dataset LSC70 y para cada imagen genera un crop
centrado en las manos (con antebrazo) usando MediaPipe
HandLandmarker. Guarda los crops en un directorio paralelo para
que el entrenamiento lea imagenes ya "limpias" (sin cara ni
fondo distractor).

Uso:
    python -m utils.preprocesar_manos
    python -m utils.preprocesar_manos --src datasets/LSC70/LSC70ANH \
                                      --dst datasets/LSC70_HAND_CROPS \
                                      --size 128

Estructura de salida (mismos subdirectorios que el original):
    datasets/LSC70_HAND_CROPS/LSC70/LSC70/LSC70W/Per01/ANNOS/*.jpg

Las imagenes donde no se detecta mano se registran en
resultados/preprocesamiento_fallidas.txt y se SALTAN.
=============================================================
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mppy
from mediapipe.tasks.python import vision

# Acceso a config desde la raiz
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from config import HAND_LANDMARKER_PATH, LSC70_PATH, RESULTADOS_DIR, PROJECT_ROOT
from utils.crop_mano import recortar_crop


def crear_hand_detector():
    """Crea un HandLandmarker para procesamiento en modo IMAGE."""
    opts = vision.HandLandmarkerOptions(
        base_options=mppy.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,  # mas permisivo en dataset estatico
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(opts)


def recorrer_fuente(src_dir: str):
    """Devuelve lista de (ruta_original, ruta_relativa_desde_src) .jpg."""
    patron = os.path.join(src_dir, "**", "*.jpg")
    archivos = glob.glob(patron, recursive=True)
    return [(p, os.path.relpath(p, src_dir)) for p in archivos]


def main():
    parser = argparse.ArgumentParser(description="Preprocesa LSC70: crop mano+antebrazo")
    parser.add_argument("--src", default=LSC70_PATH, help="Directorio fuente")
    parser.add_argument("--dst", default=os.path.join(PROJECT_ROOT, "datasets", "LSC70_HAND_CROPS"),
                        help="Directorio de salida")
    parser.add_argument("--size", type=int, default=128, help="Tamano de cada crop (cuadrado)")
    parser.add_argument("--force", action="store_true", help="Reprocesar aunque ya exista")
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        print(f"[ERROR] Directorio fuente no existe: {args.src}")
        sys.exit(1)

    os.makedirs(args.dst, exist_ok=True)
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    log_fallidas = os.path.join(RESULTADOS_DIR, "preprocesamiento_fallidas.txt")

    archivos = recorrer_fuente(args.src)
    total = len(archivos)
    if total == 0:
        print(f"[ERROR] No se encontraron .jpg en {args.src}")
        sys.exit(1)

    print(f"Imagenes encontradas: {total}")
    print(f"Fuente:  {args.src}")
    print(f"Destino: {args.dst}")
    print(f"Tamano crop: {args.size}x{args.size}")
    print("-" * 60)

    detector = crear_hand_detector()
    procesadas = 0
    saltadas = 0
    fallidas = 0
    t0 = time.time()

    with open(log_fallidas, "w", encoding="utf-8") as log:
        log.write("# Imagenes sin mano detectada (se omitieron)\n")

        for idx, (src_path, rel) in enumerate(archivos, 1):
            dst_path = os.path.join(args.dst, rel)
            if not args.force and os.path.exists(dst_path):
                saltadas += 1
                continue

            img = cv2.imread(src_path)
            if img is None:
                log.write(f"{rel}\t(no se pudo leer)\n")
                fallidas += 1
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                result = detector.detect(mp_img)
            except Exception as e:
                log.write(f"{rel}\t(detector error: {e})\n")
                fallidas += 1
                continue

            if not result.hand_landmarks:
                log.write(f"{rel}\t(sin manos)\n")
                fallidas += 1
                continue

            crop = recortar_crop(img, result.hand_landmarks, out_size=args.size)
            if crop is None:
                log.write(f"{rel}\t(bbox invalido)\n")
                fallidas += 1
                continue

            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            cv2.imwrite(dst_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            procesadas += 1

            if idx % 100 == 0 or idx == total:
                elapsed = time.time() - t0
                rate = idx / max(elapsed, 1e-6)
                eta = (total - idx) / max(rate, 1e-6)
                print(f"[{idx:5d}/{total}] ok={procesadas} saltadas={saltadas} "
                      f"fallidas={fallidas} | {rate:.1f} img/s | ETA {eta:5.0f}s")

    detector.close()
    print("-" * 60)
    print(f"Terminado. Procesadas: {procesadas} | Saltadas: {saltadas} | Fallidas: {fallidas}")
    print(f"Log de fallidas: {log_fallidas}")


if __name__ == "__main__":
    main()
