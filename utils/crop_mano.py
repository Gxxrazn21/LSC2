"""
=============================================================
UTILIDAD COMPARTIDA - CROP DE MANO + ANTEBRAZO
=============================================================

Devuelve un crop cuadrado centrado en las manos detectadas por
MediaPipe HandLandmarker, extendido hacia abajo para incluir
antebrazo. Se usa TANTO en el preprocesamiento del dataset como
en el reconocimiento en vivo (predecir_voz.py) para garantizar
que el entrenamiento y la inferencia vean el mismo recorte.
=============================================================
"""

from typing import Iterable, Optional, Tuple

import numpy as np


# Margenes por defecto (ajustables)
EXPAND_LATERAL = 0.25   # +25% a cada lado
EXPAND_ARRIBA = 0.20    # +20% hacia arriba
EXPAND_ABAJO = 0.80     # +80% hacia abajo para capturar antebrazo


def _collect_xy(hand_landmarks_list: Iterable, img_w: int, img_h: int):
    """Convierte landmarks normalizados (0..1) a pixeles y devuelve listas."""
    xs, ys = [], []
    for hand_lm in hand_landmarks_list:
        for lm in hand_lm:
            xs.append(float(lm.x) * img_w)
            ys.append(float(lm.y) * img_h)
    return xs, ys


def bbox_manos_antebrazo(
    hand_landmarks_list: Iterable,
    img_w: int,
    img_h: int,
    expand_lateral: float = EXPAND_LATERAL,
    expand_arriba: float = EXPAND_ARRIBA,
    expand_abajo: float = EXPAND_ABAJO,
) -> Optional[Tuple[int, int, int, int]]:
    """Calcula bbox cuadrado con margenes extendidos hacia abajo para antebrazo.

    Args:
        hand_landmarks_list: lista de listas de NormalizedLandmark
                             (resultado de HandLandmarker.detect().hand_landmarks).
        img_w, img_h: dimensiones de la imagen original en pixeles.
        expand_lateral: fraccion de ancho a expandir a cada lado.
        expand_arriba: fraccion de alto a expandir hacia arriba.
        expand_abajo: fraccion de alto a expandir hacia abajo (antebrazo).

    Returns:
        (x1, y1, x2, y2) en pixeles o None si no hay manos.
    """
    xs, ys = _collect_xy(hand_landmarks_list, img_w, img_h)
    if not xs:
        return None

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    # Expandir caja cruda
    x1e = x1 - w * expand_lateral
    x2e = x2 + w * expand_lateral
    y1e = y1 - h * expand_arriba
    y2e = y2 + h * expand_abajo

    # Hacer cuadrado (tomar el lado mayor) centrado en el centro actual
    cx = (x1e + x2e) / 2
    cy = (y1e + y2e) / 2
    side = max(x2e - x1e, y2e - y1e)
    half = side / 2
    x1f = cx - half
    x2f = cx + half
    y1f = cy - half
    y2f = cy + half

    # Recortar a los limites de la imagen
    x1f = max(0, int(round(x1f)))
    y1f = max(0, int(round(y1f)))
    x2f = min(img_w, int(round(x2f)))
    y2f = min(img_h, int(round(y2f)))

    # Sanity: bbox minimo para evitar crops degenerados
    if x2f - x1f < 16 or y2f - y1f < 16:
        return None

    return (x1f, y1f, x2f, y2f)


def recortar_crop(
    img_bgr: np.ndarray,
    hand_landmarks_list: Iterable,
    out_size: int = 128,
    **kwargs,
) -> Optional[np.ndarray]:
    """Recorta y redimensiona un crop cuadrado mano+antebrazo desde una imagen BGR.

    Returns:
        ndarray (out_size, out_size, 3) uint8 BGR, o None si no se puede.
    """
    import cv2

    if img_bgr is None or img_bgr.size == 0:
        return None
    h, w = img_bgr.shape[:2]
    bbox = bbox_manos_antebrazo(hand_landmarks_list, w, h, **kwargs)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
