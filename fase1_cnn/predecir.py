"""
=============================================================
FASE 1 — PREDICCION (INFERENCIA)
Usa el modelo entrenado para predecir la sena en una imagen nueva.
=============================================================
"""

import sys
import os
import torch
from torchvision import transforms
from PIL import Image
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import FASE1, DEVICE, MODELOS_DIR, HAND_LANDMARKER_PATH
from fase1_cnn.modelo import LSC_CNN
from utils.crop_mano import recortar_crop


def cargar_clases(clases_path=None):
    """Carga los nombres de las clases desde el JSON guardado en entrenamiento."""
    import json
    if clases_path is None:
        clases_path = FASE1.get("clases_path",
                                os.path.join(MODELOS_DIR, "clases_fase1.json"))
    if not os.path.exists(clases_path):
        raise FileNotFoundError(
            f"No se encontro el archivo de clases: {clases_path}\n"
            "Ejecuta primero: python -m fase1_cnn.entrenar"
        )
    with open(clases_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def cargar_modelo_entrenado(num_classes, modelo_path=None):
    """
    Carga el modelo CNN entrenado para inferencia.

    Args:
        num_classes: numero de clases del modelo
        modelo_path: ruta al archivo .pth (usa default si es None)

    Returns:
        modelo en modo evaluacion
    """
    if modelo_path is None:
        modelo_path = FASE1["modelo_path"]

    if not os.path.exists(modelo_path):
        raise FileNotFoundError(
            f"No se encontro el modelo en: {modelo_path}\n"
            "Ejecuta primero el entrenamiento."
        )

    model = LSC_CNN(num_classes=num_classes, img_size=FASE1["img_size"]).to(DEVICE)
    model.load_state_dict(torch.load(modelo_path, map_location=DEVICE, weights_only=True))
    model.eval()

    print(f"  Modelo cargado desde: {modelo_path}")
    return model


def predecir_imagen(model, imagen_path, nombres_clases, top_k=5):
    """
    Detecta manos, recorta y predice la sena en una imagen.
    """
    if not os.path.exists(imagen_path):
        raise FileNotFoundError(f"No se encontro la imagen: {imagen_path}")

    # 1. Detectar mano para hacer el crop (igual que en entrenamiento)
    img_bgr = cv2.imread(imagen_path)
    if img_bgr is None:
        raise ValueError(f"No se pudo leer la imagen: {imagen_path}")
    
    rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    
    base_options = python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    
    with vision.HandLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp_image)
        
    if not result.hand_landmarks:
        print("  [AVISO] No se detectaron manos en la imagen. Se usará la imagen completa (no recomendado).")
        img_input = Image.fromarray(rgb_img)
    else:
        # Usar la utilidad de recorte
        crop_bgr = recortar_crop(img_bgr, result.hand_landmarks, out_size=FASE1["img_size"])
        if crop_bgr is not None:
            img_input = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        else:
            img_input = Image.fromarray(rgb_img)

    # 2. Transformaciones (deben coincidir con validacion de entrenamiento)
    transform = transforms.Compose([
        transforms.Resize((FASE1["img_size"], FASE1["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    img_tensor = transform(img_input).unsqueeze(0).to(DEVICE)  # Agregar dimension de batch

    # Predecir
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(nombres_clases)))

    # Formatear resultados
    top_probs = top_probs.squeeze().cpu().numpy()
    top_indices = top_indices.squeeze().cpu().numpy()

    predicciones = []
    for prob, idx in zip(top_probs, top_indices):
        predicciones.append({
            "clase": nombres_clases[idx],
            "probabilidad": float(prob) * 100,
        })

    resultado = {
        "prediccion": predicciones[0]["clase"],
        "confianza": predicciones[0]["probabilidad"],
        "top_k": predicciones,
    }

    print(f"\n  Imagen: {os.path.basename(imagen_path)}")
    print(f"  Prediccion: {resultado['prediccion']} ({resultado['confianza']:.1f}%)")
    print(f"  Top {top_k}:")
    for p in predicciones:
        barra = "#" * int(p["probabilidad"] / 5)
        print(f"    {p['clase']:>10s}: {p['probabilidad']:5.1f}% {barra}")

    return resultado


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predecir sena en una imagen")
    parser.add_argument("imagen", help="Ruta a la imagen a clasificar")
    parser.add_argument("--clases", type=int, default=47, help="Numero de clases")
    args = parser.parse_args()

    # Carga nombres de clases guardadas durante el entrenamiento
    try:
        nombres_clases = cargar_clases()
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    print(f"  Clases cargadas: {nombres_clases}")

    # Cargar modelo y predecir
    model = cargar_modelo_entrenado(len(nombres_clases))
    predecir_imagen(model, args.imagen, nombres_clases)
