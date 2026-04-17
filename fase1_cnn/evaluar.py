"""
FASE 1 - EVALUACION (multi-contexto)

Uso:
    python -m fase1_cnn.evaluar
    python -m fase1_cnn.evaluar --contexto saludos
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import FASE1, LSC70_PATH, DEVICE, RESULTADOS_DIR, SEED
from fase1_cnn.modelo import crear_modelo
from fase1_cnn.dataset import LSCDataset
from fase1_cnn.contextos import clases_de, rutas_de, listar_contextos
from utils.metricas import evaluar_modelo, reporte_clasificacion
from utils.visualizacion import graficar_matriz_confusion

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _get_persons(dataset_path: str) -> list:
    import glob
    return sorted({
        os.path.basename(p)
        for p in glob.glob(os.path.join(dataset_path, "LSC70W", "Per*"))
        if os.path.isdir(p)
    })


def evaluar(contexto: str):
    cfg        = FASE1
    clases_obj = clases_de(contexto)
    rutas      = rutas_de(contexto)

    print("\n" + "=" * 60)
    print(f"  EVALUACION — FASE 1 — CONTEXTO: {contexto.upper()}")
    print("=" * 60)

    if not os.path.exists(rutas["modelo"]):
        print(f"  ERROR: No se encontró modelo en {rutas['modelo']}")
        print(f"         Ejecuta: python -m fase1_cnn.entrenar --contexto {contexto}")
        return None

    # Reproducir el mismo split por persona que usó entrenar.py
    all_persons = _get_persons(dataset_path=LSC70_PATH)
    rng = np.random.default_rng(SEED)
    shuffled = list(all_persons)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * cfg["valid_split"]))
    val_persons = shuffled[:n_val]

    transform_val = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    val_ds = LSCDataset(LSC70_PATH, transform=transform_val,
                        target_gestures=clases_obj, person_filter=val_persons)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=0)
    nombres_clases = val_ds.classes

    model = crear_modelo(len(nombres_clases), cfg["img_size"], DEVICE, freeze_backbone=False)
    model.load_state_dict(torch.load(rutas["modelo"], map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"  Modelo cargado: {rutas['modelo']}")

    all_preds, all_labels = evaluar_modelo(model, val_loader, DEVICE)

    ruta_reporte = os.path.join(RESULTADOS_DIR, f"fase1_{contexto}_reporte.txt")
    resultado = reporte_clasificacion(all_labels, all_preds, nombres_clases,
                                      guardar_en=ruta_reporte)

    ruta_cm = os.path.join(RESULTADOS_DIR, f"fase1_{contexto}_matriz_confusion.png")
    try:
        graficar_matriz_confusion(
            resultado["confusion_matrix"],
            nombres_clases,
            titulo=f"Matriz de Confusión — Fase 1 ({contexto})",
            guardar_en=ruta_cm,
        )
    except Exception as e:
        print(f"  [aviso] no se pudo graficar matriz: {e}")

    print("\n  Precisión por clase (val):")
    print("  " + "-" * 50)
    cm = resultado["confusion_matrix"]
    total_correctos = 0
    total_muestras  = 0
    per_class = []
    for i, nombre in enumerate(nombres_clases):
        tp       = int(cm[i, i])
        sum_fila = int(cm[i].sum())
        acc      = 100.0 * tp / sum_fila if sum_fila > 0 else 0.0
        total_correctos += tp
        total_muestras  += sum_fila
        per_class.append((acc, nombre, tp, sum_fila))
        print(f"    {nombre:12s}  {tp:4d}/{sum_fila:<4d}  ({acc:5.1f}%)")

    overall = 100.0 * total_correctos / max(total_muestras, 1)
    print("  " + "-" * 50)
    print(f"  Accuracy global: {overall:.2f}%  ({total_correctos}/{total_muestras})")

    per_class.sort()
    print("\n  Clases con peor desempeño (top 3):")
    for acc, nombre, tp, tot in per_class[:3]:
        print(f"    {nombre:12s}  {acc:5.1f}%  ({tp}/{tot})")

    print(f"\n  Reporte: {ruta_reporte}")
    print(f"  Matriz:  {ruta_cm}")
    return resultado


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--contexto", default="saludos", choices=listar_contextos())
    args = p.parse_args()
    evaluar(args.contexto)
