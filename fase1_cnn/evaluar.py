"""
=============================================================
FASE 1 - EVALUACION (multi-contexto)
=============================================================

Uso:
    python -m fase1_cnn.evaluar                      # default: saludos
    python -m fase1_cnn.evaluar --contexto numeros
    python -m fase1_cnn.evaluar --contexto letras

Genera en resultados/:
    fase1_<contexto>_reporte.txt
    fase1_<contexto>_matriz_confusion.png
=============================================================
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import FASE1, LSC70_PATH, DEVICE, RESULTADOS_DIR
from fase1_cnn.modelo import LSC_CNN
from fase1_cnn.entrenar import cargar_datos
from fase1_cnn.contextos import clases_de, rutas_de, listar_contextos
from utils.metricas import evaluar_modelo, reporte_clasificacion
from utils.visualizacion import graficar_matriz_confusion


def evaluar(contexto: str):
    cfg = FASE1
    clases_obj = clases_de(contexto)
    rutas = rutas_de(contexto)

    print("\n" + "=" * 60)
    print(f"  EVALUACION - FASE 1 - CONTEXTO: {contexto.upper()}")
    print("=" * 60)

    datos = cargar_datos(LSC70_PATH, cfg["img_size"], cfg["batch_size"],
                         cfg["valid_split"], clases_obj, rutas)
    if datos is None:
        return None
    _, val_loader, nombres_clases = datos

    if not os.path.exists(rutas["modelo"]):
        print(f"  ERROR: No se encontro modelo en {rutas['modelo']}")
        print(f"         Entrenalo primero: python -m fase1_cnn.entrenar --contexto {contexto}")
        return None

    model = LSC_CNN(num_classes=len(nombres_clases), img_size=cfg["img_size"]).to(DEVICE)
    model.load_state_dict(torch.load(rutas["modelo"], map_location=DEVICE, weights_only=True))
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
            titulo=f"Matriz de Confusion - Fase 1 ({contexto})",
            guardar_en=ruta_cm,
        )
    except Exception as e:
        print(f"  [aviso] no se pudo graficar matriz: {e}")

    # Resumen por clase
    print("\n  Precision por clase (val):")
    print("  " + "-" * 50)
    cm = resultado["confusion_matrix"]
    total_correctos = 0
    total_muestras = 0
    peores = []
    for i, nombre in enumerate(nombres_clases):
        tp = int(cm[i, i])
        sum_fila = int(cm[i].sum())
        acc_clase = 100.0 * tp / sum_fila if sum_fila > 0 else 0.0
        total_correctos += tp
        total_muestras += sum_fila
        peores.append((acc_clase, nombre, tp, sum_fila))
        print(f"    {nombre:12s}  {tp:4d}/{sum_fila:<4d}  ({acc_clase:5.1f}%)")

    overall = 100.0 * total_correctos / max(total_muestras, 1)
    print("  " + "-" * 50)
    print(f"  Accuracy global:  {overall:.2f}%  ({total_correctos}/{total_muestras})")

    peores.sort()
    print("\n  Clases con peor desempeno (top 3):")
    for acc, nombre, tp, tot in peores[:3]:
        print(f"    {nombre:12s}  {acc:5.1f}%  ({tp}/{tot})")

    print(f"\n  Reporte: {ruta_reporte}")
    print(f"  Matriz:  {ruta_cm}")
    return resultado


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluacion Fase 1 multi-contexto")
    p.add_argument("--contexto", default="saludos", choices=listar_contextos())
    args = p.parse_args()
    evaluar(args.contexto)
