"""
=============================================================
UTILIDADES DE METRICAS Y EVALUACION
Funciones compartidas para evaluar modelos en todas las fases
=============================================================
"""

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluar_modelo(model, dataloader, device, es_multimodal=False):
    """
    Evalua un modelo sobre un dataloader y retorna predicciones y etiquetas.

    Args:
        model: modelo de PyTorch
        dataloader: DataLoader con datos de evaluacion
        device: dispositivo (cpu/cuda)
        es_multimodal: True si el dataloader retorna (video, imu, landmarks, labels)

    Returns:
        all_preds: lista de predicciones
        all_labels: lista de etiquetas reales
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if es_multimodal:
                videos, imus, landmarks, labels = batch
                videos = videos.to(device)
                labels_dev = labels.to(device)
                imus = imus.to(device) if imus is not None else None
                landmarks = landmarks.to(device) if landmarks is not None else None
                outputs = model(videos, imus, landmarks)
            elif len(batch) == 3:
                # Fase 2: (secuencias, labels, lengths)
                sequences, labels, lengths = batch
                sequences = sequences.to(device)
                labels_dev = labels.to(device)
                outputs = model(sequences, lengths)
            else:
                # Fase 1: (images, labels)
                images, labels = batch
                images = images.to(device)
                labels_dev = labels.to(device)
                outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def reporte_clasificacion(all_labels, all_preds, nombres_clases, guardar_en=None):
    """
    Genera e imprime un reporte de clasificacion completo.

    Args:
        all_labels: etiquetas reales
        all_preds: predicciones del modelo
        nombres_clases: lista de nombres de clases
        guardar_en: ruta para guardar el reporte como texto

    Returns:
        dict con accuracy, reporte texto, y matriz de confusion
    """
    acc = accuracy_score(all_labels, all_preds) * 100

    reporte = classification_report(
        all_labels,
        all_preds,
        target_names=nombres_clases,
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 60)
    print("  REPORTE DE CLASIFICACION")
    print("=" * 60)
    print(f"\n  Accuracy global: {acc:.2f}%\n")
    print(reporte)

    if guardar_en:
        with open(guardar_en, "w", encoding="utf-8") as f:
            f.write(f"Accuracy global: {acc:.2f}%\n\n")
            f.write(reporte)
        print(f"  Reporte guardado en: {guardar_en}")

    return {
        "accuracy": acc,
        "reporte": reporte,
        "confusion_matrix": cm,
    }


def calcular_accuracy(outputs, labels):
    """Calcula accuracy de un batch (para uso dentro del loop de entrenamiento)."""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct, total
