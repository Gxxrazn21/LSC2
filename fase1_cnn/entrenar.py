"""
Uso:
    python -m fase1_cnn.entrenar                       # default: saludos
    python -m fase1_cnn.entrenar --contexto saludos --epochs 40
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import FASE1, LSC70_PATH, DEVICE, RESULTADOS_DIR, SEED, PRINT_EVERY
from fase1_cnn.modelo import crear_modelo
from fase1_cnn.dataset import LSCDataset
from fase1_cnn.contextos import clases_de, rutas_de, listar_contextos
from utils.visualizacion import graficar_curvas
from utils.metricas import calcular_accuracy, reporte_clasificacion, evaluar_modelo


def cargar_datos(dataset_path, img_size, batch_size, valid_split, clases_obj, rutas):
    return _build_loaders(dataset_path, img_size, batch_size, valid_split, clases_obj, rutas)


def _build_loaders(dataset_path, img_size, batch_size, valid_split, clases_obj, rutas):
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    full_train = LSCDataset(dataset_path, transform=transform_train,
                            target_gestures=clases_obj)
    full_val   = LSCDataset(dataset_path, transform=transform_val,
                            target_gestures=clases_obj)

    if len(full_train) == 0:
        print(f"  ERROR: No se encontraron imagenes en {dataset_path}")
        return None

    # Validar que todas las clases tienen imagenes
    missing = [c for c in clases_obj if c not in full_train.classes or
               not any(full_train.classes[l] == c for _, l in full_train.samples)]
    if missing:
        print(f"  ADVERTENCIA: clases sin imagenes en el dataset: {missing}")

    # Guardar lista de clases para inferencia
    os.makedirs(os.path.dirname(rutas["clases"]), exist_ok=True)
    with open(rutas["clases"], "w", encoding="utf-8") as f:
        json.dump(full_train.classes, f, ensure_ascii=False, indent=2)
    print(f"  Clases ({len(full_train.classes)}): {full_train.classes}")
    print(f"  Guardadas en: {rutas['clases']}")

    n = len(full_train)
    indices = list(range(n))
    np.random.seed(SEED)
    np.random.shuffle(indices)
    val_size = int(n * valid_split)
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    train_loader = DataLoader(Subset(full_train, train_idx),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(Subset(full_val, val_idx),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Muestras: train={len(train_idx)}  val={len(val_idx)}  total={n}")
    return train_loader, val_loader, full_train.classes


def entrenar(contexto: str, epochs: int = None, batch_size: int = None,
             learning_rate: float = None):
    cfg    = FASE1
    clases_obj = clases_de(contexto)
    rutas  = rutas_de(contexto)

    epochs        = epochs        or cfg["epochs"]
    batch_size    = batch_size    or cfg["batch_size"]
    learning_rate = learning_rate or cfg["learning_rate"]

    print("\n" + "=" * 60)
    print(f"  ENTRENAMIENTO — contexto: {contexto.upper()}")
    print("=" * 60)

    datos = _build_loaders(LSC70_PATH, cfg["img_size"], batch_size,
                           cfg["valid_split"], clases_obj, rutas)
    if datos is None:
        return None
    train_loader, val_loader, nombres_clases = datos
    num_classes = len(nombres_clases)

    model     = crear_modelo(num_classes, cfg["img_size"], DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc   = 0.0
    patience_count = 0
    EARLY_STOP     = 12  # epocas sin mejora antes de parar

    print(f"\n  Epocas={epochs}  batch={batch_size}  lr={learning_rate}"
          f"  early_stop={EARLY_STOP}")
    print("-" * 60)

    for epoch in range(epochs):
        # ── TRAIN ──────────────────────────────────────────────
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            bc, bt = calcular_accuracy(out, labels)
            correct += bc
            total   += bt
        train_loss = run_loss / max(len(train_loader), 1)
        train_acc  = 100 * correct / max(total, 1)

        # ── VAL ────────────────────────────────────────────────
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out  = model(imgs)
                loss = criterion(out, labels)
                v_loss    += loss.item()
                bc, bt = calcular_accuracy(out, labels)
                v_correct += bc
                v_total   += bt
        val_loss = v_loss / max(len(val_loader), 1)
        val_acc  = 100 * v_correct / max(v_total, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        scheduler.step(val_loss)

        # ── GUARDAR MEJOR MODELO ───────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save(model.state_dict(), rutas["modelo"])
        else:
            patience_count += 1

        if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0 or epoch == epochs - 1:
            lr_actual = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoca [{epoch+1:3d}/{epochs}] "
                f"Train {train_loss:.4f}/{train_acc:5.1f}%  "
                f"Val {val_loss:.4f}/{val_acc:5.1f}%  "
                f"lr={lr_actual:.2e}"
                + ("  *" if patience_count == 0 else "")
            )

        if patience_count >= EARLY_STOP:
            print(f"\n  Early stopping en epoca {epoch+1} "
                  f"(sin mejora en {EARLY_STOP} epocas).")
            break

    print("-" * 60)
    print(f"  Mejor val_acc: {best_val_acc:.2f}%")
    print(f"  Modelo : {rutas['modelo']}")

    # ── REPORTE FINAL POR CLASE ────────────────────────────────
    print("\n  Cargando mejor modelo para reporte final...")
    model.load_state_dict(torch.load(rutas["modelo"], map_location=DEVICE,
                                     weights_only=True))
    all_preds, all_labels = evaluar_modelo(model, val_loader, DEVICE)
    ruta_reporte = os.path.join(RESULTADOS_DIR, f"reporte_{contexto}.txt")
    reporte_clasificacion(all_labels, all_preds, nombres_clases,
                          guardar_en=ruta_reporte)

    # ── CURVAS ────────────────────────────────────────────────
    ruta_grafica = os.path.join(RESULTADOS_DIR, f"curvas_{contexto}.png")
    try:
        graficar_curvas(history, titulo=f"Fase 1 ({contexto})",
                        guardar_en=ruta_grafica)
        print(f"  Curvas : {ruta_grafica}")
    except Exception as e:
        print(f"  [aviso] no se pudieron graficar curvas: {e}")

    return {
        "contexto":       contexto,
        "model":          model,
        "history":        history,
        "best_val_acc":   best_val_acc,
        "nombres_clases": nombres_clases,
        "val_loader":     val_loader,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--contexto", default="saludos", choices=listar_contextos())
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch-size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    args = p.parse_args()

    res = entrenar(args.contexto, args.epochs, args.batch_size, args.lr)
    if res:
        print(f"\n  Listo. Mejor accuracy: {res['best_val_acc']:.2f}%")
