"""
Uso:
    python -m fase1_cnn.entrenar
    python -m fase1_cnn.entrenar --contexto saludos --epochs 60
    python -m fase1_cnn.entrenar --unfreeze   # descongelar backbone completo
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import FASE1, LSC70_PATH, DEVICE, RESULTADOS_DIR, SEED, PRINT_EVERY
from fase1_cnn.modelo import crear_modelo
from fase1_cnn.dataset import LSCDataset
from fase1_cnn.contextos import clases_de, rutas_de, listar_contextos
from utils.visualizacion import graficar_curvas
from utils.metricas import calcular_accuracy, reporte_clasificacion, evaluar_modelo

# ─────────────────────────────────────────────────────────────────
# Normalización ImageNet (obligatoria con backbone preentrenado)
# ─────────────────────────────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _get_persons(dataset_path: str) -> list:
    """Devuelve lista ordenada de personas disponibles en LSC70W."""
    import glob
    persons = sorted({
        os.path.basename(p)
        for p in glob.glob(os.path.join(dataset_path, "LSC70W", "Per*"))
        if os.path.isdir(p)
    })
    return persons


def _build_loaders(dataset_path, img_size, batch_size, valid_split, clases_obj, rutas):
    from torchvision import transforms

    # ── Augmentación de entrenamiento ──────────────────────────────
    # SIN RandomHorizontalFlip — en LSC la lateralidad del signo es semántica
    transform_train = transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),   # ligeramente más grande
        transforms.RandomCrop(img_size),                      # recorte aleatorio
        transforms.RandomRotation(degrees=12),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.88, 1.12)),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
        transforms.ColorJitter(brightness=0.4, contrast=0.35, saturation=0.25, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    # ── Split por persona (sin data leakage) ───────────────────────
    all_persons = _get_persons(dataset_path)
    if not all_persons:
        print(f"  ERROR: no se encontraron carpetas Per* en {dataset_path}/LSC70W/")
        return None

    rng = np.random.default_rng(SEED)
    shuffled = list(all_persons)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * valid_split))
    val_persons   = shuffled[:n_val]
    train_persons = shuffled[n_val:]

    print(f"  Personas -> train: {len(train_persons)} | val: {len(val_persons)}")

    train_ds = LSCDataset(dataset_path, transform=transform_train,
                          target_gestures=clases_obj, person_filter=train_persons)
    val_ds   = LSCDataset(dataset_path, transform=transform_val,
                          target_gestures=clases_obj, person_filter=val_persons)

    if len(train_ds) == 0:
        print(f"  ERROR: dataset de entrenamiento vacío.")
        return None

    # Verificar clases con imágenes
    missing = [c for c in clases_obj if not any(
        train_ds.classes[l] == c for _, l in train_ds.samples)]
    if missing:
        print(f"  ADVERTENCIA: clases sin imágenes en train: {missing}")

    # Guardar lista de clases
    os.makedirs(os.path.dirname(rutas["clases"]), exist_ok=True)
    with open(rutas["clases"], "w", encoding="utf-8") as f:
        json.dump(train_ds.classes, f, ensure_ascii=False, indent=2)
    print(f"  Clases ({len(train_ds.classes)}): {train_ds.classes}")

    # WeightedRandomSampler para balance de clases
    class_counts = np.bincount([l for _, l in train_ds.samples], minlength=len(train_ds.classes))
    weights = 1.0 / (class_counts + 1e-6)
    sample_weights = torch.tensor([weights[l] for _, l in train_ds.samples], dtype=torch.float)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    return train_loader, val_loader, train_ds.classes


def entrenar(contexto: str, epochs: int = None, batch_size: int = None,
             learning_rate: float = None, unfreeze_backbone: bool = False):
    cfg        = FASE1
    clases_obj = clases_de(contexto)
    rutas      = rutas_de(contexto)

    epochs        = epochs        or cfg["epochs"]
    batch_size    = batch_size    or cfg["batch_size"]
    learning_rate = learning_rate or cfg["learning_rate"]

    print("\n" + "=" * 60)
    print(f"  ENTRENAMIENTO - contexto: {contexto.upper()}")
    print("=" * 60)

    datos = _build_loaders(LSC70_PATH, cfg["img_size"], batch_size,
                           cfg["valid_split"], clases_obj, rutas)
    if datos is None:
        return None
    train_loader, val_loader, nombres_clases = datos
    num_classes = len(nombres_clases)

    freeze = not unfreeze_backbone
    model = crear_modelo(num_classes, cfg["img_size"], DEVICE, freeze_backbone=freeze)

    # Label smoothing para regularización adicional
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizador con grupos de parámetros diferenciados
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "classifier" not in n]
    head_params     = list(model.classifier.parameters())

    if backbone_params:
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": learning_rate * 0.1},
            {"params": head_params,     "lr": learning_rate},
        ], weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(head_params, lr=learning_rate, weight_decay=1e-4)

    # Cosine annealing con warm-up de 3 épocas
    warmup_epochs = 3

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP (mixed precision) si hay GPU
    use_amp = DEVICE.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc   = 0.0
    patience_count = 0
    EARLY_STOP     = 15

    print(f"\n  Epocas={epochs}  batch={batch_size}  lr={learning_rate}"
          f"  early_stop={EARLY_STOP}  AMP={use_amp}")
    print("-" * 60)

    for epoch in range(epochs):
        # ── TRAIN ──────────────────────────────────────────────────
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                out  = model(imgs)
                loss = criterion(out, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item()
            bc, bt = calcular_accuracy(out, labels)
            correct += bc
            total   += bt

        train_loss = run_loss / max(len(train_loader), 1)
        train_acc  = 100 * correct / max(total, 1)

        # ── VAL ────────────────────────────────────────────────────
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                    out  = model(imgs)
                    loss = criterion(out, labels)
                v_loss    += loss.item()
                bc, bt = calcular_accuracy(out, labels)
                v_correct += bc
                v_total   += bt

        val_loss = v_loss / max(len(val_loader), 1)
        val_acc  = 100 * v_correct / max(v_total, 1)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # ── GUARDAR MEJOR MODELO ───────────────────────────────────
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save(model.state_dict(), rutas["modelo"])
        else:
            patience_count += 1

        if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0 or epoch == epochs - 1 or improved:
            lr_actual = optimizer.param_groups[-1]["lr"]
            print(
                f"  Epoca [{epoch+1:3d}/{epochs}] "
                f"Train {train_loss:.4f}/{train_acc:5.1f}%  "
                f"Val {val_loss:.4f}/{val_acc:5.1f}%  "
                f"lr={lr_actual:.2e}"
                + ("  *" if improved else "")
            )

        if patience_count >= EARLY_STOP:
            print(f"\n  Early stopping en epoca {epoch+1} (sin mejora en {EARLY_STOP} epocas).")
            break

    print("-" * 60)
    print(f"  Mejor val_acc: {best_val_acc:.2f}%")
    print(f"  Modelo: {rutas['modelo']}")
    print("  NOTA: val_acc basado en split por persona - mas realista que split aleatorio.")

    # ── REPORTE FINAL ─────────────────────────────────────────────
    print("\n  Cargando mejor modelo para reporte final...")
    model.load_state_dict(torch.load(rutas["modelo"], map_location=DEVICE, weights_only=True))
    all_preds, all_labels = evaluar_modelo(model, val_loader, DEVICE)
    ruta_reporte = os.path.join(RESULTADOS_DIR, f"reporte_{contexto}.txt")
    reporte_clasificacion(all_labels, all_preds, nombres_clases, guardar_en=ruta_reporte)

    # ── CURVAS ────────────────────────────────────────────────────
    ruta_grafica = os.path.join(RESULTADOS_DIR, f"curvas_{contexto}.png")
    try:
        graficar_curvas(history, titulo=f"Fase 1 ({contexto})", guardar_en=ruta_grafica)
        print(f"  Curvas: {ruta_grafica}")
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
    p.add_argument("--contexto",   default="saludos", choices=listar_contextos())
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch-size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--unfreeze",   action="store_true",
                   help="Descongelar backbone completo (usar tras convergencia inicial)")
    args = p.parse_args()

    res = entrenar(args.contexto, args.epochs, args.batch_size, args.lr, args.unfreeze)
    if res:
        print(f"\n  Listo. Mejor accuracy val: {res['best_val_acc']:.2f}%")
