"""
=============================================================
UTILIDADES DE VISUALIZACION
Graficas de entrenamiento y matrices de confusion
=============================================================
"""

import matplotlib
matplotlib.use("Agg")  # Backend sin GUI para servidores
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def graficar_curvas(history, titulo="Entrenamiento", guardar_en=None):
    """
    Grafica las curvas de perdida y accuracy del entrenamiento.

    Args:
        history: dict con claves 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        titulo: titulo para la grafica
        guardar_en: ruta donde guardar la imagen (None = no guardar)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Grafica de perdida
    ax1.plot(epochs, history["train_loss"], label="Entrenamiento", color="#2196F3", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Validacion", color="#FF5722", linewidth=2)
    ax1.set_title(f"Perdida (Loss) — {titulo}", fontsize=14)
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Grafica de accuracy
    ax2.plot(epochs, history["train_acc"], label="Entrenamiento", color="#2196F3", linewidth=2)
    ax2.plot(epochs, history["val_acc"], label="Validacion", color="#FF5722", linewidth=2)
    ax2.set_title(f"Precision (Accuracy) — {titulo}", fontsize=14)
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if guardar_en:
        os.makedirs(os.path.dirname(guardar_en), exist_ok=True)
        plt.savefig(guardar_en, dpi=150, bbox_inches="tight")
        print(f"  Graficas guardadas en: {guardar_en}")

    plt.close(fig)
    return fig


def graficar_matriz_confusion(cm, clases, titulo="Matriz de Confusion", guardar_en=None):
    """
    Grafica una matriz de confusion como heatmap.

    Args:
        cm: numpy array con la matriz de confusion
        clases: lista de nombres de clases
        titulo: titulo para la grafica
        guardar_en: ruta donde guardar la imagen
    """
    fig_size = max(10, len(clases) * 0.35)
    plt.figure(figsize=(fig_size, fig_size * 0.9))

    # Si hay muchas clases, no mostrar numeros individuales
    mostrar_numeros = len(clases) <= 30

    sns.heatmap(
        cm,
        annot=mostrar_numeros,
        fmt="d" if mostrar_numeros else "",
        cmap="Blues",
        xticklabels=clases,
        yticklabels=clases,
        square=True,
    )
    plt.title(titulo, fontsize=16)
    plt.xlabel("Prediccion")
    plt.ylabel("Real")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    if guardar_en:
        os.makedirs(os.path.dirname(guardar_en), exist_ok=True)
        plt.savefig(guardar_en, dpi=150, bbox_inches="tight")
        print(f"  Matriz de confusion guardada en: {guardar_en}")

    plt.close()


def graficar_muestras(images, labels, clases, titulo="Muestras del Dataset", n=16):
    """
    Muestra una cuadricula de imagenes del dataset.

    Args:
        images: tensor de imagenes (N, C, H, W)
        labels: tensor de etiquetas
        clases: lista de nombres de clases
        n: numero de imagenes a mostrar
    """
    n = min(n, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for i in range(n):
        img = images[i].squeeze().numpy()
        if img.ndim == 2:
            axes[i].imshow(img, cmap="gray")
        else:
            img = np.transpose(img, (1, 2, 0))
            axes[i].imshow(img)
        axes[i].set_title(clases[labels[i]], fontsize=10)
        axes[i].axis("off")

    # Ocultar ejes sobrantes
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle(titulo, fontsize=14)
    plt.tight_layout()
    plt.close(fig)
    return fig
