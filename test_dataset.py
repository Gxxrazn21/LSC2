
import os
import sys

# Agregar raiz del proyecto al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from config import LSC70_PATH
from fase1_cnn.dataset import LSCDataset

def test_dataset():
    print(f"Probando LSCDataset con LSC70_PATH: {LSC70_PATH}")
    saludos = ["HOLA", "BUENAS", "DIAS", "TARDES", "NOCHES"]
    try:
        ds = LSCDataset(LSC70_PATH, target_gestures=saludos)
        print(f"Exito: se encontraron {len(ds)} muestras.")
        if len(ds) > 0:
            print(f"Clases encontradas: {ds.classes}")
            img, label = ds[0]
            print(f"Muestra 0 - Tipo de imagen: {type(img)}, Label: {label} ({ds.classes[label]})")
        else:
            print("ERROR: No se encontraron muestras.")
    except Exception as e:
        print(f"ERROR al cargar el dataset: {e}")

if __name__ == "__main__":
    test_dataset()
