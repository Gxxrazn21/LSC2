"""
=============================================================
CONFIGURACION CENTRAL DEL PROYECTO
Sistema de Reconocimiento de Lengua de Senas Colombiana (LSC)
=============================================================

Todos los hiperparametros y rutas en un solo lugar.
Modifica este archivo, o sobreescribe valores via archivo .env
en la raiz del proyecto (ver .env.example).
"""

import os
import torch

# ─────────────────────────────────────────────
# RUTAS DEL PROYECTO + CARGA DE .env
# ─────────────────────────────────────────────

# Directorio raiz del proyecto (donde esta este archivo)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path: str) -> None:
    """Carga variables desde un archivo .env muy simple (KEY=VALUE por linea).

    - Ignora lineas vacias y que empiezan con '#'.
    - No sobreescribe variables ya presentes en os.environ.
    - Soporta valores entre comillas simples o dobles.
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
    except Exception as e:
        print(f"  [config] Aviso: no se pudo leer .env ({e})")


# Cargar .env lo antes posible (si existe)
_load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def _env_path(var: str, default: str) -> str:
    """Lee una ruta de os.environ; si es relativa, la ancla a PROJECT_ROOT."""
    value = os.environ.get(var, default)
    if not os.path.isabs(value):
        value = os.path.join(PROJECT_ROOT, value)
    return value


def _env_int(var: str, default: int) -> int:
    try:
        return int(os.environ.get(var, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(var: str, default: float) -> float:
    try:
        return float(os.environ.get(var, str(default)))
    except (TypeError, ValueError):
        return default


# Rutas a los datasets (pueden sobrescribirse desde .env)
DATASETS_DIR = _env_path("LSC_DATASETS_DIR", "datasets")
LSC70_PATH = _env_path("LSC70_PATH", os.path.join("datasets", "LSC70_HAND_CROPS", "LSC70", "LSC70"))

# Rutas de salida
MODELOS_DIR = _env_path("LSC_MODELOS_DIR", "modelos_guardados")
RESULTADOS_DIR = _env_path("LSC_RESULTADOS_DIR", "resultados")

# Crear directorios si no existen
os.makedirs(MODELOS_DIR, exist_ok=True)
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DISPOSITIVO (CPU / GPU)
# ─────────────────────────────────────────────

_device_override = os.environ.get("LSC_DEVICE", "").strip().lower()
if _device_override in ("cuda", "gpu") and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif _device_override == "cpu":
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# FASE 1: CNN + LSC70
# ─────────────────────────────────────────────

FASE1 = {
    "batch_size": _env_int("FASE1_BATCH_SIZE", 32),
    "epochs": _env_int("FASE1_EPOCHS", 60),
    "learning_rate": _env_float("FASE1_LR", 3e-4),
    "img_size": _env_int("FASE1_IMG_SIZE", 224),   # requerido por MobileNetV3 preentrenado
    "valid_split": _env_float("FASE1_VALID_SPLIT", 0.20),
    "modelo_path": os.path.join(MODELOS_DIR, "mejor_modelo_fase1.pth"),
    "clases_path": os.path.join(MODELOS_DIR, "clases_fase1.json"),
}

# ─────────────────────────────────────────────
# MEDIAPIPE (DETECCION POR CAMARA)
# ─────────────────────────────────────────────

HAND_LANDMARKER_PATH = _env_path("HAND_LANDMARKER_PATH", "hand_landmarker.task")
FACE_DETECTOR_PATH = _env_path("FACE_DETECTOR_PATH", "face_detector.task")

# ─────────────────────────────────────────────
# RECONOCIMIENTO EN VIVO (camara, TTS, estabilidad)
# ─────────────────────────────────────────────

CAMERA_INDEX = _env_int("CAMERA_INDEX", 0)
UMBRAL_CONFIANZA = _env_float("UMBRAL_CONFIANZA", 0.70)
FRAMES_ESTABLES = _env_int("FRAMES_ESTABLES", 10)
COOLDOWN_TTS_SEG = _env_float("COOLDOWN_TTS_SEG", 1.5)
TTS_RATE = _env_int("TTS_RATE", 160)

# ─────────────────────────────────────────────
# CONFIGURACION GENERAL
# ─────────────────────────────────────────────

SEED = _env_int("SEED", 42)
PRINT_EVERY = _env_int("PRINT_EVERY", 5)


def resumen():
    """Imprime un resumen de la configuracion actual."""
    print("=" * 60)
    print("  CONFIGURACION DEL PROYECTO LSC (FASE 1)")
    print("=" * 60)
    print(f"  Dispositivo:        {DEVICE}")
    print(f"  Directorio raiz:    {PROJECT_ROOT}")
    print(f"  Datasets:           {DATASETS_DIR}")
    print(f"  Modelos guardados:  {MODELOS_DIR}")
    print(f"  Resultados:         {RESULTADOS_DIR}")
    print()
    print(f"  FASE 1 - CNN + LSC70")
    print(f"    Dataset: {LSC70_PATH}")
    print(f"    Epochs: {FASE1['epochs']}  |  Batch: {FASE1['batch_size']}  |  LR: {FASE1['learning_rate']}")
    print()
    print(f"  CAMARA / TTS")
    print(f"    Camera index:     {CAMERA_INDEX}")
    print(f"    Umbral conf:      {UMBRAL_CONFIANZA}")
    print(f"    Frames estables:  {FRAMES_ESTABLES}")
    print(f"    Cooldown TTS:     {COOLDOWN_TTS_SEG}s")
    print("=" * 60)


if __name__ == "__main__":
    resumen()
