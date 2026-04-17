import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODELOS_DIR  # noqa: E402

CONTEXTO = "saludos"

CLASES = ["HOLA", "BUENAS", "TARDES", "NOCHES", "NOMBRE"]

TRADUCCIONES_TTS = {
    "HOLA":   "hola",
    "BUENAS": "buenas",
    "TARDES": "tardes",
    "NOCHES": "noches",
    "NOMBRE": "nombre",
}


def listar_contextos() -> list:
    return [CONTEXTO]


def clases_de(contexto: str) -> list:
    return list(CLASES)


def rutas_de(contexto: str) -> dict:
    return {
        "modelo": os.path.join(MODELOS_DIR, f"mejor_modelo_{contexto}.pth"),
        "clases": os.path.join(MODELOS_DIR, f"clases_{contexto}.json"),
        "curvas": os.path.join(MODELOS_DIR, f"curvas_{contexto}.png"),
    }


def texto_para_voz(clase: str) -> str:
    return TRADUCCIONES_TTS.get(clase, clase.lower())
