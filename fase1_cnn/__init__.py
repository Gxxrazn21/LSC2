"""
FASE 1: Clasificacion de Senas LSC con CNN
Dataset: LSC70 (imagenes de mano dominante, crops pre-procesados)
"""

from .modelo import LSC_CNN
from .entrenar import entrenar
from .evaluar import evaluar
