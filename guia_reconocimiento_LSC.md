# 🤟 Guía Completa: Sistema de Reconocimiento de Lengua de Señas Colombiana (LSC)

## Desarrollo Progresivo en 3 Fases con PyTorch

---

## 📋 Resumen de los Datasets

| Dataset | Tipo de datos | Señas | Muestras | Resolución/Formato |
|---------|--------------|-------|----------|-------------------|
| **LSC70** | Imágenes JPG | 47 señas (alfabeto + números + palabras) | 35,208 frames | 640×480 px (cuerpo) / 120×120 px (mano) |
| **LSC-54** | Landmarks 3D (series temporales JSON) | 54 señas (colores + frases + números) | 22 participantes × 3-5 repeticiones × 45 aumentos | Coordenadas 3D (cara, torso, manos) |
| **LSC50** | Video RGB-D + IMU + Landmarks | 50 señas | 5 participantes (nativos y no nativos) | Video + sensores inerciales + CSV landmarks |

---

## 🔧 Configuración Inicial del Entorno

Antes de todo, prepara tu entorno. Esto funciona **sin GPU potente** (CPU es suficiente para empezar).

```bash
# Crear entorno virtual
python -m venv lsc_env
source lsc_env/bin/activate  # Linux/Mac
# lsc_env\Scripts\activate   # Windows

# Instalar dependencias
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib scikit-learn pillow tqdm seaborn
```

> **¿Por qué la versión CPU de PyTorch?** Es más liviana (~200 MB vs ~2 GB con CUDA) y suficiente para modelos pequeños. Si después consigues GPU, solo cambias la instalación de PyTorch.

---

# FASE 1: Clasificación de Imágenes con CNN (LSC70)

## 1.1 ¿Qué vamos a hacer?

Imagina que le muestras una foto de una mano haciendo una seña al computador, y este te dice "eso es la letra A". Eso es clasificación de imágenes.

**Concepto clave — CNN (Red Neuronal Convolucional):**
Una CNN aprende a "ver" patrones en imágenes de forma jerárquica:
- Primeras capas → detectan bordes y texturas simples
- Capas medias → combinan bordes en formas (dedos, palma)
- Últimas capas → reconocen la seña completa

```
Imagen → [Filtros detectan bordes] → [Filtros detectan formas] → [Clasificador] → "Letra A"
```

## 1.2 Estructura Esperada del Dataset LSC70

Descarga el dataset de Mendeley Data y organízalo así:

```
LSC70/
├── LSC70ANH/           # Imágenes de mano dominante (120×120) ← USAREMOS ESTE
│   ├── A/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── B/
│   ├── 1/
│   └── ...
├── LSC70AN/            # Cuerpo completo (640×480)
└── LSC70W/             # Palabras (640×480)
```

> **Recomendación para principiantes:** Empieza con **LSC70ANH** (imágenes de mano a 120×120). Son más livianas y el modelo se enfoca directamente en la mano, sin distracciones del fondo.

## 1.3 Arquitectura Recomendada

Una CNN simple con 3 bloques convolucionales. Nada sofisticado, pero funcional.

```
Entrada (1 × 64 × 64)     ← Imagen en escala de grises, redimensionada
       │
  [Conv2d 32 filtros + ReLU + MaxPool]    ← Bloque 1: detecta bordes
       │
  [Conv2d 64 filtros + ReLU + MaxPool]    ← Bloque 2: detecta formas
       │
  [Conv2d 128 filtros + ReLU + MaxPool]   ← Bloque 3: detecta patrones complejos
       │
  [Aplanar → Linear 256 → Dropout → Linear 47]  ← Clasificador
       │
  Salida: probabilidad para cada una de las 47 señas
```

## 1.4 Código Completo — Fase 1

```python
"""
=============================================================
FASE 1: Clasificación de Señas LSC con CNN
Dataset: LSC70 (imágenes de mano dominante)
=============================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# ─────────────────────────────────────────────
# PASO 1: CONFIGURACIÓN
# ─────────────────────────────────────────────

# Ajusta esta ruta a donde tengas el dataset
DATASET_PATH = "./LSC70/LSC70ANH"  # Carpeta con subcarpetas por seña

# Hiperparámetros (valores razonables para empezar)
BATCH_SIZE = 32      # Cuántas imágenes procesa a la vez
EPOCHS = 30          # Cuántas veces recorre todo el dataset
LEARNING_RATE = 0.001  # Qué tan rápido aprende (muy alto = inestable, muy bajo = lento)
IMG_SIZE = 64        # Redimensionar imágenes a 64×64 (más rápido que 120×120)
VALID_SPLIT = 0.2    # 20% de datos para validación

# Detectar si hay GPU disponible (funciona igual con CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

# ─────────────────────────────────────────────
# PASO 2: CARGAR Y PREPARAR LOS DATOS
# ─────────────────────────────────────────────

# Transformaciones que se aplican a cada imagen
# Esto es como "preprocesar" la foto antes de dársela al modelo
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convertir a escala de grises
    transforms.Resize((IMG_SIZE, IMG_SIZE)),        # Redimensionar
    transforms.RandomHorizontalFlip(p=0.3),        # Voltear horizontalmente (aumentación)
    transforms.RandomRotation(10),                  # Rotar ligeramente (aumentación)
    transforms.ToTensor(),                          # Convertir a tensor (0-1)
    transforms.Normalize([0.5], [0.5])             # Normalizar a rango (-1, 1)
])

transform_val = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ImageFolder carga automáticamente las imágenes usando la estructura de carpetas
# Cada subcarpeta = una clase (A, B, C, 1, 2, 3, etc.)
full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform_train)

# Ver cuántas clases y muestras tenemos
num_classes = len(full_dataset.classes)
print(f"\nClases encontradas: {num_classes}")
print(f"Total de imágenes: {len(full_dataset)}")
print(f"Nombres de clases: {full_dataset.classes[:10]}...")  # Primeras 10

# Dividir en entrenamiento (80%) y validación (20%)
val_size = int(len(full_dataset) * VALID_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# IMPORTANTE: Aplicar transformaciones sin aumentación al conjunto de validación
# (Para eso en un caso más riguroso se crearían datasets separados,
#  pero para empezar esto funciona bien)

# DataLoaders: se encargan de entregar los datos en lotes (batches)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Imágenes de entrenamiento: {train_size}")
print(f"Imágenes de validación: {val_size}")

# ─────────────────────────────────────────────
# PASO 3: DEFINIR EL MODELO CNN
# ─────────────────────────────────────────────

class LSC_CNN(nn.Module):
    """
    CNN simple con 3 bloques convolucionales.
    Cada bloque: Convolución → Activación ReLU → MaxPooling
    Al final: capas lineales para clasificar.
    """
    def __init__(self, num_classes):
        super(LSC_CNN, self).__init__()
        
        # Bloque 1: Entrada 1 canal → 32 filtros
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32 filtros de 3×3
            nn.BatchNorm2d(32),   # Normalización (estabiliza el entrenamiento)
            nn.ReLU(),            # Activación: convierte negativos en 0
            nn.MaxPool2d(2, 2)    # Reduce tamaño a la mitad: 64→32
        )
        
        # Bloque 2: 32 canales → 64 filtros
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)    # 32→16
        )
        
        # Bloque 3: 64 canales → 128 filtros
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)    # 16→8
        )
        
        # Clasificador (capas fully connected)
        # Después de 3 MaxPool: 64/(2^3) = 8, entonces 128 × 8 × 8 = 8192
        self.classifier = nn.Sequential(
            nn.Flatten(),              # Aplana de 3D a 1D
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),           # Apaga 50% de neuronas al azar (evita sobreajuste)
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x

# Crear el modelo y moverlo al dispositivo
model = LSC_CNN(num_classes=num_classes).to(device)

# Verificar la arquitectura
print(f"\nArquitectura del modelo:")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal de parámetros: {total_params:,}")

# ─────────────────────────────────────────────
# PASO 4: CONFIGURAR ENTRENAMIENTO
# ─────────────────────────────────────────────

# Función de pérdida: mide qué tan equivocado está el modelo
criterion = nn.CrossEntropyLoss()

# Optimizador: ajusta los pesos para reducir la pérdida
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Scheduler: reduce el learning rate si el modelo deja de mejorar
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5
)

# ─────────────────────────────────────────────
# PASO 5: BUCLE DE ENTRENAMIENTO
# ─────────────────────────────────────────────

# Listas para guardar métricas (las usaremos para graficar)
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': []
}

best_val_acc = 0.0  # Para guardar el mejor modelo

print("\n" + "="*60)
print("INICIANDO ENTRENAMIENTO")
print("="*60)

for epoch in range(EPOCHS):
    # --- FASE DE ENTRENAMIENTO ---
    model.train()  # Modo entrenamiento (activa Dropout y BatchNorm)
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Mover datos al dispositivo (CPU o GPU)
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass: pasar imágenes por el modelo
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass: calcular gradientes y actualizar pesos
        optimizer.zero_grad()  # Limpiar gradientes anteriores
        loss.backward()        # Calcular nuevos gradientes
        optimizer.step()       # Actualizar pesos
        
        # Estadísticas
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Clase con mayor probabilidad
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    # --- FASE DE VALIDACIÓN ---
    model.eval()  # Modo evaluación (desactiva Dropout)
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():  # No calcular gradientes (ahorra memoria)
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    
    # Guardar métricas
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    # Actualizar scheduler
    scheduler.step(val_loss)
    
    # Guardar mejor modelo
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "mejor_modelo_fase1.pth")
    
    # Imprimir progreso cada 5 épocas
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Época [{epoch+1:3d}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%")

print(f"\n✅ Mejor accuracy de validación: {best_val_acc:.1f}%")

# ─────────────────────────────────────────────
# PASO 6: VISUALIZAR EL APRENDIZAJE
# ─────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfica de pérdida
ax1.plot(history['train_loss'], label='Entrenamiento', color='#2196F3', linewidth=2)
ax1.plot(history['val_loss'], label='Validación', color='#FF5722', linewidth=2)
ax1.set_title('Pérdida (Loss) por Época', fontsize=14)
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfica de accuracy
ax2.plot(history['train_acc'], label='Entrenamiento', color='#2196F3', linewidth=2)
ax2.plot(history['val_acc'], label='Validación', color='#FF5722', linewidth=2)
ax2.set_title('Precisión (Accuracy) por Época', fontsize=14)
ax2.set_xlabel('Época')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fase1_curvas_aprendizaje.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 Gráficas guardadas en 'fase1_curvas_aprendizaje.png'")
```

## 1.5 ¿Cómo saber si el modelo está aprendiendo?

### Señales POSITIVAS (✅ va bien):
- **Train loss baja progresivamente** → El modelo aprende los datos
- **Val loss también baja** → Generaliza a datos nuevos
- **Train acc y val acc suben juntas** → Aprendizaje saludable
- **Las curvas se estabilizan** → Convergió

### Señales de PROBLEMAS (⚠️):
- **Train loss baja pero val loss sube** → **Sobreajuste (overfitting)**. El modelo memoriza en vez de aprender. Soluciones: más Dropout, data augmentation, reducir tamaño del modelo.
- **Ambas pérdidas permanecen altas** → **Subajuste (underfitting)**. El modelo es muy simple. Soluciones: más capas/filtros, más épocas, verificar que los datos cargan bien.
- **Loss oscila mucho** → Learning rate muy alto. Soluciones: reducir a 0.0001.
- **Accuracy estancada en ~2%** → El modelo adivina al azar (1/47 ≈ 2%). Verificar que las etiquetas son correctas.

## 1.6 Evaluación Detallada

```python
"""
EVALUACIÓN: Matriz de confusión y métricas por clase
Ejecutar después del entrenamiento
"""
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Cargar el mejor modelo
model.load_state_dict(torch.load("mejor_modelo_fase1.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Reporte de clasificación
print("\n📋 Reporte de Clasificación:")
print(classification_report(
    all_labels, all_preds,
    target_names=full_dataset.classes,
    zero_division=0
))

# Matriz de confusión
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=False, cmap='Blues',
            xticklabels=full_dataset.classes,
            yticklabels=full_dataset.classes)
plt.title('Matriz de Confusión - Fase 1', fontsize=16)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('fase1_matriz_confusion.png', dpi=150)
plt.show()
```

## 1.7 Errores Comunes y Soluciones

| Error | Causa probable | Solución |
|-------|---------------|----------|
| `RuntimeError: size mismatch` | El tamaño del Flatten no coincide con el Linear | Verifica: `128 × (IMG_SIZE / 2^3)^2` |
| `FileNotFoundError` en dataset | Ruta incorrecta o estructura de carpetas diferente | Imprime `os.listdir(DATASET_PATH)` para verificar |
| Accuracy ~2% sin mejorar | Etiquetas incorrectas o datos no cargan | Visualiza un batch: `plt.imshow(images[0].squeeze())` |
| `CUDA out of memory` | Batch size muy grande para tu GPU | Reduce `BATCH_SIZE` a 16 o 8 |
| El modelo no mejora después de 10 épocas | Learning rate inadecuado | Prueba 0.0005 o 0.0001 |

---

# FASE 2: Datos Secuenciales con LSTM (LSC-54)

## 2.1 ¿Qué cambia respecto a la Fase 1?

| Aspecto | Fase 1 (CNN + LSC70) | Fase 2 (LSTM + LSC-54) |
|---------|---------------------|----------------------|
| **Entrada** | Una imagen estática | Una secuencia de coordenadas 3D en el tiempo |
| **Lo que capta** | Forma de la mano en un instante | Movimiento completo de la seña |
| **Analogía** | Ver una foto | Ver un video |
| **Tipo de modelo** | CNN (patrones espaciales) | LSTM (patrones temporales) |

**Concepto clave — LSTM (Long Short-Term Memory):**
Una LSTM es como una red neuronal con "memoria". Puede recordar información de pasos anteriores en una secuencia. Perfecta para señas que involucran movimiento.

```
Tiempo 1: mano arriba      →  LSTM recuerda "mano arriba"
Tiempo 2: mano se mueve     →  LSTM recuerda "subió y se movió"
Tiempo 3: mano baja         →  LSTM decide: "Esa secuencia = seña HOLA"
```

## 2.2 Estructura del Dataset LSC-54

LSC-54 usa landmarks extraídos con MediaPipe. Cada muestra es un archivo JSON con coordenadas 3D a lo largo del tiempo.

```
LSC-54/
├── colores/
│   ├── rojo/
│   │   ├── participante01_rep1.json
│   │   ├── participante01_rep2.json
│   │   └── ...
│   ├── azul/
│   └── ...
├── frases/
│   ├── hola/
│   ├── gracias/
│   └── ...
└── numeros/
    ├── 1/
    ├── 2/
    └── ...
```

Cada JSON contiene algo como:
```json
{
  "frames": [
    {
      "frame_id": 0,
      "face_landmarks": [[x, y, z], ...],
      "left_hand_landmarks": [[x, y, z], ...],
      "right_hand_landmarks": [[x, y, z], ...],
      "pose_landmarks": [[x, y, z], ...]
    },
    ...
  ],
  "label": "hola"
}
```

## 2.3 Arquitectura Recomendada

```
Entrada: secuencia de T frames × F features
(T = número de frames, F = coordenadas aplanadas)
       │
  [LSTM bidireccional, 128 unidades, 2 capas]
       │
  [Tomar última salida oculta]
       │
  [Linear 256 → ReLU → Dropout → Linear 54]
       │
  Salida: probabilidad para cada seña
```

## 2.4 Código Completo — Fase 2

```python
"""
=============================================================
FASE 2: Clasificación de Señas con LSTM
Dataset: LSC-54 (landmarks secuenciales)
=============================================================
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PASO 1: CONFIGURACIÓN
# ─────────────────────────────────────────────

DATASET_PATH = "./LSC-54"
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128      # Tamaño de la memoria del LSTM
NUM_LAYERS = 2         # Capas de LSTM apiladas
MAX_SEQ_LEN = 60       # Máximo de frames por secuencia (truncar/padding)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

# ─────────────────────────────────────────────
# PASO 2: DATASET PERSONALIZADO
# ─────────────────────────────────────────────

class LSC54Dataset(Dataset):
    """
    Carga los archivos JSON de LSC-54 y los convierte en tensores.
    
    Cada muestra es una secuencia temporal de landmarks:
    - Mano derecha: 21 puntos × 3 coordenadas = 63 features
    - Mano izquierda: 21 puntos × 3 coordenadas = 63 features
    - Pose (torso/brazos): seleccionamos los 11 puntos relevantes × 3 = 33 features
    - Total por frame: 63 + 63 + 33 = 159 features
    """
    
    def __init__(self, root_dir, max_seq_len=60):
        self.samples = []      # Lista de (secuencia, etiqueta)
        self.labels = []
        self.label_map = {}    # Mapeo nombre → número
        self.max_seq_len = max_seq_len
        
        # Recorrer todas las carpetas y cargar archivos
        label_idx = 0
        for category in sorted(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category)
            if not os.path.isdir(category_path):
                continue
                
            for sign_name in sorted(os.listdir(category_path)):
                sign_path = os.path.join(category_path, sign_name)
                if not os.path.isdir(sign_path):
                    continue
                
                # Asignar un número a cada seña
                if sign_name not in self.label_map:
                    self.label_map[sign_name] = label_idx
                    label_idx += 1
                
                # Cargar cada archivo JSON
                for fname in os.listdir(sign_path):
                    if fname.endswith('.json'):
                        fpath = os.path.join(sign_path, fname)
                        try:
                            sequence = self._load_json(fpath)
                            if sequence is not None and len(sequence) > 0:
                                self.samples.append(sequence)
                                self.labels.append(self.label_map[sign_name])
                        except Exception as e:
                            print(f"Error cargando {fpath}: {e}")
        
        self.num_classes = len(self.label_map)
        print(f"Dataset cargado: {len(self.samples)} muestras, {self.num_classes} clases")
        print(f"Clases: {list(self.label_map.keys())[:10]}...")
    
    def _load_json(self, filepath):
        """
        Carga un JSON y extrae los landmarks relevantes.
        Adapta esta función según el formato exacto de tus archivos.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        frames = data.get('frames', data)  # Adaptable al formato
        if isinstance(frames, dict):
            frames = list(frames.values())
        
        sequence = []
        for frame in frames:
            features = []
            
            # Extraer landmarks de manos (21 puntos × 3 coords cada una)
            for hand_key in ['right_hand_landmarks', 'left_hand_landmarks']:
                hand_data = frame.get(hand_key, [])
                if hand_data and len(hand_data) > 0:
                    for point in hand_data[:21]:  # 21 puntos por mano
                        if isinstance(point, (list, tuple)) and len(point) >= 3:
                            features.extend([float(point[0]), float(point[1]), float(point[2])])
                        else:
                            features.extend([0.0, 0.0, 0.0])
                else:
                    features.extend([0.0] * 63)  # 21 puntos × 3 = 63
            
            # Extraer landmarks de pose/torso (puntos relevantes del cuerpo)
            pose_data = frame.get('pose_landmarks', [])
            # Índices de MediaPipe para torso/brazos: 11-22
            pose_indices = list(range(11, 23)) if len(pose_data) >= 23 else range(len(pose_data))
            for idx in pose_indices[:11]:
                if idx < len(pose_data):
                    point = pose_data[idx]
                    if isinstance(point, (list, tuple)) and len(point) >= 3:
                        features.extend([float(point[0]), float(point[1]), float(point[2])])
                    else:
                        features.extend([0.0, 0.0, 0.0])
                else:
                    features.extend([0.0, 0.0, 0.0])
            
            if len(features) > 0:
                sequence.append(features)
        
        return sequence if len(sequence) > 0 else None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sequence = self.samples[idx]
        label = self.labels[idx]
        
        # Convertir a tensor
        seq_tensor = torch.FloatTensor(sequence)
        
        # Truncar si es muy largo
        if len(seq_tensor) > self.max_seq_len:
            seq_tensor = seq_tensor[:self.max_seq_len]
        
        return seq_tensor, label


def collate_fn(batch):
    """
    Función personalizada para manejar secuencias de diferente longitud.
    Rellena (pad) las secuencias cortas con ceros para que todas tengan
    el mismo largo en un batch.
    """
    sequences, labels = zip(*batch)
    
    # Guardar largos originales (útil para pack_padded_sequence)
    lengths = torch.LongTensor([len(seq) for seq in sequences])
    
    # Pad: rellenar con ceros hasta que todas tengan el mismo largo
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    labels = torch.LongTensor(labels)
    
    return padded, labels, lengths


# ─────────────────────────────────────────────
# PASO 3: CARGAR DATOS
# ─────────────────────────────────────────────

dataset = LSC54Dataset(DATASET_PATH, max_seq_len=MAX_SEQ_LEN)

# Dividir en entrenamiento y validación
val_size = int(len(dataset) * 0.2)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, collate_fn=collate_fn
)

# Determinar el tamaño de features de entrada
sample_seq, _, _ = next(iter(train_loader))
INPUT_SIZE = sample_seq.shape[2]  # Número de features por frame
print(f"Features por frame: {INPUT_SIZE}")

# ─────────────────────────────────────────────
# PASO 4: DEFINIR EL MODELO LSTM
# ─────────────────────────────────────────────

class LSC_LSTM(nn.Module):
    """
    LSTM bidireccional para clasificación de secuencias temporales.
    
    Bidireccional = procesa la secuencia de inicio a fin Y de fin a inicio,
    capturando contexto en ambas direcciones.
    """
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSC_LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Capa de normalización de entrada
        self.input_norm = nn.LayerNorm(input_size)
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # Formato: (batch, secuencia, features)
            bidirectional=True,     # Procesa en ambas direcciones
            dropout=0.3 if num_layers > 1 else 0  # Dropout entre capas LSTM
        )
        
        # Clasificador
        # × 2 porque es bidireccional (concatena ambas direcciones)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, lengths=None):
        # Normalizar entrada
        x = self.input_norm(x)
        
        # Pasar por LSTM
        # output shape: (batch, seq_len, hidden_size * 2)
        output, (hidden, cell) = self.lstm(x)
        
        # Tomar la última salida oculta de ambas direcciones
        # hidden shape: (num_layers * 2, batch, hidden_size)
        # Concatenar la última capa forward y backward
        forward_hidden = hidden[-2]   # Última capa, dirección forward
        backward_hidden = hidden[-1]  # Última capa, dirección backward
        combined = torch.cat((forward_hidden, backward_hidden), dim=1)
        
        # Clasificar
        out = self.classifier(combined)
        return out


# Crear modelo
model_lstm = LSC_LSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=dataset.num_classes
).to(device)

print(f"\nArquitectura LSTM:")
print(model_lstm)
total_params = sum(p.numel() for p in model_lstm.parameters())
print(f"Total de parámetros: {total_params:,}")

# ─────────────────────────────────────────────
# PASO 5: ENTRENAMIENTO
# ─────────────────────────────────────────────

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0.0

print("\n" + "="*60)
print("INICIANDO ENTRENAMIENTO LSTM")
print("="*60)

for epoch in range(EPOCHS):
    # --- ENTRENAMIENTO ---
    model_lstm.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for padded_seqs, labels, lengths in train_loader:
        padded_seqs = padded_seqs.to(device)
        labels = labels.to(device)
        
        outputs = model_lstm(padded_seqs, lengths)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: evita que los gradientes "exploten"
        torch.nn.utils.clip_grad_norm_(model_lstm.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    # --- VALIDACIÓN ---
    model_lstm.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    
    with torch.no_grad():
        for padded_seqs, labels, lengths in val_loader:
            padded_seqs = padded_seqs.to(device)
            labels = labels.to(device)
            
            outputs = model_lstm(padded_seqs, lengths)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    scheduler.step(val_loss)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_lstm.state_dict(), "mejor_modelo_fase2.pth")
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Época [{epoch+1:3d}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%")

print(f"\n✅ Mejor accuracy de validación: {best_val_acc:.1f}%")

# Visualizar curvas (mismo código que Fase 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history['train_loss'], label='Train', color='#2196F3', linewidth=2)
ax1.plot(history['val_loss'], label='Val', color='#FF5722', linewidth=2)
ax1.set_title('Pérdida - Fase 2 (LSTM)'); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.plot(history['train_acc'], label='Train', color='#2196F3', linewidth=2)
ax2.plot(history['val_acc'], label='Val', color='#FF5722', linewidth=2)
ax2.set_title('Accuracy - Fase 2 (LSTM)'); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fase2_curvas_aprendizaje.png', dpi=150)
plt.show()
```

## 2.5 Errores Comunes — Fase 2

| Error | Causa probable | Solución |
|-------|---------------|----------|
| `KeyError` al cargar JSON | El formato del JSON no coincide con lo esperado | Imprime `data.keys()` de un archivo para ver la estructura real |
| Pérdida NaN o Inf | Gradientes explotan | Asegurar `clip_grad_norm_` y reducir learning rate |
| Secuencias muy cortas | Archivos con pocos frames | Filtrar: `if len(sequence) < 5: continue` |
| Accuracy baja | Features no informativas | Verificar que las coordenadas no son todas 0 (mano no detectada) |
| `RuntimeError: expected scalar type Long` | Labels no son enteros | Asegurar `torch.LongTensor(labels)` |

---

# FASE 3: Fusión Multimodal (LSC50)

## 3.1 ¿Qué es la fusión multimodal?

Es como combinar lo que ves (video), lo que mides (sensores de movimiento) y lo que calculas (landmarks) para tomar una mejor decisión.

**Analogía:** Si quieres saber si alguien está bailando:
- Solo con una foto → Puedes ver la pose
- Solo con un acelerómetro → Sabes que se mueve rítmicamente
- Con ambos → Estás mucho más seguro

### Tipos de fusión:
1. **Fusión temprana (early fusion):** Concatenar todos los datos desde el inicio
2. **Fusión tardía (late fusion):** Cada tipo de dato tiene su propio modelo, y al final se combinan las predicciones ← **RECOMENDADA para empezar**
3. **Fusión intermedia:** Combinar representaciones intermedias

Usaremos **fusión tardía** porque es más simple y cada rama se puede entrenar/depurar por separado.

## 3.2 Dataset LSC50

LSC50 contiene por cada seña y participante:
- **Videos RGB** (frames de video normal)
- **Datos IMU** (acelerómetro + giroscopio de sensores en el cuerpo)
- **Landmarks** (coordenadas de articulaciones extraídas del video)

```
LSC50/
├── participant_01/
│   ├── sign_hola/
│   │   ├── video.mp4          # Video RGB
│   │   ├── imu_data.csv       # Datos del sensor inercial
│   │   └── landmarks.csv      # Coordenadas de articulaciones
│   ├── sign_gracias/
│   └── ...
├── participant_02/
└── ...
```

## 3.3 Arquitectura: Fusión Tardía

```
        Video RGB                  IMU (acelerómetro)           Landmarks
            │                           │                          │
    [CNN extrae features         [LSTM procesa              [LSTM procesa
     frame por frame]             secuencia IMU]             secuencia landmarks]
            │                           │                          │
    [LSTM captura                 [Vector 128]               [Vector 128]
     movimiento temporal]               │                          │
            │                           │                          │
    [Vector 128]                        │                          │
            │                           │                          │
            └───────────┬───────────────┘──────────────────────────┘
                        │
                [Concatenar: 128 + 128 + 128 = 384]
                        │
                [Linear 256 → ReLU → Dropout]
                        │
                [Linear 50]  →  Predicción
```

## 3.4 Código Completo — Fase 3

```python
"""
=============================================================
FASE 3: Fusión Multimodal
Dataset: LSC50 (Video + IMU + Landmarks)
=============================================================
"""

import os
import cv2  # pip install opencv-python
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PASO 1: CONFIGURACIÓN
# ─────────────────────────────────────────────

DATASET_PATH = "./LSC50"
BATCH_SIZE = 8           # Más bajo porque cada muestra es grande
EPOCHS = 30
LEARNING_RATE = 0.0005
MAX_FRAMES = 30          # Máximo de frames de video por muestra
IMG_SIZE = 64            # Tamaño de cada frame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# PASO 2: DATASET MULTIMODAL
# ─────────────────────────────────────────────

class LSC50MultimodalDataset(Dataset):
    """
    Carga las tres modalidades de LSC50:
    1. Video RGB → secuencia de frames como imágenes
    2. IMU → datos del acelerómetro/giroscopio como serie temporal
    3. Landmarks → coordenadas del cuerpo como serie temporal
    """
    
    def __init__(self, root_dir, max_frames=30, img_size=64):
        self.max_frames = max_frames
        self.img_size = img_size
        self.samples = []  # (video_path, imu_path, landmarks_path, label)
        self.label_map = {}
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        label_idx = 0
        for participant in sorted(os.listdir(root_dir)):
            part_path = os.path.join(root_dir, participant)
            if not os.path.isdir(part_path):
                continue
            
            for sign_folder in sorted(os.listdir(part_path)):
                sign_path = os.path.join(part_path, sign_folder)
                if not os.path.isdir(sign_path):
                    continue
                
                # Extraer nombre de la seña del nombre de carpeta
                sign_name = sign_folder.replace("sign_", "")
                if sign_name not in self.label_map:
                    self.label_map[sign_name] = label_idx
                    label_idx += 1
                
                # Buscar archivos de cada modalidad
                video_file = self._find_file(sign_path, ['.mp4', '.avi'])
                imu_file = self._find_file(sign_path, ['imu'], ext='.csv')
                landmarks_file = self._find_file(sign_path, ['landmark'], ext='.csv')
                
                if video_file:  # Mínimo necesitamos el video
                    self.samples.append({
                        'video': video_file,
                        'imu': imu_file,
                        'landmarks': landmarks_file,
                        'label': self.label_map[sign_name]
                    })
        
        self.num_classes = len(self.label_map)
        print(f"Dataset: {len(self.samples)} muestras, {self.num_classes} clases")
    
    def _find_file(self, directory, keywords, ext=None):
        """Busca un archivo que contenga alguna keyword en su nombre."""
        for fname in os.listdir(directory):
            fname_lower = fname.lower()
            for kw in keywords:
                if kw in fname_lower:
                    if ext is None or fname_lower.endswith(ext):
                        return os.path.join(directory, fname)
        return None
    
    def _load_video(self, video_path):
        """Extrae frames del video de forma uniforme."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return torch.zeros(self.max_frames, 1, self.img_size, self.img_size)
        
        # Seleccionar frames uniformemente distribuidos
        indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = self.transform(frame)
                frames.append(frame_tensor)
            else:
                frames.append(torch.zeros(1, self.img_size, self.img_size))
        
        cap.release()
        
        # Asegurar que tenemos exactamente max_frames
        while len(frames) < self.max_frames:
            frames.append(torch.zeros(1, self.img_size, self.img_size))
        
        return torch.stack(frames[:self.max_frames])  # (T, 1, H, W)
    
    def _load_csv(self, csv_path, max_len):
        """Carga datos de un CSV como tensor."""
        if csv_path is None or not os.path.exists(csv_path):
            return None
        
        try:
            df = pd.read_csv(csv_path)
            # Seleccionar solo columnas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            data = df[numeric_cols].values.astype(np.float32)
            
            # Reemplazar NaN con 0
            data = np.nan_to_num(data, 0.0)
            
            # Truncar o pad
            if len(data) > max_len:
                indices = np.linspace(0, len(data) - 1, max_len, dtype=int)
                data = data[indices]
            elif len(data) < max_len:
                pad = np.zeros((max_len - len(data), data.shape[1]))
                data = np.vstack([data, pad])
            
            return torch.FloatTensor(data)
        except Exception:
            return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Cargar cada modalidad
        video = self._load_video(sample['video'])
        imu = self._load_csv(sample['imu'], self.max_frames)
        landmarks = self._load_csv(sample['landmarks'], self.max_frames)
        
        label = sample['label']
        
        return video, imu, landmarks, label


def multimodal_collate(batch):
    """Collate personalizado que maneja datos faltantes."""
    videos, imus, landmarks, labels = zip(*batch)
    
    videos = torch.stack(videos)
    labels = torch.LongTensor(labels)
    
    # IMU y landmarks pueden ser None si no existen
    has_imu = all(x is not None for x in imus)
    has_landmarks = all(x is not None for x in landmarks)
    
    imu_batch = torch.stack(imus) if has_imu else None
    landmarks_batch = torch.stack(landmarks) if has_landmarks else None
    
    return videos, imu_batch, landmarks_batch, labels

# ─────────────────────────────────────────────
# PASO 3: MODELO MULTIMODAL (FUSIÓN TARDÍA)
# ─────────────────────────────────────────────

class VideoEncoder(nn.Module):
    """Procesa video: CNN por frame + LSTM temporal."""
    
    def __init__(self, output_dim=128):
        super().__init__()
        # CNN simple para extraer features de cada frame
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),  # Reduce a 4×4 sin importar el tamaño original
            nn.Flatten()  # 64 × 4 × 4 = 1024
        )
        # LSTM para capturar movimiento entre frames
        self.lstm = nn.LSTM(1024, output_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(output_dim * 2, output_dim)
    
    def forward(self, x):
        # x shape: (batch, T, 1, H, W)
        batch_size, T = x.shape[0], x.shape[1]
        
        # Procesar cada frame con la CNN
        x = x.view(batch_size * T, *x.shape[2:])  # (batch*T, 1, H, W)
        x = self.cnn(x)                             # (batch*T, 1024)
        x = x.view(batch_size, T, -1)               # (batch, T, 1024)
        
        # Procesar secuencia temporal con LSTM
        output, (hidden, _) = self.lstm(x)
        combined = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(combined)  # (batch, output_dim)


class SequenceEncoder(nn.Module):
    """Procesa datos secuenciales (IMU o landmarks)."""
    
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, output_dim, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(output_dim * 2, output_dim)
    
    def forward(self, x):
        x = self.norm(x)
        output, (hidden, _) = self.lstm(x)
        combined = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(combined)


class MultimodalFusion(nn.Module):
    """
    Modelo de fusión tardía.
    Cada modalidad tiene su propio encoder.
    Al final, se concatenan y pasan por un clasificador conjunto.
    """
    
    def __init__(self, num_classes, imu_dim=6, landmark_dim=99):
        super().__init__()
        
        self.video_encoder = VideoEncoder(output_dim=128)
        self.imu_encoder = SequenceEncoder(input_dim=imu_dim, output_dim=128)
        self.landmark_encoder = SequenceEncoder(input_dim=landmark_dim, output_dim=128)
        
        # Clasificador que combina las 3 representaciones
        # 128 × 3 = 384 si todas las modalidades están presentes
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Clasificadores individuales (para cuando faltan modalidades)
        self.video_classifier = nn.Linear(128, num_classes)
    
    def forward(self, video, imu=None, landmarks=None):
        # Siempre procesamos video
        video_feat = self.video_encoder(video)   # (batch, 128)
        
        # Procesar modalidades disponibles
        if imu is not None and landmarks is not None:
            imu_feat = self.imu_encoder(imu)              # (batch, 128)
            landmark_feat = self.landmark_encoder(landmarks)  # (batch, 128)
            combined = torch.cat([video_feat, imu_feat, landmark_feat], dim=1)
            return self.classifier(combined)
        else:
            # Fallback: solo video
            return self.video_classifier(video_feat)


# ─────────────────────────────────────────────
# PASO 4: CARGAR DATOS Y CREAR MODELO
# ─────────────────────────────────────────────

dataset = LSC50MultimodalDataset(DATASET_PATH, max_frames=MAX_FRAMES, img_size=IMG_SIZE)

val_size = int(len(dataset) * 0.2)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, collate_fn=multimodal_collate)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                       shuffle=False, collate_fn=multimodal_collate)

# Determinar dimensiones de IMU y landmarks del primer batch
sample = dataset[0]
imu_dim = sample[1].shape[1] if sample[1] is not None else 6
landmark_dim = sample[2].shape[1] if sample[2] is not None else 99

model_multi = MultimodalFusion(
    num_classes=dataset.num_classes,
    imu_dim=imu_dim,
    landmark_dim=landmark_dim
).to(device)

print(f"\nModelo Multimodal:")
total_params = sum(p.numel() for p in model_multi.parameters())
print(f"Total de parámetros: {total_params:,}")

# ─────────────────────────────────────────────
# PASO 5: ENTRENAMIENTO MULTIMODAL
# ─────────────────────────────────────────────

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_multi.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0.0

print("\n" + "="*60)
print("INICIANDO ENTRENAMIENTO MULTIMODAL")
print("="*60)

for epoch in range(EPOCHS):
    # --- ENTRENAMIENTO ---
    model_multi.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for videos, imus, landmarks, labels in train_loader:
        videos = videos.to(device)
        labels = labels.to(device)
        imus = imus.to(device) if imus is not None else None
        landmarks = landmarks.to(device) if landmarks is not None else None
        
        outputs = model_multi(videos, imus, landmarks)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_multi.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    # --- VALIDACIÓN ---
    model_multi.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    
    with torch.no_grad():
        for videos, imus, landmarks, labels in val_loader:
            videos = videos.to(device)
            labels = labels.to(device)
            imus = imus.to(device) if imus is not None else None
            landmarks = landmarks.to(device) if landmarks is not None else None
            
            outputs = model_multi(videos, imus, landmarks)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    scheduler.step(val_loss)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_multi.state_dict(), "mejor_modelo_fase3.pth")
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Época [{epoch+1:3d}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%")

print(f"\n✅ Mejor accuracy de validación: {best_val_acc:.1f}%")
```

## 3.5 Errores Comunes — Fase 3

| Error | Causa | Solución |
|-------|-------|----------|
| `cv2` no encontrado | OpenCV no instalado | `pip install opencv-python` |
| Video no carga frames | Codec no soportado | Instalar `ffmpeg`: `sudo apt install ffmpeg` |
| OOM (out of memory) | Videos muy grandes en memoria | Reducir `MAX_FRAMES` a 15 o `IMG_SIZE` a 48 |
| Dimensiones no coinciden | IMU/landmarks tienen columnas diferentes | Imprimir `df.shape` y ajustar `imu_dim`/`landmark_dim` |
| Accuracy peor que Fase 1 | Pocas muestras (5 participantes) | Aplicar más data augmentation, o usar solo 2 modalidades |

---

# 🗺️ Mapa del Proyecto Completo

```
FASE 1 (Semanas 1-2)          FASE 2 (Semanas 3-4)         FASE 3 (Semanas 5-8)
─────────────────             ─────────────────            ─────────────────
LSC70 (imágenes)              LSC-54 (landmarks)           LSC50 (multimodal)
CNN simple                    LSTM bidireccional           CNN + LSTM + fusión
47 clases                     54 clases                    50 clases
~35K imágenes                 ~22 participantes            Video + IMU + landmarks
                                                           
Meta: >70% accuracy           Meta: >60% accuracy          Meta: >50% accuracy
```

> Las metas de accuracy son orientativas. Lo importante es que las curvas de aprendizaje muestren mejora real y que entiendas cada paso antes de avanzar al siguiente.

---

# 📌 Consejos Finales

1. **No saltes fases.** Asegúrate de que la Fase 1 funcione bien antes de pasar a la 2.
2. **Visualiza siempre.** Si no puedes ver las gráficas de aprendizaje, no sabes si funciona.
3. **Empieza pequeño.** Prueba con 5 clases primero, luego escala a todas.
4. **Guarda tus modelos.** Usa `torch.save()` después de cada experimento exitoso.
5. **Documenta resultados.** Anota qué hiperparámetros usaste y qué accuracy obtuviste.
6. **Adapta el código.** La estructura exacta de cada dataset puede variar — siempre inspecciona los archivos antes de codificar el loader.
