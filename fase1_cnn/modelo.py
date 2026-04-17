import torch.nn as nn


class LSC_CNN(nn.Module):
    """
    CNN 3 bloques conv + AdaptiveAvgPool + clasificador ligero.
    Entrada: imagen RGB (img_size x img_size).
    Salida: logits por clase.

    AdaptiveAvgPool2d(4) fija el flatten a 256*16 = 4 096 independientemente
    del img_size, reduciendo parametros de ~33 M a ~1.1 M y el riesgo de
    sobreajuste con datasets pequenos.
    """

    def __init__(self, num_classes: int, img_size: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            # Bloque 1 — 3 -> 64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bloque 2 — 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bloque 3 — 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Pool adaptativo -> (256, 4, 4) = 4 096 independiente de img_size
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def crear_modelo(num_classes: int, img_size: int = 128, device=None):
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSC_CNN(num_classes=num_classes, img_size=img_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Modelo CNN: {total_params:,} parametros  |  Dispositivo: {device}")
    return model
