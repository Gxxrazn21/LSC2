import torch
import torch.nn as nn
from torchvision import models


class LSC_CNN(nn.Module):
    """
    Backbone MobileNetV3-Small preentrenado en ImageNet.
    Se congela el feature extractor y se reemplaza el clasificador
    para las N clases de LSC.

    Con ~400 imágenes por clase, transfer learning es esencial.
    """

    def __init__(self, num_classes: int, img_size: int = 224, freeze_backbone: bool = True):
        super().__init__()

        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            # Congelar todo excepto el último bloque convolucional
            for name, param in base.features.named_parameters():
                block_idx = name.split(".")[0] if name[0].isdigit() else None
                if block_idx is not None and int(block_idx) < 10:
                    param.requires_grad = False

        self.features = base.features
        self.avgpool = base.avgpool

        in_features = base.classifier[0].in_features  # 576
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def crear_modelo(num_classes: int, img_size: int = 224, device=None, freeze_backbone: bool = True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSC_CNN(num_classes=num_classes, img_size=img_size, freeze_backbone=freeze_backbone).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Modelo MobileNetV3-Small: {total:,} params totales | {trainable:,} entrenables | {device}")
    return model
