import glob
import os

from PIL import Image
from torch.utils.data import Dataset


class LSCDataset(Dataset):
    """
    Dataset LSC70_HAND_CROPS.
    Estructura: base_path/**/{clase}/*.jpg

    Si se pasa target_gestures, solo se cargan esas clases.
    """

    def __init__(self, base_path: str, transform=None, target_gestures: list = None):
        self.transform = transform
        self.samples = []

        if target_gestures:
            self.classes = sorted(target_gestures)
        else:
            # Descubrir todas las clases presentes bajo LSC70W
            found = glob.glob(
                os.path.join(base_path, "**", "LSC70W", "Per*", "*"),
                recursive=True,
            )
            self.classes = sorted({
                os.path.basename(p) for p in found if os.path.isdir(p)
            })

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls_name in self.classes:
            pattern = os.path.join(base_path, "**", cls_name, "*.jpg")
            for img_path in glob.glob(pattern, recursive=True):
                self.samples.append((img_path, self.class_to_idx[cls_name]))

        # Orden determinista para split reproducible
        self.samples.sort(key=lambda x: x[0])

        counts = {c: 0 for c in self.classes}
        for _, idx in self.samples:
            counts[self.classes[idx]] += 1
        print(f"  Dataset: {len(self.samples)} imagenes  |  clases: {dict(counts)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Imagen corrupta: devolver imagen negra del mismo tamano
            image = Image.new("RGB", (128, 128), 0)
        if self.transform:
            image = self.transform(image)
        return image, label
