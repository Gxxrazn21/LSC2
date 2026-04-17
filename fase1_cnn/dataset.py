import glob
import os

from PIL import Image
from torch.utils.data import Dataset


class LSCDataset(Dataset):
    """
    Dataset LSC70_HAND_CROPS.
    Estructura: base_path/LSC70W/{Per##}/{clase}/*.jpg

    Args:
        base_path:        Raíz del dataset (contiene LSC70W/).
        transform:        Transformaciones torchvision.
        target_gestures:  Lista de clases a cargar (ej. ["HOLA","BUENAS"]).
        person_filter:    Lista de IDs de persona a incluir (ej. ["Per01","Per02"]).
                          None = todas las personas.
    """

    def __init__(
        self,
        base_path: str,
        transform=None,
        target_gestures: list = None,
        person_filter: list = None,
    ):
        self.transform = transform
        self.samples = []

        if target_gestures:
            self.classes = sorted(target_gestures)
        else:
            found = glob.glob(
                os.path.join(base_path, "LSC70W", "Per*", "*"),
                recursive=False,
            )
            self.classes = sorted({
                os.path.basename(p) for p in found if os.path.isdir(p)
            })

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        person_set = set(person_filter) if person_filter else None

        for cls_name in self.classes:
            pattern = os.path.join(base_path, "LSC70W", "Per*", cls_name, "*.jpg")
            for img_path in glob.glob(pattern):
                # img_path: .../LSC70W/Per03/HOLA/Per03_HOLA_0.jpg
                person_id = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
                if person_set is not None and person_id not in person_set:
                    continue
                self.samples.append((img_path, self.class_to_idx[cls_name]))

        self.samples.sort(key=lambda x: x[0])

        counts = {c: 0 for c in self.classes}
        for _, idx in self.samples:
            counts[self.classes[idx]] += 1
        persons_desc = f"personas={len(person_filter)}" if person_filter else "todas las personas"
        print(f"  Dataset ({persons_desc}): {len(self.samples)} imgs | clases: {dict(counts)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), 0)
        if self.transform:
            image = self.transform(image)
        return image, label
