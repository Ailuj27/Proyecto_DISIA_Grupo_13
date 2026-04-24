from io import BytesIO
from pathlib import Path
import json
import numpy as np
from PIL import Image


IMAGE_SIZE = (256, 256)


def load_pil_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Carga una imagen desde bytes y la convierte a RGB.
    """
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def load_pil_image_from_path(image_path: str) -> Image.Image:
    """
    Carga una imagen desde disco y la convierte a RGB.
    """
    return Image.open(image_path).convert("RGB")


def prepare_image_for_model(pil_image: Image.Image, image_size=IMAGE_SIZE) -> np.ndarray:
    image = pil_image.resize(image_size)
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def save_class_names(class_names, output_path: str):
    """
    Guarda la lista de clases en JSON.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4, ensure_ascii=False)


def load_class_names(classes_path: str):
    """
    Carga la lista de clases desde JSON.
    """
    with open(classes_path, "r", encoding="utf-8") as f:
        return json.load(f)