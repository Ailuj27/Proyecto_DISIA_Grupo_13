from pathlib import Path
import uuid
import json
import tensorflow as tf

from preprocess import (
    load_pil_image_from_bytes,
    load_pil_image_from_path,
    prepare_image_for_model,
    load_class_names,
)
from gradcam import (
    make_gradcam_heatmap,
    overlay_heatmap_on_image,
    save_pil_image_as_png,
)


class TumorPredictor:
    def __init__(
        self,
        model_path: str = "/app/models/best_model.keras",
        classes_path: str = "/app/models/class_names.json",
        heatmaps_dir: str = "/app/generated/heatmaps",
    ):
        model_file = Path(model_path)
        classes_file = Path(classes_path)
        heatmaps_folder = Path(heatmaps_dir)

        if not model_file.exists():
            raise FileNotFoundError(f"No se encontró el modelo en: {model_file}")

        if not classes_file.exists():
            raise FileNotFoundError(f"No se encontró el archivo de clases en: {classes_file}")

        heatmaps_folder.mkdir(parents=True, exist_ok=True)

        self.model = tf.keras.models.load_model(model_file)
        self.class_names = load_class_names(str(classes_file))
        self.heatmaps_dir = heatmaps_folder
        self.gradcam_target_layer = "top_activation"

        print("Modelo cargado correctamente.")
        print(f"Clases cargadas: {self.class_names}")
        print(f"Capa Grad-CAM usada: {self.gradcam_target_layer}")

    def _predict_array(self, img_array):
        predictions = self.model.predict(img_array, verbose=0)[0]
        pred_idx = int(tf.argmax(predictions).numpy())
        pred_class = self.class_names[pred_idx]
        confidence = float(predictions[pred_idx])

        probabilities = {
            class_name: float(prob)
            for class_name, prob in zip(self.class_names, predictions)
        }

        return pred_idx, pred_class, confidence, probabilities

    def _generate_heatmap_file(self, pil_image, img_array, pred_idx, pred_class):
        heatmap = make_gradcam_heatmap(
            img_array=img_array,
            model=self.model,
            target_layer_name=self.gradcam_target_layer,
            pred_index=pred_idx,
        )

        overlay_image = overlay_heatmap_on_image(pil_image, heatmap)

        filename = f"heatmap_{pred_class}_{uuid.uuid4().hex}.png"
        output_path = self.heatmaps_dir / filename

        save_pil_image_as_png(overlay_image, str(output_path))
        return filename

    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        pil_image = load_pil_image_from_bytes(image_bytes)
        img_array = prepare_image_for_model(pil_image)

        pred_idx, pred_class, confidence, probabilities = self._predict_array(img_array)
        filename = self._generate_heatmap_file(pil_image, img_array, pred_idx, pred_class)

        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "heatmap_filename": filename,
            "gradcam_layer": self.gradcam_target_layer,
        }

    def predict_from_path(self, image_path: str) -> dict:
        pil_image = load_pil_image_from_path(image_path)
        img_array = prepare_image_for_model(pil_image)

        pred_idx, pred_class, confidence, probabilities = self._predict_array(img_array)
        filename = self._generate_heatmap_file(pil_image, img_array, pred_idx, pred_class)

        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "heatmap_filename": filename,
            "gradcam_layer": self.gradcam_target_layer,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="/app/models/best_model.keras")
    parser.add_argument("--classes_path", type=str, default="/app/models/class_names.json")
    parser.add_argument("--heatmaps_dir", type=str, default="/app/generated/heatmaps")
    args = parser.parse_args()

    predictor = TumorPredictor(
        model_path=args.model_path,
        classes_path=args.classes_path,
        heatmaps_dir=args.heatmaps_dir,
    )

    result = predictor.predict_from_path(args.image_path)
    print(json.dumps(result, indent=4, ensure_ascii=False))