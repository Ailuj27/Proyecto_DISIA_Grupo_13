import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def make_gradcam_heatmap(img_array, model, target_layer_name="top_activation", pred_index=None):
    """
    Genera el heatmap Grad-CAM para un modelo cuya última parte es:
    ... -> top_conv -> top_bn -> top_activation -> flatten -> dropout -> dense -> dense_1 -> dense_2
    """
    target_layer = model.get_layer(target_layer_name)

    feature_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=target_layer.output
    )

    classifier_input = tf.keras.Input(shape=target_layer.output.shape[1:])
    x = classifier_input

    passed_target = False
    for layer in model.layers:
        if layer.name == target_layer_name:
            passed_target = True
            continue

        if passed_target:
            x = layer(x)

    classifier_model = tf.keras.Model(classifier_input, x)

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        feature_maps = feature_model(img_tensor, training=False)
        tape.watch(feature_maps)

        preds = classifier_model(feature_maps, training=False)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])

        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, feature_maps)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    feature_maps = feature_maps[0]

    heatmap = feature_maps @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap_on_image(original_pil_image: Image.Image, heatmap, alpha=0.4, output_size=(256, 256)):
    """
    Superpone el heatmap sobre la imagen original.
    """
    original = original_pil_image.resize(output_size)
    original_np = np.array(original)

    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_img = Image.fromarray(heatmap_uint8)
    heatmap_img = heatmap_img.resize(output_size, resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img)

    cmap = plt.get_cmap("jet")
    colored_heatmap = cmap(heatmap_resized / 255.0)[:, :, :3]
    colored_heatmap = np.uint8(colored_heatmap * 255)

    superimposed = np.uint8((1 - alpha) * original_np + alpha * colored_heatmap)
    return Image.fromarray(superimposed)


def save_pil_image_as_png(pil_image: Image.Image, output_path: str) -> str:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(output_file, format="PNG")
    return str(output_file)