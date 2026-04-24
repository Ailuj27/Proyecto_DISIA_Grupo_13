import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

from preprocess import save_class_names


IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
RANDOM_STATE = 42


def build_validation_dataset(val_dir: Path, batch_size: int = BATCH_SIZE):
    """
    Dataset de validación con image_dataset_from_directory.
    No aplicamos reescalado manual para mantener consistencia
    con EfficientNetV2 y el notebook original.
    """
    ds_validation = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False
    )

    ds_validation = ds_validation.map(
        lambda x, y: (tf.cast(x, tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(1)

    return ds_validation


def build_train_generator(train_dir: Path, batch_size: int = BATCH_SIZE, seed: int = RANDOM_STATE):
    """
    Generador de entrenamiento fiel al notebook.
    """
    train_datagen = ImageDataGenerator(
        rotation_range=0.1,
        zoom_range=0.2,
        fill_mode="constant",
        cval=0
    )

    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        batch_size=batch_size,
        target_size=IMAGE_SIZE,
        class_mode="sparse",
        seed=seed
    )

    return train_generator


def compute_mapped_class_weights_from_generator(train_generator):
    """
    Replica la idea de mapped_class_weights del notebook.
    """
    classes = train_generator.classes
    unique_classes = np.unique(classes)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=classes
    )

    mapped_class_weights = {
        int(cls): float(weight)
        for cls, weight in zip(unique_classes, class_weights)
    }

    return mapped_class_weights


def build_model(num_classes: int):
    """
    Construcción del modelo fiel al notebook.
    """
    include_top = False
    weights = "imagenet"
    input_shape = (256, 256, 3)

    model_transf = tf.keras.applications.EfficientNetV2M(
        include_top=include_top,
        weights=weights,
        input_shape=input_shape
    )

    fine_tuning = 446
    for layer in model_transf.layers[:fine_tuning]:
        layer.trainable = False

    # Igual que en el notebook
    model_transf.layers[0].trainable = True

    x = model_transf.output
    x = keras.layers.Flatten(data_format="channels_last")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(model_transf.input, x)

    lr = 0.001
    loss = "sparse_categorical_crossentropy"
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    metrics = ["accuracy"]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


def main(args):
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    output_model = Path(args.output_model)
    output_classes = Path(args.output_classes)
    output_history = Path(args.output_history)
    output_log = Path(args.output_log)

    if not train_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de entrenamiento: {train_dir}")

    if not val_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de validación: {val_dir}")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_classes.parent.mkdir(parents=True, exist_ok=True)
    output_history.parent.mkdir(parents=True, exist_ok=True)
    output_log.parent.mkdir(parents=True, exist_ok=True)

    print("Construyendo generador de entrenamiento...")
    train_generator = build_train_generator(
        train_dir=train_dir,
        batch_size=args.batch_size,
        seed=args.seed
    )

    print("Construyendo dataset de validación...")
    ds_validation = build_validation_dataset(
        val_dir=val_dir,
        batch_size=args.batch_size
    )

    class_names = list(train_generator.class_indices.keys())
    save_class_names(class_names, str(output_classes))
    print(f"Clases detectadas: {class_names}")
    print(f"Clases guardadas en: {output_classes}")

    mapped_class_weights = compute_mapped_class_weights_from_generator(train_generator)
    print(f"Class weights: {mapped_class_weights}")

    print("Construyendo modelo EfficientNetV2M...")
    model = build_model(num_classes=len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_model),
            save_best_only=True,
            monitor="val_loss"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger(str(output_log), append=True)
    ]

    print("Comenzando entrenamiento...")
    history = model.fit(
        x=train_generator,
        validation_data=ds_validation,
        callbacks=callbacks,
        epochs=args.epochs,
        class_weight=mapped_class_weights
    )

    # Guardado final, igual que en el notebook
    model.save(str(output_model))

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(output_history, index=False)

    print(f"Modelo guardado en: {output_model}")
    print(f"Historial guardado en: {output_history}")
    print(f"Log CSV guardado en: {output_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, default="/app/data/train")
    parser.add_argument("--val_dir", type=str, default="/app/data/val")
    parser.add_argument("--output_model", type=str, default="/app/models/transf_model.keras")
    parser.add_argument("--output_classes", type=str, default="/app/models/class_names.json")
    parser.add_argument("--output_history", type=str, default="/app/models/training_history_tf.csv")
    parser.add_argument("--output_log", type=str, default="/app/models/training_log_tf.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)