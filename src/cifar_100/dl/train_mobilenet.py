"""
Module for training a tranfer learning trained MobileNet model.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import keras
import dvc.api
from dvclive.keras import DVCLiveCallback
import dvclive

# pylint: disable=unnecessary-lambda-assignment
vprint = lambda *a, **k: None

def create_model():
    """
    Creates the deep learning model.
    """
    vprint("Building model...")

    inputs = keras.layers.Input(shape=(32, 32, 3))

    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomContrast(0.1)
    ], name="data_augmentation")

    x = data_augmentation(inputs)
    x = keras.layers.Resizing(96, 96)(x)

    base_model = keras.applications.mobilenet_v2.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=x,
        alpha=0.75
    )

    base_model.trainable = False

    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(100, activation="softmax", kernel_regularizer=keras.regularizers.l2(1e-4))(x)

    model = keras.Model(inputs=inputs, outputs=x)

    return model, base_model

def train_model(train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset):
    """
    Trains the deep learning model.
    """
    model, base_model = create_model()

    params = dvc.api.params_show()
    feature_extraction_epochs = params["train_mobilenet"]["feature_extraction_epochs"]
    feature_extraction_learning_rate = params["train_mobilenet"]["feature_extraction_learning_rate"]
    fine_tuning_epochs = params["train_mobilenet"]["fine_tuning_epochs"]
    fine_tuning_learning_rate = params["train_mobilenet"]["fine_tuning_learning_rate"]
    fine_tuning_unfreezed_layer_count = params["train_mobilenet"]["fine_tuning_unfreezed_layer_count"]

    feature_extraction_early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    fine_tuning_early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    fine_tuning_reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    vprint(f"Training model for {feature_extraction_epochs} feature extraction epochs...")

    # Feature extraction

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=feature_extraction_learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    with dvclive.Live("dvclive/mobilenet/training/feature-extraction") as live:
        dvclive_callback = DVCLiveCallback(live=live)

        history = model.fit(train_dataset,
                            epochs=feature_extraction_epochs,
                            validation_data=validation_dataset,
                            callbacks=[dvclive_callback, feature_extraction_early_stopping])

    vprint(f"Training model for {fine_tuning_epochs} fine-tuning epochs...")

    # Fine-tuning

    base_model.trainable = True

    for layer in base_model.layers[:-fine_tuning_unfreezed_layer_count]:
        layer.trainable = False

    for layer in base_model.layers[-fine_tuning_unfreezed_layer_count:]:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=fine_tuning_learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    with dvclive.Live("dvclive/mobilenet/training/fine-tuning") as live:
        dvclive_callback = DVCLiveCallback(live=live)

        history = model.fit(train_dataset,
                            epochs=fine_tuning_epochs,
                            validation_data=validation_dataset,
                            callbacks=[dvclive_callback, fine_tuning_early_stopping, fine_tuning_reduce_lr])

    return model, history

def main():
    parser = argparse.ArgumentParser(prog="train_mobilenet.py",
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    description="Trains a tranfer learning trained MobileNet model.")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print out verbose messages.")
    parser.add_argument("-t", "--train", type=str, required=True,
                        help="Path of train input directory. Will be saved as Tensorflow shards.")
    parser.add_argument("-e", "--validation", type=str, required=True,
                help="Path of validation input directory. Will be saved as Tensorflow shards.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output file path. Must a directory.")

    args = parser.parse_args()

    if args.verbose:
        global vprint
        vprint = print

    params = dvc.api.params_show()
    batch_size = params["general"]["batch_size"]

    vprint("Loading training and validation data...")

    train_dataset = (tf.data.Dataset.load(args.train)
                     .batch(batch_size)
                     .prefetch(tf.data.AUTOTUNE))

    validation_dataset = (tf.data.Dataset.load(args.validation)
                          .batch(batch_size)
                          .prefetch(tf.data.AUTOTUNE))

    model, history = train_model(train_dataset, validation_dataset)

    vprint("Saving model...")

    model.save(os.path.join(args.output, "mobilenet_model.keras"))

    vprint("Saving history...")

    np.save(os.path.join(args.output, "mobilenet_history.npy"), history.history)

if __name__ == "__main__":
    main()
