"""
Module for training a tranfer learning trained ResNet50 model.
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

    x = keras.layers.Input(shape=(32, 32, 3))

    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")

    x = data_augmentation(x)

    base_model = keras.applications.resnet50.ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=x
    )

    base_model.trainable = False

    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(100, activation="softmax", kernel_regularizer=keras.regularizers.l2(0.01))(x)

    model = keras.Model(inputs=base_model.input, outputs=x)

    return model, base_model

def train_model(train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset):
    """
    Trains the deep learning model.
    """
    model, base_model = create_model()

    params = dvc.api.params_show()
    epochs = params["train_resnet50"]["epochs"]
    feature_extraction_learning_rate = params["train_resnet50"]["feature_extraction_learning_rate"]
    fine_tuning_learning_rate = params["train_resnet50"]["fine_tuning_learning_rate"]
    fine_tuning_unfreezed_layer_count = params["train_resnet50"]["fine_tuning_unfreezed_layer_count"]

    vprint(f"Training model for {epochs} epochs...")

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # Feature extraction

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=feature_extraction_learning_rate, momentum=0.9),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    with dvclive.Live("dvclive/resnet50/training/feature-extraction") as live:
        dvclive_callback = DVCLiveCallback(live=live)

        history = model.fit(train_dataset,
                            epochs=epochs,
                            validation_data=validation_dataset,
                            callbacks=[dvclive_callback, early_stopping_callback, reduce_lr_callback])

    # Fine-tuning

    base_model.trainable = False

    for layer in base_model.layers[-fine_tuning_unfreezed_layer_count:]:
        layer.trainable = True

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=fine_tuning_learning_rate, momentum=0.9),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    with dvclive.Live("dvclive/resnet50/training/fine-tuning") as live:
        dvclive_callback = DVCLiveCallback(live=live)

        history = model.fit(train_dataset,
                            epochs=epochs,
                            validation_data=validation_dataset,
                            callbacks=[dvclive_callback, early_stopping_callback])

    return model, history

def main():
    parser = argparse.ArgumentParser(prog="train_resnet50.py",
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    description="Trains a tranfer learning trained ResNet50 model.")

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

    model.save(os.path.join(args.output, "resnet50_model.keras"))

    vprint("Saving history...")

    np.save(os.path.join(args.output, "resnet50_history.npy"), history.history)

if __name__ == "__main__":
    main()
