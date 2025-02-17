"""
Module for training a simple neural network regression model.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import keras
import dvc.api

# pylint: disable=unnecessary-lambda-assignment
vprint = lambda *a, **k: None

def create_model():
    """
    Creates the deep learning model.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=((32, 32, 3))),
        keras.layers.Rescaling(1./255),
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.2),
        keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(100, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

def train_model(train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset):
    """
    Trains the deep learning model.
    """

    vprint("Building model...")

    model = create_model()

    vprint("Training model...")

    params = dvc.api.params_show()
    epochs = params["train_cnn"]["epochs"]

    history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset)

    return model, history

def main():
    parser = argparse.ArgumentParser(prog="train_cnn.py",
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    description="Trains a convolutional neural network model.")

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
    batch_size = params["train_cnn"]["batch_size"]

    vprint("Loading training and validation data...")

    train_dataset = (tf.data.Dataset.load(args.train)
                     .batch(batch_size)
                     .prefetch(tf.data.AUTOTUNE))

    validation_dataset = (tf.data.Dataset.load(args.validation)
                          .batch(batch_size)
                          .prefetch(tf.data.AUTOTUNE))

    model, history = train_model(train_dataset, validation_dataset)

    vprint("Saving model...")

    model.save(os.path.join(args.output, "cnn_model.keras"))

    vprint("Saving history...")

    np.save(os.path.join(args.output, "cnn_history.npy"), history.history)

if __name__ == "__main__":
    main()
