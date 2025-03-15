"""
Module for preprocessing data for a tranfer learning trained ResNet50 model.
"""

import argparse
import tensorflow as tf
import keras

# pylint: disable=unnecessary-lambda-assignment
vprint = lambda *a, **k: None

def preprocess(dataset: tf.data.Dataset):
    """
    Preprocesses data for a tranfer learning trained ResNet50 model.

    This function expects a dataset of images and labels, and returns a
    preprocessed dataset of images and labels.
    """
    vprint("Preprocessing data...")

    dataset = dataset.map(lambda x, y: (tf.image.resize(x, (32, 32)), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x, y: (keras.applications.resnet50.preprocess_input(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)

    return dataset

def main():
    """
    Entry point for preprocessing data for a tranfer learning trained ResNet50 model.
    """
    parser = argparse.ArgumentParser(prog="preprocess_resnet50.py",
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    description="Preprocesses data for a tranfer learning trained ResNet50 model.")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print out verbose messages.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the dataset to preprocess. Must be a directory with Tensorflow shards.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path of output directory. Will be saved as Tensorflow shards.")

    args = parser.parse_args()

    if args.verbose:
        global vprint
        vprint = print

    vprint(f"Loading data from '{args.input}'...")

    dataset = tf.data.Dataset.load(args.input)

    preprocessed_dataset = preprocess(dataset)

    vprint(f"Saving data to '{args.output}'...")

    preprocessed_dataset.save(args.output)

    vprint("Done.")

if __name__ == "__main__":
    main()
