"""
Module for preprocessing data for a simple convolutional neural network model.
"""

import argparse
import tensorflow as tf

# pylint: disable=unnecessary-lambda-assignment
vprint = lambda *a, **k: None

def preprocess(dataset: tf.data.Dataset):
    """
    Preprocesses data for a simple convolutional neural network model.

    This function expects a dataset of images and labels, and returns a
    preprocessed dataset of images and labels.
    """
    vprint("Preprocessing data...")

    @tf.function
    def resize_and_rescale(x, y):
        x = tf.image.resize(x, (32, 32))
        x = x / 255.0

        return x, y

    dataset = dataset.map(resize_and_rescale, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset

def main():
    """
    Entry point for preprocessing data for a simple convolutional neural network model.
    """
    parser = argparse.ArgumentParser(prog="preprocess_cnn.py",
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    description="Preprocesses data for a simple convolutional neural network model.")

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
