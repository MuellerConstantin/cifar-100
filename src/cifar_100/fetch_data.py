"""
Module for fetching data for modeling.
"""

import os
import argparse
import tensorflow as tf
import keras

# pylint: disable=unnecessary-lambda-assignment
vprint = lambda *a, **k: None

def fetch():
    """
    Download data via keras datasets.
    """
    vprint("Downloading data ...")

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar100.load_data(
        label_mode="fine")

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    return train_dataset, test_dataset

def main():
    """
    Entry point for fetching data for modeling.
    """
    parser = argparse.ArgumentParser(prog="fetch_data.py",
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    description="Fetch data for modeling.")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print out verbose messages.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path of output directory. Will be saved as Tensorflow shards.")

    args = parser.parse_args()

    if args.verbose:
        global vprint
        vprint = print

    vprint("Loading data from server")

    train_dataset, test_dataset = fetch()

    vprint(f"Saving data to '{args.output}' ...")

    os.makedirs(args.output, exist_ok=True)

    train_dataset.save(os.path.join(args.output, "train"))
    test_dataset.save(os.path.join(args.output, "test"))

    vprint("Done.")

if __name__ == "__main__":
    main()
