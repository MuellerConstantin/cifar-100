"""
Module for splitting the data into train and validation sets.
"""

import argparse
import tensorflow as tf
import dvc.api

# pylint: disable=unnecessary-lambda-assignment
vprint = lambda *a, **k: None

def split(full_dataset: tf.data.Dataset):
    """
    Splits the training data into train and validation sets.
    """
    vprint("Splitting data ...")

    params = dvc.api.params_show()
    validation_split_ratio = params["split"]["validation_split_ratio"]

    total_size = full_dataset.cardinality().numpy()
    validation_size = int(validation_split_ratio * total_size)

    validation_dataset = full_dataset.take(validation_size)
    train_dataset = full_dataset.skip(validation_size)

    return train_dataset, validation_dataset

def main():
    """
    Entry point for splitting the data into train and validation sets.
    """
    parser = argparse.ArgumentParser(prog="split.py",
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    description="Split data into train and test/validation sets.")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print out verbose messages.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input path to full data. Must be a directory with Tensroflow shards.")
    parser.add_argument("-t", "--train", type=str, required=True,
                        help="Path of train output directory. Will be saved as Tensorflow shards.")
    parser.add_argument("-e", "--validation", type=str, required=True,
                help="Path of validation output directory. Will be saved as Tensorflow shards.")

    args = parser.parse_args()

    if args.verbose:
        global vprint
        vprint = print

    vprint(f"Loading data from '{args.input}' ...")

    full_dataset = tf.data.Dataset.load(args.input)
    train_dataset, validation_dataset = split(full_dataset)

    vprint("Saving data...")

    train_dataset.save(args.train)
    validation_dataset.save(args.validation)

    vprint("Done.")

if __name__ == "__main__":
    main()
