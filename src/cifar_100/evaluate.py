"""
Module for evaluating a trained model.
"""

import os
import argparse
import tensorflow as tf
import keras
import dvc.api
import dvclive

# pylint: disable=unnecessary-lambda-assignment
vprint = lambda *a, **k: None

def evaluate(model_name: str, model: keras.Sequential, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset):
    """
    Evaluates a trained model.
    """
    vprint("Evaluating model ...")

    train_accuracy = model.evaluate(train_dataset, verbose=2)
    test_accuracy = model.evaluate(test_dataset, verbose=2)

    with dvclive.Live() as live:
        live.log_metric(f"{model_name}/train/accuracy", train_accuracy[1])
        live.log_metric(f"{model_name}/test/accuracy", test_accuracy[1])

def main():
    """
    Entry point for evaluating a trained model.
    """
    parser = argparse.ArgumentParser(prog="evaluate.py",
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    description="Evaluate a trained model.")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print out verbose messages.")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Path to the model to evaluate. Must be a Keras model file.")
    parser.add_argument("-t", "--train", type=str, required=True,
                help="Path to the train dataset to evaluate. Must be a directory with Tensorflow shards.")
    parser.add_argument("-e", "--test", type=str, required=True,
                help="Path of test dataset to evaluate. Must be a directory with Tensorflow shards.")

    args = parser.parse_args()

    if args.verbose:
        global vprint
        vprint = print

    vprint(f"Loading model from '{args.model}' ...")

    params = dvc.api.params_show()
    batch_size = params["evaluate_cnn"]["batch_size"]

    model_name = os.path.splitext(os.path.basename(args.model))[0]
    model = keras.saving.load_model(args.model)

    train_dataset = (tf.data.Dataset.load(args.train)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))

    test_dataset = (tf.data.Dataset.load(args.test)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))

    evaluate(model_name, model, train_dataset, test_dataset)

    vprint("Done.")

if __name__ == "__main__":
    main()
