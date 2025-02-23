"""
Module for evaluating a transfer learning trained ResNet50 model.
"""

import argparse
import tensorflow as tf
import keras
import dvc.api
import dvclive

# pylint: disable=unnecessary-lambda-assignment
vprint = lambda *a, **k: None

def evaluate(model, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset):
    """
    Evaluates a trained model.
    """
    vprint("Evaluating model...")

    train_accuracy = model.evaluate(train_dataset, verbose=2)
    test_accuracy = model.evaluate(test_dataset, verbose=2)

    with dvclive.Live("dvclive/resnet50/evaluation") as live:
        live.log_metric("train/accuracy", train_accuracy[1])
        live.log_metric("test/accuracy", test_accuracy[1])

def main():
    """
    Entry point for evaluating a trained model.
    """
    parser = argparse.ArgumentParser(prog="evaluate_resnet50.py",
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    description="Evaluate a transfer learning trained ResNet50 model.")

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

    vprint(f"Loading model from '{args.model}'...")

    params = dvc.api.params_show()
    batch_size = params["general"]["batch_size"]

    model = keras.saving.load_model(args.model)

    train_dataset = (tf.data.Dataset.load(args.train)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))

    test_dataset = (tf.data.Dataset.load(args.test)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))

    evaluate(model, train_dataset, test_dataset)

    vprint("Done.")

if __name__ == "__main__":
    main()
