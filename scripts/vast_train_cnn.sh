#!/usr/bin/env bash

dvc pull data/preprocessed/train/cnn
dvc pull data/preprocessed/validation/cnn
dvc repro --force --single-item train_cnn
