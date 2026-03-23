#!/usr/bin/env bash

dvc pull data/preprocessed/train/resnet50
dvc pull data/preprocessed/validation/resnet50
dvc repro --force --single-item train_cnn
