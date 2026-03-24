#!/usr/bin/env bash

dvc pull data/preprocessed/train/mobilenet
dvc pull data/preprocessed/validation/mobilenet
dvc repro --force --single-item train_mobilenet
