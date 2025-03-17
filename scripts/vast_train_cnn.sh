#!/usr/bin/env bash

dvc pull --with-deps train_cnn
dvc repro --force --single-item train_cnn
