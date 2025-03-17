#!/usr/bin/env bash

dvc repro --pull --force --single-item --downstream train_cnn
