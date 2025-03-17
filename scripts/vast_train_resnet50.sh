#!/usr/bin/env bash

dvc pull --with-deps train_resnet50
dvc repro --force --single-item train_resnet50
