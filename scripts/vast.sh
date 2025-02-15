#!/usr/bin/env bash

git config --local user.name "Constantin Müller"
git config --local user.email "info@mueller-constantin.de"

pip install -r requirements.txt

dvc pull
