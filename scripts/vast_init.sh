#!/usr/bin/env bash

apt-get update && apt-get install -y git
apt-get update && apt-get install -y nano

git config --local user.name "Constantin Müller"
git config --local user.email "info@mueller-constantin.de"

pip install -e .[dev]
