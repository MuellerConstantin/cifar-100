"""
Setup file for the project source code package.
"""

from setuptools import setup, find_packages

setup(
  name="cifar-100",
  version="0.1.0",
  packages=find_packages(include=["cifar_100", "cifar_100.*"]),
  install_requires=[
    "numpy",
    "tqdm",
    "keras==3.5.0",
    "tensorflow==2.16.1",
    "dvc",
    "dvclive"
  ],
  entry_points={
    "console_scripts": [
      "fetch_data=cifar_100.fetch_data:main",
      "split=cifar_100.split:main",
      "preprocess_cnn=cifar_100.dl.preprocess_cnn:main",
      "train_cnn=cifar_100.dl.train_cnn:main",
      "evaluate_cnn=cifar_100.dl.evaluate_cnn:main",
      "preprocess_resnet50=cifar_100.dl.preprocess_resnet50:main",
      "train_resnet50=cifar_100.dl.train_resnet50:main",
      "evaluate_resnet50=cifar_100.dl.evaluate_resnet50:main"
    ]
  }
)
