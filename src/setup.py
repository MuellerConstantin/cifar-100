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
    "keras",
    "tensorflow"
  ],
  entry_points={
    "console_scripts": [
      "fetch_data=cifar_100.fetch_data:main",
      "split=cifar_100.split:main"
    ]
  }
)
