# CIFAR-100 - Image Classification

> Data analysis about the famous [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

## Table of contents

- [Introduction](#introduction)
- [Development](#development)
- [Remote Training](#remote-training)
- [License](#license)
  - [Forbidden](#forbidden)

## Introduction

This is a data analysis project about the famous [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The dataset itself is obtained from [University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html).

## Development

The actual models are trained remotely using [Vast.ai](https://vast.ai/) and the results are analyzed locally.
To enable this, the data pipeline described above exists. Model inference is performed locally, while
training takes place on high-performance machines at Vast.ai. For this purpose, the pipeline's source
code is exchanged via GitHub, and the data via DVC, as well as a Hetzner storage box.

![Dataflow](./docs/images/dataflow.svg)

## Remote Training

The training is performed on Vast.ai, which provides access to powerful GPUs and a flexible environment for machine learning tasks. The training scripts are executed remotely, and the results are stored and analyzed locally after the training is complete.

To start the training process, follow these steps:

1. Rent a machine on Vast.ai with the desired specifications (e.g., GPU type, RAM, storage). Tensorflow and
CUDA support is required for the training process.
2. Clone the repository and set up the environment on the rented machine using `git clone https://github.com/MuellerConstantin/cifar-100.git`.
3. Initialize the environment with all ist dependencies using `sh ./scripts/vast_init.sh`.
4. Configure the storage box credentials by setting the password, required for fetching training data and
pushing the results back to the storage box. Using this command: `dvc remote modify --local storagebox password <PASSWORD>`.
5. Start the training process by executing the training script `sh ./scripts/vast_train_<MODEL>.sh`.

## License

Copyright (c) 2025 Constantin Müller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

[MIT License](https://opensource.org/licenses/MIT) or [LICENSE](LICENSE) for
more details.

### Forbidden

**Hold Liable**: Software is provided without warranty and the software
author/license owner cannot be held liable for damages.
