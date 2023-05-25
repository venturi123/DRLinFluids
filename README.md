# DRLinFluids
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/venturi123/DRLinFluids/blob/main/LICENSE)

Welcome to DRLinFluids!

DRLinFluids is a flexible Python package that enables the application of Deep Reinforcement Learning (DRL) techniques to Computational Fluid Dynamics (CFD). With this package, you can leverage the power of DRL to your CFD simulations and discover new insights into fluid dynamics.

## Introduction

Deep Reinforcement learning is a field of machine learning. It studies by interacting with the environment. It emphasizes how to make corresponding behavior in a specific environment in order to maximize the expected benefits. However, for reinforcement learning, it is necessary to define a specific interaction environment for specific problems, which is rather cumbersome and takes up a lot of time of researchers in related fields, and delays the research speed of researchers in reinforcement learning and fluid cross field. For this purpose, a reinforcement learning platform based on open source computational fluid dynamics software OpenFOAM is proposed, which is DRLinFluids. The platform has the characteristics of automation, quickness and simplicity, and can quickly call reinforcement learning for different research problems.

Different from TensorFlow, PyTorch and other general machine learning frameworks, this package takes OpenFOAM as an interactive environment, and further develops **a general CFD deep reinforcement learning package**.

[OpenFOAM](https://en.wikipedia.org/wiki/OpenFOAM) (for "Open-source Field Operation And Manipulation") is a C++ toolbox for the development of customized numerical solvers, and pre-/post-processing utilities for the solution of continuum mechanics problems, most prominently including computational fluid dynamics (CFD). In fact, due to the versatility of OpenFOAM, in addition to computational fluid dynamics problems, it can also deal with any ODE or PDE problems. Users can create their own solver for practical application by setting the control equations and boundary conditions of specific problems. This also gives DRLinFluids a wider usage.

## Features
- Built on top of the open-source software OpenFOAM, which can be downloaded and used by anyone free of charge.
- Provides many demonstration code (example cases) and will be continually updated.
- Offers a parallel DRL environment interface that maximizes computer resources.
- Uses RegExps technology to provide a scalable, real-time way to add, delete, search and modify OpenFOAM dictionary files.
- Implemented in pure Python without relying on any DLLs or C/C++ code.
- Simple and lightweight, with core code that is less than 1000 lines.

## Installation

We understand that setting up a development environment can be a painful process, which is why we have created [Singularity](https://github.com/sylabs/singularity) images for you to use quickly. These images also include a thoroughly compatibility tested OpenFOAM 8 environment.

Singularity containers is the state of the art and the promising future for seamless deployment of computations and workloads across OSes, machines, clusters etc. Here are some references for Singularity:
- https://docs.sylabs.io/guides/3.11/user-guide/
- https://www.youtube.com/watch?v=nQTMJ9hqKNI

Example on some HPC systems: 
- https://hpc.hku.hk/hpc/software/singularity-container/
- https://www.cuhk.edu.hk/itsc/hpc/singularity.html
- https://researchcomputing.princeton.edu/support/knowledge-base/singularity
- https://hpc.nih.gov/apps/singularity.html

### From Singularity (Recommended)
Here we provide a one-line command for installing Singularity and DRLinFluids.
```bash
wget https://raw.githubusercontent.com/venturi123/DRLinFluids/main/DRLinFluids_install.sh && sudo bash DRLinFluids_install.sh && singularity version
```
If the correct Singularity version information is displayed, use the following command to enter the container.
```bash
singularity shell DRLinFluids.sif

# Activate DRLinFluids environment
drl

# Activate OpenFOAM8 environment
of8
```

or install Singularity-CE manually.
```bash
sudo apt update
sudo apt install runc wget
wget https://github.com/sylabs/singularity/releases/download/v3.11.3/singularity-ce_VERSION.deb
sudo dpkg -i ./singularity-ce_VERSION.deb

# Check the installation status of Singularity
singularity version

# Pull the Singularity image from Singularity library
singularity pull DRLinFluids.sif library://qlwang/main/drlinfluids:latest
```

Of course, you can install the package from PyPI or source code. However, we do not recommend it as a first choice because it may cause potential compatibility issues.
### From PyPI

```bash
pip install drlinfluids
```

### From Source code

```
git clone https://github.com/venturi123/DRLinFluids.git
pip3 install -e drlinfluids
```

## Examples

Please see `/examples` directory for quick start.

We have merge the examples repository [DRLinFluids-examples](https://github.com/venturi123/DRLinFluids-examples) into the main repository. Now you can find all the examples in the `/examples` directory.

## How to cite

Please cite the framework as follows if you use it in your publications:

```
Qiulei Wang (王秋垒), Lei Yan (严雷), Gang Hu (胡钢), Chao Li (李朝), Yiqing Xiao (肖仪清), Hao Xiong (熊昊), Jean Rabault, and Bernd R. Noack , "DRLinFluids: An open-source Python platform of coupling deep reinforcement learning and OpenFOAM", Physics of Fluids 34, 081801 (2022) https://doi.org/10.1063/5.0103113
```

## Contributors

DRLinFluids is currently developed and maintained by 

[AIWE Lab, HITSZ](http://aiwe.hitsz.edu.cn)

- [Qiulei Wang](https://github.com/venturi123)

- [Lei Yan](https://github.com/1900360)

- [Gang Hu](http://faculty.hitsz.edu.cn/hugang)

[Jean Rabault](https://github.com/jerabaul29)

[Bernd Noack](http://www.berndnoack.com/)

## Publications using DRLinFluids package
1. [DRLinFluids: An open-source Python platform of coupling deep reinforcement learning and OpenFOAM](https://doi.org/10.1063/5.0103113) (Archived to DRLinFluids examples)
2. [Deep reinforcement learning-based active flow control of vortex-induced vibration of a square cylinder](https://doi.org/10.1063/5.0152777) (Archived to DRLinFluids examples)

## Contributing
Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

We invite all users to further discuss and ask for help directly on Github, through the issue system, and we commit to helping develop a community around the DRLinFluids framework by providing in-depth documentation and help to new users.

## License
`DRLinFluids` is licensed under the terms of the Apache License 2.0 license.
