# DRLinFluids
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/venturi123/DRLinFluids/blob/main/LICENSE)

DRLinFluids is a flexible package to utilize Deep Reinforcement Learning in the field of Computational Fluid Dynamics (CFD).

**Note: This package is still in rapid development cycle under Ubuntu 20.04 LTS and OpenFOAM 8. The APIs are not stable yet. We recommend users to keep the package at the latest version.**

## Table of contents

- [Introduction](#introduction)
- [Installation](#installation)
- [How to cite](#how-to-cite)
- [Core development team and contributors](#core-development-team-and-contributors)
- [Contributing](#contributing)
- [License](#license)


## Introduction

Reinforcement learning is a field of machine learning. It studies by interacting with the environment. It emphasizes how to make corresponding behavior in a specific environment in order to maximize the expected benefits. However, for reinforcement learning, it is necessary to define a specific interaction environment for specific problems, which is rather cumbersome and takes up a lot of time of researchers in related fields, and delays the research speed of researchers in reinforcement learning and fluid cross field. For this purpose, a reinforcement learning platform based on open source computational fluid dynamics software OpenFOAM is proposed, which is DRLinFluids. The platform has the characteristics of automation, quickness and simplicity, and can quickly call reinforcement learning for different research problems.

Different from TensorFlow, PyTorch and other general machine learning frameworks, this platform takes OpenFOAM as an interactive environment, and further develops **a general CFD reinforcement learning package**.

[OpenFOAM](https://en.wikipedia.org/wiki/OpenFOAM) (for "Open-source Field Operation And Manipulation") is a C++ toolbox for the development of customized numerical solvers, and pre-/post-processing utilities for the solution of continuum mechanics problems, most prominently including computational fluid dynamics (CFD). In fact, due to the versatility of OpenFOAM, in addition to computational fluid dynamics problems, it can also deal with any ODE or PDE problems. Users can create their own solver for practical application by setting the control equations and boundary conditions of specific problems. This also gives DRLinFluids a wider usage.


## Installation

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

Besides, we also build an additional repository [DRLinFluids-examples](https://github.com/venturi123/DRLinFluids-examples) for quick understanding and testing, and the user-friendly DRLinFluids package will be uploaded on this page in the near future.

We have developed Docker and Singularity containers with examples of DRLinFluids. For specific application commands, please refer to [DRLinFluids-examples](https://github.com/venturi123/DRLinFluids-examples).

Singularity containers is the state of the art and the promising future for seamless deployment of computations and workloads across OSes, machines, clusters etc: https://docs.sylabs.io/guides/3.5/user-guide/introduction.html https://en.wikipedia.org/wiki/Singularity_(software) , example on some HPC systems: https://documentation.sigma2.no/software/containers.html https://blogs.iu.edu/ncgas/2021/04/29/a-quick-intro-to-singularity-containers/ https://centers.hpc.mil/users/singularity.html https://ulhpc-tutorials.readthedocs.io/en/latest/containers/singularity/ 

## How to cite

Please cite the framework as follows if you use it in your publications:

```
Qiulei Wang (王秋垒), Lei Yan (严雷), Gang Hu (胡钢), Chao Li (李朝), Yiqing Xiao (肖仪清), Hao Xiong (熊昊), Jean Rabault, and Bernd R. Noack , "DRLinFluids: An open-source Python platform of coupling deep reinforcement learning and OpenFOAM", Physics of Fluids 34, 081801 (2022) https://doi.org/10.1063/5.0103113
```

For more citation formats, please see https://aip.scitation.org/action/showCitFormats?type=show&doi=10.1063%2F5.0103113.


## Core development team and contributors

DRLinFluids is currently developed and maintained by 

[AIWE Lab, HITSZ](http://aiwe.hitsz.edu.cn)

- [Qiulei Wang](https://github.com/venturi123)

- [Lei Yan](https://github.com/1900360)

- [Gang Hu](http://faculty.hitsz.edu.cn/hugang)

[Jean Rabault](https://github.com/jerabaul29)

[Bernd Noack](http://www.berndnoack.com/)


## Contributing
Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

We invite all users to further discuss and ask for help directly on Github, through the issue system, and we commit to helping develop a community around the DRLinFluids framework by providing in-depth documentation and help to new users.

## Continue work
In the future the following functionality is planned to be added:
- DMD/DMDc for simulation
- Apply to other bluff bodies

## License
`DRLinFluids` is licensed under the terms of the Apache License 2.0 license.
