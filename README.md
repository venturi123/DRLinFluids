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

Also, we recommend a wonderful [practical tutorial](https://github.com/jerabaul29/guidelines_workflow_project) here on how to use singularity technology, feel free to watch it.

### From Singularity (Preferred by most users)
Here we provide a one-line command for installing Singularity and DRLinFluids automatically.
```bash
wget https://raw.githubusercontent.com/venturi123/DRLinFluids/main/singularity_install.sh && sudo bash singularity_install.sh && rm -rf singularity_install.sh && singularity version
```
If the correct Singularity version information is displayed, use the following command to pull and enter the container.
```bash
# Pull DRLinFluids image from Singularity library (may take a while)
singularity pull DRLinFluids.sif library://qlwang/main/drlinfluids:latest

# Verify the image (optional)
singularity verify DRLinFluids.sif

# Enter DRLinFluids container
singularity shell DRLinFluids.sif

# Activate DRLinFluids environment
drl

# or Activate OpenFOAM 8 environment
of8
```

### From Singularity `tar` package (Recommanded for stale Singularity version)
There may be a variety of reasons why you can only use a fairly old version of singularity, and we also provide a compressed image file as a tar archive. After extracting it, you can get a `folder-format` image instead of a `sif` image.

1. Go to the [Releases](https://github.com/venturi123/DRLinFluids/releases) page
2. Find the latest version and download all `DRLinFluids-v0.1.1.tar.gz__part.xx` (take v0.1.1 for example) and `sha256sums` (for validation) files in the same folder
3. Following the next steps
```bash
cat DRLinFluids-v0.1.1.tar.gz__part.?? > DRLinFluids-v0.1.1.tar.gz

sha256sum --check sha256sums

# The sha256sum line should print:
# DRLinFluids-v0.1.1.tar.gz: OK

# Then untar the file
tar xfp DRLinFluids-v0.1.1.tar.gz

# To enter the container
singularity shell --writable --fakeroot --no-home DRLinFluids-v0.1.1/
```

### From Singularity manually
```bash
sudo apt update
sudo apt install runc wget
wget https://github.com/sylabs/singularity/releases/download/v3.11.3/singularity-ce_${VERSION}.deb
sudo dpkg -i ./singularity-ce_${VERSION}.deb

# Check the installation status of Singularity
singularity version

# Pull the Singularity image from Singularity library
singularity pull DRLinFluids.sif library://qlwang/main/drlinfluids:latest
singularity shell --writable --fakeroot --no-home DRLinFluids-v0.1.1/
```

## Examples

Please see `/examples` directory for quick start.

> **Note**
> When you mount a Singularity image, it will be mounted with the directories `$HOME`, `/tmp`, `/proc`, `/sys`, `/dev`, and `$PWD` by default. You can move your own cases to the `$HOME` directory or its subdirectories, or use the `--bind` argument to bind other storage devices. For example, if you need to work under the `/media` directory, you can use the command: `singularity shell --bind /media DRLinFluids.sif`

### Run step by step
```bash
cd DRLinFluids/examples

# Compile the custom boundary condition (No need, just for illustration only)
# (All custom boundary condition have been compiled in DRLinFluids container)
cd newbc
./wmakeall

# or run 2D cylinder case
cd cylinder2D_multiprocessing
python DRLinFluids_cylinder/launch_multiprocessing_traning_cylinder.py

# or run 2D square case
cd square2D_multiprocessing
python DRLinFluids_square/launch_multiprocessing_traning_square.py

# or run square2D_VIV case
cd square2D_VIV_multiprocessing
python DRLinFluids_square2D_VIV/launch_multiprocessing_traning_square2D_VIV.py
```
The following contents indicate a successful runnning of DRLinFluids
```
(DRLinFluids) Singularity> python DRLinFluids_cylinder/launch_multiprocessing_traning_cylinder.py
OpenFOAM_init running time： 2.67 s
OpenFOAM_init running time： 2.39 s
OpenFOAM_init running time： 2.49 s
OpenFOAM_init running time： 2.43 s
OpenFOAM_init running time： 2.43 s
WARNING:root:No min_value bound specified for state.
Agent defined DONE!
Runner defined DONE!
Episodes:   0%|  
```
Then you just have to wait patiently and marvel at the amazing power of DRL : )

## How to cite

Please cite the framework as follows if you use it in your publications:

```
Qiulei Wang (王秋垒), Lei Yan (严雷), Gang Hu (胡钢), Chao Li (李朝), Yiqing Xiao (肖仪清), Hao Xiong (熊昊), Jean Rabault, and Bernd R. Noack , "DRLinFluids: An open-source Python platform of coupling deep reinforcement learning and OpenFOAM", Physics of Fluids 34, 081801 (2022) https://doi.org/10.1063/5.0103113
```

## Publications using DRLinFluids package
1. [DRLinFluids: An open-source Python platform of coupling deep reinforcement learning and OpenFOAM](https://doi.org/10.1063/5.0103113) (Archived to DRLinFluids examples)
2. [Deep reinforcement learning-based active flow control of vortex-induced vibration of a square cylinder](https://doi.org/10.1063/5.0152777) (Archived to DRLinFluids examples)
3. [Stabilizing the square cylinder wake using deep reinforcement learning for different jet locations](https://doi.org/10.1063/5.0171188) 
4. [Aerodynamic force reduction of rectangular cylinder using deep reinforcement learning-controlled multiple jets](https://doi.org/10.1063/5.0189009)
5. [Intelligent active flow control of long-span bridge deck using deep reinforcement learning integrated transfer learning](https://doi.org/10.1016/j.jweia.2023.105632) (Archived to DRLinFluids examples)
6. [Dynamic feature-based deep reinforcement learning for flow control of circular cylinder with sparse surface pressure sensing](https://doi.org/10.1017/jfm.2024.333) (Archived to DRLinFluids examples)

To be continued ...
## Contributing
Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

We invite all users to further discuss and ask for help directly on Github, through the issue system, and we commit to helping develop a community around the DRLinFluids framework by providing in-depth documentation and help to new users.

## License
`DRLinFluids` is licensed under the terms of the Apache License 2.0 license.
