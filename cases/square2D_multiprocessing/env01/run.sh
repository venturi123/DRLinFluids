#!/bin/bash

decomposePar

mpirun -np 4 pimpleFoam -parallel

reconstructPar
