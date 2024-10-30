# CMA Light

This repository contains all the files needed to run and reproduce the results presented in the pre-print https://arxiv.org/abs/2307.15775 which is currently under peer-revision.

The results on regression tasks can be reproduced after downloading the datasets from the drive below. https://drive.google.com/drive/folders/1-MIe3ub6NaBRBiOIpX3cqXKjFhHuM00Q?usp=sharing.

__Organization of the repository__

1) cma.py -> CMA Algorithm from https://arxiv.org/abs/2212.01848
2) cmalight.py -> CMA Light algorithm
3) main.py -> Main file to run the tests on regression problems
4) main_imclass.py -> Main file to run the tests on classification problems with ResNets
5) network.py -> Wrapper for the different types of neural networks and for the data handling
6) plot_history.py -> To replicate the history of the training loss and/or validation accuracy against time and/or epochs
7) utils.py -> General utilities
8) utils_cmalight.py -> Specific utils for CMA Light
