![example workflow](https://github.com/twitter-research/graph-neural-pde/actions/workflows/python-package.yml/badge.svg)

![Cora_animation_16](https://user-images.githubusercontent.com/5874124/143270624-265c2d01-39ca-488c-b118-b68f876dfbfa.gif)

## Introduction

This repository contains the source code for the publications [GRAND: Graph Neural Diffusion](https://icml.cc/virtual/2021/poster/8889) and [Beltrami Flow and Neural Diffusion on Graphs (BLEND)](https://arxiv.org/abs/2110.09443).
These approaches treat deep learning on graphs as a continuous diffusion process and Graph Neural
Networks (GNNs) as discretisations of an underlying PDE. In both models, the layer structure and
topology correspond to the discretisation choices
of temporal and spatial operators. Our approach allows a principled development of a broad new
class of GNNs that are able to address the common plights of graph learning models such as
depth, oversmoothing, and bottlenecks. Key to
the success of our models are stability with respect to perturbations in the data and this is addressed for both 
implicit and explicit discretisation schemes. We develop linear and nonlinear
versions of GRAND, which achieve competitive results on many standard graph benchmarks. BLEND is a non-Euclidean extension of GRAND that jointly evolves the feature and positional encodings of each node providing a principled means to perform graph rewiring.

## Running the experiments

### Requirements
Check the dependencies in `environment.yml`.
Create a conda environment from yml file:
```
conda env create -f environment.yml
```
Also please refer to the [original repository](https://github.com/twitter-research/graph-neural-pde).


## GRAND (Graph Neural Diffusion)

### Dataset and Preprocessing
Create a root level folder
```
./data
```
This will be automatically populated the first time each experiment is run.

### Experiments
First, move into folder `src`.

In `src`, there are two main files:
- run_GNN.py: run experiments with 1 train-val-test split, 20 random inits.
- run_reproduce.py: can specify the number of train-val-test split to run (the paper experiments with 100 splits, 20 random inits for each split).

Commands to run experiments are specified in `run_constant.sh`, `run_nl.sh` and `run_rl_rewire.sh`, please refer to these files for more details. 

## NOTE:
- CUDA errors can occur when running experiments on 2 datasets: PubMed and Computer (need to be fixed).
- For the other 5 datasets (Cora, Citeseer, Photo, CoauthorCS, ogbn-arxiv):
  - run_GNN.py:
      - linear: can run on all datasets
      - non-linear: can run on all datasets
      - non-linear-rewire: can run out of memory (need to be fixed) due to the increased number of edges during rewiring
  - run_reproduce.py:
      - linear: can run on all datasets except for ogbn-arxiv
      - non-linear: can run on all datasets except for ogbn-arxiv
      - non-linear-rewire: can run out of memory (need to be fixed) due to the increased number of edges during rewiring