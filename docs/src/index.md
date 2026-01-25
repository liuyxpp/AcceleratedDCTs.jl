# AcceleratedDCTs.jl

Documentation for [AcceleratedDCTs](https://github.com/lyx/AcceleratedDCTs.jl).

## Introduction

AcceleratedDCTs.jl aims to provide the fastest possible Discrete Cosine Transform (DCT) for Julia, running on both CPUs and GPUs. It focuses on the **DCT-II** (Standard "DCT") and **DCT-III** (Inverse DCT), commonly used in signal processing and solving partial differential equations (PDEs).

The core innovation of this package is the implementation of **Algorithm 2 (2D)** and **Algorithm 3 (3D)**, which reduce $N$-dimensional DCTs to $N$-dimensional Real-to-Complex (R2C) FFTs with $O(N)$ pre/post-processing steps, avoiding the overhead of separable 1D transforms (which require redundant transposes).
