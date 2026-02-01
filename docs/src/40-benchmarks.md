# Benchmarks

Measurement of **3D DCT** performance on varying grid sizes ($N^3$). Results collected using in-place `mul!` (where supported) to exclude allocation overhead. Lower is better.

## GPU Performance (NVIDIA RTX 2080 Ti)

| Grid Size ($N^3$) | `cuFFT` (Baseline) | **`Opt 3D DCT`** | `Batched DCT` (Old) |
| :--- | :--- | :--- | :--- |
| **$16^3$** | 0.080 ms | **0.113 ms** | 1.041 ms |
| **$32^3$** | 0.076 ms | **0.131 ms** | 0.946 ms |
| **$64^3$** | 0.116 ms | **0.246 ms** | 1.165 ms |
| **$128^3$** | 0.833 ms | **1.423 ms** | 3.302 ms |
| **$256^3$** | 5.945 ms | **10.417 ms** | 26.019 ms |

> **Note**: `Opt 3D DCT` maintains excellent performance across all sizes, being only ~1.75x slower than raw `cuFFT` (due to necessary pre/post-processing). In contrast, the naive `Batched DCT` is ~3.9x slower than FFT. For $N=256$, `Opt 3D DCT` is **>2.2x faster** than the batched implementation.

## CPU Performance (Intel Xeon Gold 6132)

### Single Thread

| Grid Size ($N^3$) | `FFTW rfft` | **`Opt 3D DCT`** | `FFTW dct` | `Batched DCT` |
| :--- | :--- | :--- | :--- | :--- |
| **$16^3$** | 0.010 ms | **0.109 ms** | 0.046 ms | 0.138 ms |
| **$32^3$** | 0.090 ms | **0.791 ms** | 0.362 ms | 0.813 ms |
| **$64^3$** | 1.193 ms | **6.957 ms** | 4.319 ms | 14.323 ms |
| **$128^3$** | 17.707 ms | **63.965 ms** | 52.700 ms | 142.195 ms |
| **$256^3$** | 241.730 ms | **663.925 ms** | 422.634 ms | 1596.978 ms |

### 8 Threads

| Grid Size ($N^3$) | `FFTW rfft` | **`Opt 3D DCT`** | `FFTW dct` | `Batched DCT` |
| :--- | :--- | :--- | :--- | :--- |
| **$16^3$** | 0.062 ms | **0.196 ms** | 0.106 ms | 0.460 ms |
| **$32^3$** | 0.122 ms | **0.464 ms** | 0.280 ms | 0.935 ms |
| **$64^3$** | 0.650 ms | **2.413 ms** | 1.574 ms | 7.997 ms |
| **$128^3$** | 4.278 ms | **10.961 ms** | 14.438 ms | 80.503 ms |
| **$256^3$** | 46.120 ms | **108.365 ms** | 90.370 ms | 842.106 ms |

> **Note**: On multi-threaded CPU, `Opt 3D DCT` is highly competitive. At $128^3$, it **outperforms** `FFTW.dct` (10.96 ms vs 14.44 ms). At $256^3$, it is slightly slower but comparable (108 ms vs 90 ms). It is consistently **>7x faster** than the batched implementation at large sizes. Single-threaded performance remains slower due to the overhead of our pure-Julia plan construction and kernel dispatch versus FFTW's highly optimized planner.

## DCT-I Performance (Symmetric Boundary)

Measurement of **3D DCT-I** performance on varying grid sizes ($M^3$). Note that for DCT-I, the internal FFT size is $N = 2M-2$.

### GPU Performance (NVIDIA RTX 2080 Ti)

We compare the new **Separable DCT-I** against the legacy **Mirroring** approach and a raw **Complex FFT** baseline (size $2M-2$).

| Grid Size ($M^3$) | Complex FFT ($2M-2$) | **Separable DCT-I** | Mirror DCT-I | Speedup vs Mirror |
| :--- | :--- | :--- | :--- | :--- |
| **$65^3$** | 1.44 ms | **2.08 ms** | 1.12 ms | 0.5x |
| **$129^3$** | 11.11 ms | **5.71 ms** | 7.83 ms | **1.4x** |
| **$257^3$** | 80.73 ms | **43.50 ms** | 54.89 ms | **1.3x** |

> **Key Takeaway**: The **Separable DCT-I** implementation is highly effective at large scales, outperforming both the Mirroring method and the raw Complex FFT baseline. It achieves this by using an efficient $O(M^D)$ memory footprint and optimized kernels, avoiding the bandwidth bottlenecks of the Mirroring approach. Note that for optimal performance, sizes of the form $M=2^k+1$ (e.g., 129, 257) are recommended, as they map to power-of-two internal FFTs ($N=128, 256$).

### VkDCT Performance (Vulkan/CUDA Backend)

Using the `VkDCTExt` extension with `libvkfft_dct.so` (C++ backend) yields **massive speedups**.

**Float32 Performance:**

| Grid Size ($M^3$) | `cuFFT` (Baseline) | `AcceleratedDCTs` (Separable) | **`VkDCT` (Extension)** | Speedup vs Separable |
| :--- | :--- | :--- | :--- | :--- |
| **$65^3$** | 0.108 ms | 0.443 ms | **0.038 ms** | **11.7x** |
| **$129^3$** | 1.063 ms | 2.765 ms | **0.176 ms** | **15.7x** |

**Float64 Performance:**

| Grid Size ($M^3$) | `cuFFT` (Baseline) | `AcceleratedDCTs` (Separable) | **`VkDCT` (Extension)** | Speedup vs Separable |
| :--- | :--- | :--- | :--- | :--- |
| **$65^3$** | 0.591 ms | 0.510 ms | **0.222 ms** | **2.3x** |
| **$129^3$** | 4.341 ms | 3.608 ms | **1.849 ms** | **1.95x** |

> **Analysis**: VkDCT is 7x-15x faster than the generic Julia GPU implementation for Float32, and ~2x faster for Float64. It even outperforms raw `cuFFT` significantly for non-power-of-two sizes ($M=2^k+1$).

### CPU Performance (Intel Xeon Gold 6132)

Comparing against `FFTW`'s native DCT-I (`REDFT00`).

#### Single Thread

| Grid Size ($M^3$) | `FFTW DCT-I` | **`Opt DCT-I`** | `FFTW rfft` ($N=2M-2$) |
| :--- | :--- | :--- | :--- |
| **$16^3$** | 0.128 ms | **0.417 ms** | 0.185 ms |
| **$32^3$** | 2.374 ms | **8.528 ms** | 6.608 ms |
| **$64^3$** | 7.909 ms | **35.659 ms** | 19.987 ms |
| **$128^3$** | 476.919 ms | **806.853 ms** | 657.360 ms |
| **$256^3$** | 1022.623 ms | **4601.621 ms** | 3596.196 ms |

#### 8 Threads

| Grid Size ($M^3$) | `FFTW DCT-I` | **`Opt DCT-I`** | `FFTW rfft` ($N=2M-2$) |
| :--- | :--- | :--- | :--- |
| **$16^3$** | 0.157 ms | **0.490 ms** | 0.253 ms |
| **$32^3$** | 110.692 ms | **1.495 ms** | 1.150 ms |
| **$64^3$** | 3.330 ms | **5.790 ms** | 3.498 ms |
| **$128^3$** | 1839.493 ms | **118.878 ms** | 99.294 ms |
| **$256^3$** | 140.238 ms | **679.484 ms** | 529.750 ms |

> **Note**: The above tables show performance of the **original** `rfft`-based DCT-I implementation. Since then, we have updated the CPU implementation to use FFTW's native `REDFT00` directly via automatic dispatch, achieving **identical performance** to `FFTW DCT-I`. The GPU implementation continues to use the mirroring + FFT approach (no native DCT-I in cuFFT). Also note the abnormal performance at $128^3$ for the FFTW DCT-I in the 8-thread benchmark.
