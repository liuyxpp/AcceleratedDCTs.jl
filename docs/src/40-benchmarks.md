# Benchmarks

Measured on **NVIDIA RTX 2080 Ti**, grid size **384x384x384**.

| Implementation | Description | Time | Speedup vs Batched |
| :--- | :--- | :--- | :--- |
| **`Opt 3D DCT`** | **This Package (Algorithm 3) `dct`** | **42 ms** | **1.8x** |
| `dct_batched` | Batched 1D Separable (Reference) | 77 ms | 1.0x |
| `cuFFT rfft` | Theoretical Lower Bound (FFT only) | 26 ms | - |

On CPU (Single Thread 128^3), `dct` is **~2.5x faster** than the batched approach.
