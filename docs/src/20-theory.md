# Theory & Algorithms

This package provides fast, device-agnostic implementations of the following DCT variants for 1D, 2D, and 3D data:

*   **DCT-II** (Forward) and **DCT-III** (Inverse): The most common DCT pair, used in signal processing, image compression (JPEG), and solving PDEs with half-sample symmetric boundary conditions.
*   **DCT-I** and **IDCT-I**: DCT with whole-sample symmetric boundary conditions, useful for applications requiring symmetric extensions.

## DCT-II / DCT-III Definitions

**Forward Transform (DCT-II)**
```math
X_k = \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi}{N}\left(n+\frac{1}{2}\right)k\right)
```

**Inverse Transform (DCT-III)**
```math
x_n = \frac{1}{2}X_0 + \sum_{k=1}^{N-1} X_k \cos\left(\frac{\pi}{N}n\left(k+\frac{1}{2}\right)\right)
```

## DCT-I Definition

**DCT-I**
```math
Y_k = X_0 + (-1)^k X_{M-1} + 2\sum_{j=1}^{M-2} X_j \cos\left(\frac{\pi j k}{M-1}\right)
```

The IDCT-I is computed as `DCT-I(x) / (2M-2)`, making it self-inverse up to scaling.

## The General Idea of Implementation (Makhoul's Algorithm)

A standard DCT-II of length $N$ can be computed by:
1.  **Preprocessing**: Permuting the input sequence $x$ into a new sequence $x'$.
    *   $x'$ takes even indices $0, 2, 4...$ from the front.
    *   $x'$ takes odd indices $1, 3, 5...$ from the back (reversed).
2.  **FFT**: Computing the Real FFT of $x'$.
3.  **Postprocessing**: Applying complex weights (twiddle factors) to the FFT output to recover DCT coefficients.

This approach is faster than $O(N^2)$ direct matrix multiplication and often faster than other $O(N \log N)$ approaches because highly optimized FFT libraries (FFTW, cuFFT) can be leveraged.

## Algorithm 2 (2D)

For a 2D $N_1 \times N_2$ grid, we apply the permutation logic independently to both rows and columns.
*   **Input**: $x(n_1, n_2)$
*   **Permutation**: $x'(n_1, n_2) = x(\tau(n_1), \tau(n_2))$
*   **Transform**: $X = \text{RFFT}(x')$ (2D Real FFT)
*   **Reconstruction**: $y(n_1, n_2) = 2 \operatorname{Re}(\dots)$ involving sums of 4 symmetric points from $X$.

## Algorithm 3 (3D)

We extend this to 3D.
*   **Separable Permutation**: $x'(n_1, n_2, n_3) = x(\tau(n_1), \tau(n_2), \tau(n_3))$.
    *   This scatters the spatial correlation but allows us to use a single 3D FFT.
*   **3D RFFT**: $X = \text{3D\_RFFT}(x')$.
*   **Recursive Reconstruction**:
    The post-processing extracts the cosine components.

```math
y = 2 \operatorname{Re} \{ W_3 \cdot [ W_2 \cdot ( W_1 \cdot X + \dots ) + \dots ] \}
```

This is implemented efficiently in a single kernel pass in `src/dct_optimized.jl`.

## Major Contribution

The major contribution of this package is the implementation of **Algorithm 2 (2D)** and **Algorithm 3 (3D)** in pure Julia in a device agnostic way, which reduce $N$-dimensional DCTs to $N$-dimensional Real-to-Complex (R2C) FFTs with $O(N)$ pre/post-processing steps, avoiding the overhead of separable 1D transforms (which require redundant transposes). In particular, the 3D version is the first implementation of Algorithm 3 to the best of our knowledge.

## DCT-I Algorithm

The DCT-I is computed using a **symmetric extension + FFT** approach:

1.  **Mirroring**: Extend the input $X$ of length $M$ into a symmetric sequence of length $N = 2M-2$:
    $$g = [X_0, X_1, \ldots, X_{M-1}, X_{M-2}, \ldots, X_1]$$
2.  **FFT**: Compute the Real FFT of the mirrored sequence.
3.  **Extraction**: Take the real part of the first $M$ FFT coefficients.

This approach leverages optimized R2C FFT libraries and is implemented in `src/dct1_optimized.jl`.

## References

- J. Makhoul, “A fast cosine transform in one and two dimensions,” IEEE Transactions on Acoustics, Speech, and Signal Processing, 1980.
- See also https://arxiv.org/abs/2110.01172 for detailed algorithms for 1D & 2D DCTs.
- And their CUDA C implementation (1D & 2D): https://github.com/JeremieMelo/dct_cuda