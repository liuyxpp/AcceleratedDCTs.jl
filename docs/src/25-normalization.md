# Normalization & Conventions

This page clarifies the normalization conventions used in `AcceleratedDCTs.jl` compared to `FFTW.jl`. This is a common source of confusion, as `AcceleratedDCTs.jl` (following Makhoul's algorithm) uses a **non-unitary** definition, while `FFTW.dct` (and generally `AbstractFFTs` ecosystem) defaults to an **orthogonal (unitary)** definition (DCT-II).

## Definitions Used in AcceleratedDCTs.jl

We implement the standard "pure sum" definition for the Forward transform (DCT-II) and the matching Inverse transform (DCT-III):

**Forward Transform (DCT-II)**
```math
X_k = \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi}{N}\left(n+\frac{1}{2}\right)k\right)
```

**Inverse Transform (DCT-III)**
```math
x_n = \frac{1}{2}X_0 + \sum_{k=1}^{N-1} X_k \cos\left(\frac{\pi}{N}n\left(k+\frac{1}{2}\right)\right)
```

In this package's high-level API (`dct`, `idct`), we ensure `idct(dct(x)) ≈ x`. The necessary inverse scaling factors are handled internally by the inverse transform.

## Summary of Differences

| Feature | `AcceleratedDCTs.jl` (Makhoul) | `FFTW.jl` (Standard DCT-II) |
| :--- | :--- | :--- |
| **Mathematical Type** | Non-unitary DCT-II | Orthogonal (Unitary) DCT-II |
| **Scaling (Roundtrip)** | $2^d \cdot N$ | $2^d \cdot N$ `*` |
| **Forward Transform** | Unscaled sum | Scaled by normalization factors |
| **Inverse Transform** | Scaled by $2/N$ implicitly | Scaled by normalization factors |

`*` *Note: FFTW's unnormalized plans also involve scaling, but `FFTW.dct` usually implies the unitary one unless specified.*

## Detailed Normalization Coefficients

Here we compare the normalization coefficient $C$ applied to each element in the **Forward** transform. The coefficient depends on the frequency index $k$ in each dimension.

$$
y = C(k) \cdot \text{DCT}_{\text{sum}}(x)
$$

For `FFTW.jl` (Orthogonal), the rule is simple: **Any dimension with index $k=0$ contributes a factor of $\sqrt{1/N}$, while any dimension with index $k>0$ contributes $\sqrt{2/N}$.**

| Dimension | Type of Index ($k$) | `AcceleratedDCTs.jl` | `FFTW.jl` (Orthogonal) |
| :--- | :--- | :--- | :--- |
| **1D** | $k=0$ (DC) | $1$ | $\sqrt{1/N}$ |
| | $k>0$ (AC) | $1$ | $\sqrt{2/N}$ |
| **2D** | $(0,0)$ | $1$ | $\frac{1}{\sqrt{N_1 N_2}}$ |
| | $(0, k)$ with $k>0$ | $1$ | $\frac{1}{\sqrt{N_1}} \sqrt{\frac{2}{N_2}}$ |
| | $(k, 0)$ with $k>0$ | $1$ | $\sqrt{\frac{2}{N_1}} \frac{1}{\sqrt{N_2}}$ |
| | $(k, l)$ where $k,l>0$ | $1$ | $\frac{2}{\sqrt{N_1 N_2}}$ |
| **3D** | $(0,0,0)$ | $1$ | $\frac{1}{\sqrt{N_1 N_2 N_3}}$ |
| | $(0,0,k)$ ($k>0$) | $1$ | $\frac{1}{\sqrt{N_1 N_2}} \sqrt{\frac{2}{N_3}}$ |
| | $(0,k,l)$ ($k,l>0$) | $1$ | $\frac{1}{\sqrt{N_1}} \frac{2}{\sqrt{N_2 N_3}}$ |
| | $(k,l,m)$ ($k,l,m > 0$) | $1$ | $\frac{2\sqrt{2}}{\sqrt{N_1 N_2 N_3}}$ |

## Conclusion

`AcceleratedDCTs.jl` does **NOT** apply these position-dependent normalization factors during the forward transform. It computes the raw cosine sums.

If you need to match `FFTW`'s output exactly (e.g. for comparison), you must manually apply these scaling factors to the output of `AcceleratedDCTs.dct`.
