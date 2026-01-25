## Cosine transforms on the GPU

Unfortunately cuFFT does not provide cosine transforms and so we must write our own fast cosine
transforms for the GPU. We implemented the fast 1D and 2D cosine transforms described by [Makhoul80](@citet)
which compute it by applying the regular Fourier transform to a permuted version of the array.

In this section we will be using the DCT-II as the definition of the forward cosine transform
for a real signal of length ``N``
```math
    \begin{equation}
    \label{eq:FCT}
    \text{DCT}(X): \quad Y_k = 2 \sum_{j=0}^{N-1} \cos \left[ \frac{\pi(j + \frac{1}{2})k}{N} \right] X_j \, ,
    \end{equation}
```
and the DCT-III as the definition of the inverse cosine transform
```math
    \begin{equation}
    \label{eq:IFCT}
    \text{IDCT}(X): \quad Y_k = X_0 + 2 \sum_{j=1}^{N-1} \cos \left[ \frac{\pi j (k + \frac{1}{2})}{N} \right] X_j \, ,
    \end{equation}
```
and will use ``\omega_M = e^{-2 \pi \mathrm{i} / M}`` to denote the ``M^\text{th}`` root of unity, sometimes called the twiddle factors
in the context of FFT algorithms.

### 1D fast cosine transform
To calculate \eqref{eq:FCT} using the fast Fourier transform, we first permute the input signal along the appropriate
dimension by ordering the odd elements first followed by the even elements to produce a permuted signal
```math
    X^\prime_n =
    \begin{cases}
        \displaystyle X_{2N}, \quad 0 \le n \le \left[ \frac{N-1}{2} \right] \, , \\
        \displaystyle X_{2N - 2n - 1}, \quad \left[ \frac{N+1}{2} \right] \le n \le N-1 \, ,
    \end{cases}
```
where ``[a]`` indicates the integer part of ``a``. This should produce, for example,
```math
    \begin{equation}
    \label{eq:permutation}
    (a, b, c, d, e, f, g, h) \quad \rightarrow \quad (a, c, e, g, h, f, d, b) \, ,
    \end{equation}
```
after which \eqref{eq:FCT} is computed using
```math
  Y = \text{DCT}(X) = 2 \text{Re} \left\lbrace \omega_{4N}^k \text{FFT} \lbrace X^\prime \rbrace \right\rbrace \, .
```

### 1D fast inverse cosine transform
The inverse \eqref{eq:IFCT} can be computed using
```math
  Y = \text{IDCT}(X) = \text{Re} \left\lbrace \omega_{4N}^{-k} \text{IFFT} \lbrace X \rbrace \right\rbrace \, ,
```
after which the inverse permutation of \eqref{eq:permutation} must be applied.

### 2D fast cosine transform
Unfortunately, the 1D algorithm cannot be applied dimension-wise so the 2D algorithm is more
complicated. Thankfully, the permutation \eqref{eq:permutation} can be applied dimension-wise.
The forward cosine transform for a real signal of length ``N_1 \times N_2`` is then given by
```math
Y_{k_1, k_2} = \text{DCT}(X_{n_1, n_2}) =
2 \text{Re} \left\lbrace
    \omega_{4N_1}^k \left( \omega_{4N_2}^k \tilde{X} + \omega_{4N_2}^{-k} \tilde{X}^- \right)
\right\rbrace \, ,
```
where ``\tilde{X} = \text{FFT}(X^\prime)`` and ``\tilde{X}^-`` indicates that ``\tilde{X}`` is indexed in reverse.

### 2D fast inverse cosine transform
The inverse can be computed using
```math
Y_{k_1, k_2} = \text{IDCT}(X_{n_1, n_2}) =
\frac{1}{4} \text{Re} \left\lbrace
    \omega_{4N_1}^{-k} \omega_{4N_2}^{-k}
    \left( \tilde{X} - M_1 M_2 \tilde{X}^{--} \right)
    - \mathrm{i} \left( M_1 \tilde{X}^{-+} + M_2 \tilde{X}^{+-} \right)
\right\rbrace \, ,
```
where ``\tilde{X} = \text{IFFT}(X)`` here, ``\tilde{X}^{-+}`` is indexed in reverse along the first dimension,
``\tilde{X}^{-+}`` along the second dimension, and ``\tilde{X}^{--}`` along both. ``M_1`` and ``M_2`` are masks of lengths
``N_1`` and ``N_2`` respectively, both containing ones except at the first element where ``M_0 = 0``. Afterwards, the inverse
permutation of \eqref{eq:permutation} must be applied.

Due to the extra steps involved in calculating the cosine transform in 2D, running with two
wall-bounded dimensions typically slows the model down by a factor of 2. Switching to the FACR
algorithm may help here as a 2D cosine transform won't be necessary anymore.
