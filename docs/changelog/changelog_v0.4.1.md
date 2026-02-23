# Changelog

All notable changes to AcceleratedDCTs.jl will be documented in this file.

## [v0.4.1] - 2026-02-23

### Added
- `VkDCT_jll` (v1.3.4) as a direct dependency, providing pre-compiled `libvkfft_dct` for GPU DCT-I. Users no longer need to manually compile the CUDA shim.
- `lib/VkDCT/build_tarballs.jl`: Yggdrasil recipe (reference copy) for building `VkDCT_jll`.
- `docs/design/vkdct_jll_integration.md`: Design document for the JLL integration.

### Changed
- `ext/VkDCTExt.jl`: Library loading now follows priority: `ENV["VKDCT_LIB"]` → `VkDCT_jll.libvkfft_dct` → local fallback.
- Updated `README.md`, `docs/src/index.md`, `docs/src/10-tutorial.md`, and `docs/src/30-implementation.md` to reflect zero-setup VkDCT integration.

### Fixed
- Removed dead links in `README.md` (Issue #5).
- Removed duplicate "3D Optimized" entry in `docs/src/index.md`.

## [v0.4.0] - 2026-02-01

### Added
- VkDCT extension (`ext/VkDCTExt.jl`) with VkFFT-based GPU DCT-I backend.
- `lib/VkDCT`: C++/CUDA shim wrapping VkFFT for 3D DCT-I (Float32 & Float64).
- Benchmarks for VkDCT extension.

## [v0.3.0]

### Added
- Separable split-radix DCT-I algorithm with permuted GPU strategy.
- FFTW-based DCT-I for CPU arrays.
- Mirror-based DCT-I (legacy).
