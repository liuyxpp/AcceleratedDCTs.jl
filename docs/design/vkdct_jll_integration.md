# PR: Integrate VkDCT_jll for zero-setup GPU DCT-I support

## Summary

This PR integrates the newly registered [`VkDCT_jll`](https://github.com/JuliaBinaryWrappers/VkDCT_jll.jl) package (v1.3.4) into AcceleratedDCTs.jl, enabling GPU-accelerated 3D DCT-I transforms without any manual compilation steps. Users now only need `using CUDA, AcceleratedDCTs` to get full GPU support.

## Changes

| File | Description |
|------|-------------|
| `Project.toml` | Add `VkDCT_jll` as a direct dependency with compat `1.3.4` |
| `ext/VkDCTExt.jl` | Load `libvkfft_dct` from JLL (primary), with ENV override and local fallback for development |
| `lib/VkDCT/build_tarballs.jl` | **[NEW]** Yggdrasil recipe for building VkDCT_jll (reference copy) |
| `lib/VkDCT/Project.toml` | **[NEW]** Minimal project file for the build recipe |
| `test/Project.toml` | Add `CUDA` to test dependencies |
| `.gitignore` | Add entry for compiled `.so` files |

## Library loading priority in `VkDCTExt`

1. `ENV["VKDCT_LIB"]` — manual override for development/testing
2. `VkDCT_jll.libvkfft_dct` — production path (if JLL artifact is available)
3. Local `lib/VkDCT/libvkfft_dct.so` — fallback for local development

## Testing

All 8 VkDCT tests pass (Float32/Float64 forward DCT, IDCT roundtrip, plan inversion, `mul!`, `ldiv!`):

```
Test Summary:                       | Pass  Total   Time
VkDCT Float64 and IDCT Verification |    8      8  26.4s
```

## Notes

- `VkDCT_jll` is a lightweight JLL wrapper; it installs safely on non-GPU systems (no CUDA hardware required). The actual binary artifact is only downloaded on supported Linux x86_64 platforms.
- The Yggdrasil recipe (`lib/VkDCT/build_tarballs.jl`) is kept as a reference copy. The canonical version lives in [Yggdrasil/V/VkDCT](https://github.com/JuliaPackaging/Yggdrasil).
