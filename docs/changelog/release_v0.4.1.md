# AcceleratedDCTs.jl v0.4.1 Release Notes

## Zero-Setup GPU DCT-I via VkDCT_jll

This release integrates the newly registered [`VkDCT_jll`](https://github.com/JuliaBinaryWrappers/VkDCT_jll.jl) (v1.3.4) package, eliminating the need for manual CUDA shim compilation. GPU-accelerated 3D DCT-I transforms now work out of the box.

### What Changed

**Before (v0.4.0)**:
```bash
# Users had to manually compile:
cd lib/VkDCT && ./compile.sh
```

**After (v0.4.1)**:
```julia
# Just load the packages — everything is automatic
using CUDA, AcceleratedDCTs
p = plan_dct1(CuArray(rand(128, 128, 128)))  # VkDCT backend used automatically
```

### Details

- **`VkDCT_jll`** is added as a direct dependency. It is a lightweight JLL wrapper that installs safely on all platforms (no CUDA required for installation).
- The `VkDCTExt` extension loads `libvkfft_dct` from the JLL automatically when `CUDA.jl` is loaded.
- Library loading priority: `ENV["VKDCT_LIB"]` (dev override) → `VkDCT_jll` (production) → local fallback.
- Documentation updated throughout (`README.md`, tutorial, implementation details).

### Performance

No performance changes. The VkDCT backend continues to provide ~7x-15x speedup over the generic separable DCT-I implementation on NVIDIA GPUs.

### Full Changelog

See [CHANGELOG.md](./CHANGELOG.md) for the complete list of changes.
