# Troubleshooting / Q&A

## Q: Why do I get `StackOverflowError` when running on CPU?

**A:** This typically happens if you are using standard `Array` (CPU) inputs but haven't loaded an FFT backend.
`AcceleratedDCTs.jl` relies on `AbstractFFTs.jl` to dispatch to the correct FFT implementation. For CPU arrays, you **must** load `FFTW.jl`.

**Solution:**
Add `using FFTW` to your script or project.

```julia
using AcceleratedDCTs
using FFTW  # Required for CPU FFTs

x = rand(1024)
y = dct(x)  # Works!
```

Without `FFTW`, `plan_rfft` (used internally) fails to find a specific implementation and may fall back to generic methods that cause infinite recursion.
