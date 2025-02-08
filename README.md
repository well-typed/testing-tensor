# Tensor library for use in QuickCheck tests

This is a pure Haskell implementation of tensors, emphasizing simplicity over
everything else. It is intended to be used as a model in tests.

## System dependencies

### For the library

The library does not have any system dependencies.

### For the tests

#### fftw

The FFT test suite module depends on
[`fft`](https://hackage.haskell.org/package/fft), which in turn depends on
[fftw3](https://www.fftw.org/). On Ubuntu this can be installed using

```
sudo apt install libfftw3-dev
```

#### cuDNN

The cuDNN test suite depends on the NVIDIA cuDNN library. You might need to
add something like this to your `cabal.project.local`:

```
package testing-tensor
  extra-lib-dirs: /usr/local/cuda-12.5/lib64
  extra-include-dirs: /usr/local/cuda-12.5/include
```
