---
layout: post
title:  "Fourier Interpolation Classes"
date:   2023-09-15
description: Learn how to interpolate periodic, uniform data using the Fourier transform!
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<p class="intro"><span class="dropcap">I</span>n today's post, we are going to review how to interpolate periodic, uniform data using the discrete Fourier transform. You may find the accompanying <a href="https://github.com/KunkelAlexander/periodicinterpolation-python">Python code on GitHub</a>. The repository also includes code to up- and downscale images using the Fourier transform.</p>


## Intro
This is a brief review of three interpolation methods using the Fourier transform with the aim of providing short code snippets that work in $$d=2$$ dimensions with obvious generalisations to higher dimensions for odd input and output sizes. For the definitions of the Discrete Fourier Transform (DFT) and its inverse (IDFT) used in the following see the excellent [NumPy FFT documentation][numpy-fft-documentation]. In the following, we are going to make use of NumPy's fftfreq, ifftfreq, fftshift and ifftshift functions that account for the different spectra for even and odd $$N$$.


<img src="{{ site.baseurl }}/assets/img/fourierinterpolation-python/header.png" alt="" width="100%">



### Setup
For testing different interpolation strategies, we sample a periodic test function on a squared-sized 2D grid with side length $$L$$ and 1D subgrids like $$\{0, \Delta x, ..., (N-1)\cdot \Delta x\}$$ for $$\Delta x = L/N$$. We are going to verify the interpolation result on the 2D grid with 1D subgrids with $$2\cdot N$$ points like $$\{0, \Delta x/2, 2\cdot \frac{\Delta x}{2}, ..., (N_i-1)\cdot \frac{\Delta x}{2}\}$$.

{%- highlight python -%}
# import libraries
import numpy as np
from numpy.testing import assert_allclose as isclose
# set up test problem
L, N, Ni = 2, 32, 2*32
dx, dxi  = L/N, L/Ni
f        = lambda tx, ty : np.sin(2*np.pi*tx) * np.cos(2*np.pi*ty)
# 1D subarray excluding point f(L) since f(L) = f(0)
t        = np.arange(0, N ) * dx
tx, ty   = np.meshgrid(t, t)
# build 2D arrays
ti       = np.arange(0, Ni) * dxi
tix, tiy = np.meshgrid(ti, ti)
# momentum array respecting even and odd N
k        = 2*np.pi*np.fft.fftfreq(N)*N/L
kx, ky   = np.meshgrid(k, k)
# n-dimensional forward FFT including norm factor
fk       = np.fft.fftn(f(tx, ty), norm="forward")
{%- endhighlight -%}

### Direct Evaluation of IDFT
One way to interpolate data using the DFT, is to directly evaluate the definition of the IDFT at the $$N_i$$ interpolation points $$t_{i} \in \mathbb{R}$$. This method allows the interpolation of the input data at non-uniform interpolation points, but has the disadvantage of requiring $$\mathcal{O}(N^d\cdot N_i)$$ operations.

{%- highlight python -%}
# METHOD 1: evaluate IDFT at point t
# outer products to size [N, N, Ni, Ni] in exponential function
# then sum all plane waves in first and second dimension
fi1 = np.sum(fk[..., None, None]*\
    np.exp(1j*(kx[..., None, None]*tix +
               ky[..., None, None]*tiy)), axis=(0, 1))
# check result
isclose(np.abs(fi1), np.abs(f(tix, tiy)), atol=1e-14)
{%- endhighlight -%}

### Evaluation via Time-Shifting property
A second method to interpolate data is to make use of the fact that a phase rotation in the frequency domain equals a shift in the position domain. This method allows the interpolation of the input data on a uniform grid with $$N\cdot (2^d - 1)$$ complex multiplications in frequency space plus $$\mathcal{O}(N^d\cdot \log(N))$$ operations by leveraging the speed of the Fast Fourier Transform (FFT).

{%- highlight python -%}
# METHOD 2: phase rotation in frequency space
# shift in x-, y- and xy-direction
fix  = np.fft.ifftn(fk * np.exp(1j*(kx*dx/2          )), norm="forward")
fiy  = np.fft.ifftn(fk * np.exp(1j*(        + ky*dx/2)), norm="forward")
fixy = np.fft.ifftn(fk * np.exp(1j*(kx*dx/2 + ky*dx/2)), norm="forward")
# zip original array and shifted arrays
fi3             = np.zeros((Ni, Ni), dtype=complex)
fi3[ ::2,  ::2] = f(tx, ty)
fi3[ ::2, 1::2] = fix
fi3[1::2,  ::2] = fiy
fi3[1::2, 1::2] = fixy
# check result
isclose(np.abs(fi3), np.abs(f(tix, tiy)), atol=1e-15)
{%- endhighlight -%}


### Zero-Padding
The third and final method for data interpolation using the DFT is *zero-padding*. It exploits the fact that adding additional high-frequency modes with vanishing amplitude to the data in frequency space and then computing the IDFT is equal to sampling the input function on a finer grid with grid spacing $$\Delta x_i = \frac{L}{N_{i}}$$. Likewise, we can subtract high-frequency modes and downscale the input data with an IDFT. While we cannot freely choose interpolation points using zero-padding, the method is fast since it only requires one FFT with $$\mathcal{O}(N^d\cdot \log(N))$$.

{%- highlight python -%}
# METHOD 3: zero-padding
# shift zero frequencies to center of cube
fkPad    = np.fft.fftshift(fk)
# determine size of padding of negative frequencies
NPadN    = int(np.floor(Ni/2-N/2))
# if either the input or output size is uneven
# add one additional positive frequency
NPadP    = NPadN+(Ni+N)%2
fkPad    = np.pad(fkPad, ((NPadP, NPadN), (NPadP, NPadN)))
# shift zero frequencies back to outside of cube
fkPad    = np.fft.ifftshift(fkPad)
# go back to position space
fi3      = np.fft.ifftn(fkPad, norm="forward")
# check result
isclose(np.abs(fi3), np.abs(f(tix, tiy)), atol=1e-14)
{%- endhighlight -%}




[periodicinterpolation-python-gh]: https://github.com/KunkelAlexander/periodicinterpolation-python
[numpy-fft-documentation]: https://numpy.org/doc/stable/reference/routines.fft.html