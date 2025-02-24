# CUDA fused Fourier ptychography

This is an implementation of MATLAB + CUDA programming for the acceleration of Fourier ptychographic microscopy (FPM) reconstruction.

The codes were built based on MATLAB 2024b and CUDA v12.8, and was tested with a graphic card of NIVIDA RTX 3090 24GB.

## Acceleration ratio

The codes were tested on a personal desktop running a Windows 11 OS, with 64GB RAM, and a graphic card of NIVIDA RTX 3090 24GB. 

In FPM implementation, a total of **361 images** were collected with **2048 by 2048 pixels** 16bits. The reconstruction upsample rate is 8 so the reconstructed image is of **16384 by 16384** pixels. 

The reconstruction duration using the fused FP is **100s** on average compared to conventional MATLAB GPU implementation, which tasks about **300s**. Acceleration by about 3 folds.

## Requirements

* MATLAB 2024b
* CUDA v12.8
* Visual Studio 2022 community.
