# CUDA fused Fourier ptychography

This is an implementation of MATLAB + CUDA programming for the acceleration of Fourier ptychographic microscopy (FPM) reconstruction.

The codes were built based on MATLAB 2024b and CUDA v12.8, and were tested with a graphic card of NIVIDA RTX 3090 24GB.

## Acceleration ratio

The codes were tested on a personal desktop running a Windows 11 x64, with 64GB RAM, and a graphic card of NIVIDA RTX 3090 24GB.

In FPM implementation, a total of **361 images** were collected with **2048 by 2048 pixels** 16 bits. The reconstruction upsample rate is 8 so the resolution of reconstructed image is of **16384 by 16384** pixels.

The reconstruction duration using the fused FP is **100s** on average compared to conventional MATLAB GPU implementation, which tasks about **300s**. Acceleration is about **3 folds**.

| Implementation        | Execution duration (s)   | 
| --------   | -----:  |
| MATLAB    | 500 s   |
| MATLAB + GPU        |   310 s   |
| MATLAB + CUDA (fused FPM)        |    100.2 s    |

## Requirements

* MATLAB 2024b
* CUDA v12.8
* Visual Studio 2022 community

## To Build the codes

The cuda codes are designed and implemented based on MATLAB c++ interfaces including "mex.h" and "mxGPUArray.h". The "mex.h" provide basic support to mex and build the cuda codes. The "mxGPUArray.h" provides support for the array types of MATLAB.

To build the codes "fullyfusedFPM.cu" you will need  "mexcuda" to run "mexcuda -lcufft fullyfusedFPM.cu" command in the command line of the MATLAB, at the root of the file "fullyfusedFPM.cu".

The "mexcuda" need several preconditions.
First, to download "visual studio 2022 community" and add "cl.exe" into the environment variables.
Then, to download "Windows SDK".
Third, when the first and second steps were done, the following should be added to the environment variables

> C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64

Create a new system variables named "INCLUDE", and add the following to the list.

> C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include; <br>
> C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt; <br>
> C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um; <br>
> C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared; <br>

Create a new system variables named "LIB", and add the following to the list.

> C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\lib\x64; <br>
> C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64; <br>
> C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64; <br>

When running mexcuda in matlab, one should first type
> setenv("NVCC_APPEND_FLAGS", '-allow-unsupported-compiler')

in the MATLAB command line so that the MATLAB can use the compiler of VS 2022 community. Otherwise the MATLAB may pop up warning. 
