# CUDA-fused Fourier ptychography (cuFPM)

This implements MATLAB + CUDA programming for the acceleration of Fourier ptychographic microscopy (FPM) reconstruction. FPM's forward and backward progress are all calculated purely by CUDA in "fullyfusedFPM.cu".

:bell: The codes were built based on MATLAB 2024b and CUDA v12.8, and were tested with a graphic card of NVIDIA RTX 3090 24GB.

:collision: **The codes are only available for images with even numbers of pixels** due to the implementation of "fftshift" kernel. [[cufftShift]](https://github.com/marwan-abdellah/cufftShift)
## Acceleration ratio

The codes were tested on a personal desktop <br>
* OS: WINDOWS 11 Pro x64, with 128GB RAM <br>
* GPU: NVIDIA RTX 3090 24GB <br>
* CPU: 12th Gen Intel(R) Core(TM) i9-12900K 3.2GHz <br>

The following image shows the benchmarks. The cuFPM was compared against MATLAB + CPU or MATLAB + GPU. The raw image is of 21gigapixel which can be obtained from [here](http://profoundism.com/21_gigapixel_total_renovation_of_girl_with_a_pearl_earring_for_sale_to_the_wisest_art_lover.html). 

<div align = 'center'>
<img src = "https://github.com/THUHoloLab/Fused-Fourier-Ptycho/blob/main/sources/benchmark.jpg" width = "700" alt="" align = center />
</div><br>

In FPM implementation, a total of **361 images** were collected with **2048 by 2048 pixels** 16 bits. The reconstruction upsample rate is 8 so the resolution of the reconstructed image is **16384 by 16384** pixels. [[dataset]](https://drive.google.com/drive/folders/1oWm-0svOYzlnrEdqr_P8A-UoB4-NcQxF?usp=drive_link).

The reconstruction duration using the fused FP is **100s** on average compared to conventional MATLAB GPU implementation, which tasks about **300s**. Acceleration is about **3 folds**.

| Implementation          | Size of input images       | Reconstructed size   | Batch size | Execution duration (s)   | 
| :----:                | :----:                     | :-----:        | :-----:          | :-----:  |
| MATLAB                   | 2048    × 2048 × 361      | 16384 × 16384   | 26            | 800 s   |
| MATLAB + GPU              | 2048    × 2048 × 361     | 16384 × 16384   | 26           | 310 s   |
| MATLAB + CUDA (cuFPM) | 2048    × 2048 × 361     | 16384 × 16384   | 26          | 100 s   |
| cuFPM-v2 | 2048    × 2048 × 361     | 16384 × 16384   | 26          | 68 s   |
| cuFPM-v2 | 2048    × 2048 × 361     | 16384 × 16384   | 36          | 54 s   |
| cuFPM-v2 | 512    × 512 × 361     | 4096 × 4096   | 26          | 2.5 s   |
| cuFPM-v2 | 1024    × 1024 × 361     | 8192 × 8192   | 26          | 10.3 s   |
| cuFPM-v2 | 2048    × 2048 × 93     | 16384 × 16384   | 26          | 21 s   |

## Requirements
* An NVIDIA GPU; All shown results come from an RTX 3090.
* MATLAB 2024b
* CUDA v12.8
* Visual Studio 2022 Community
* Windows Kits 10.0.26100.0 (higher version may be available but not tested)

## To build the codes

The cuda codes are designed and implemented based on MATLAB c++ interfaces including "mex.h" and "mxGPUArray.h". The "mex.h" provides basic support for mex and to build the cuda codes. The "mxGPUArray.h" provides support for the array types of MATLAB.

To build the codes "fullyfusedFPM.cu" you will need  "mexcuda" to run 
>mexcuda -lcufft fullyfusedFPM.cu

command in the command line of the MATLAB, at the root of the file "fullyfusedFPM.cu". "-lcufft" is a setting for "mexcuda" that tells the mex to use "cuFFT.h", the CUDA fast Fourier transform library [[cuFFT]](https://docs.nvidia.com/cuda/cufft/).

The "mexcuda" needs several preconditions.
First, to download [Visual Studio 2022 community](https://visualstudio.microsoft.com/vs/community/) and add "cl.exe" into the environment variables.
Then, to download [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/).
Third, when the first and second steps were done, the following should be added to the environment variables

> C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64

Create a new system variable named "INCLUDE", and add the following to the list.

> C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include; <br>
> C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt; <br>
> C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um; <br>
> C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared; <br>

Create a new system variable named "LIB", and add the following to the list.

> C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\lib\x64; <br>
> C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64; <br>
> C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64; <br>

When running mexcuda in MATLAB, one should first type
> setenv("NVCC_APPEND_FLAGS", '-allow-unsupported-compiler')

in the MATLAB command line so that MATLAB can use the compiler of the VS 2022 community. Otherwise, MATLAB may pop up errors. 

## License and Citation

This framework is licensed under the BSD 3-clause license. Please see `LICENSE.txt` for details.

If you use it in your research, we would appreciate a citation via
```bibtex
@software{CUDA-fused FPM,
	author = {THU Hololab},
	license = {BSD-3-Clause},
	month = {2},
	title = {{CUDA-fused FPM}},
	url = {https://github.com/THUHoloLab/Fused-Fourier-Ptycho},
	version = {1.0},
	year = {2025}
}
```

## Questions?
If you find any questions during the implementation please feel-free to open an issue. Thank you very much :blush:!
