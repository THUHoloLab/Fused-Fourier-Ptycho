#include "mex.h"
#include "mxGPUArray.h"

#include <cuda_runtime.h>
#include <cufft.h>

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)

void FFT_sample_forward(
    const int sz_h,
    const int sz_w,
    const creal32_T* __restrict__ x,
    creal32_T* __restrict__ y
);

void FFT_sample_backward(
    const dim3 imgSz,
    creal32_T* __restrict__ y
);

void getLatentZ(
    const creal32_T* __restrict__ wavefront1,
    const creal32_T* __restrict__ pupil,
    const int2* __restrict__ ledIdx,
    const dim3 imLs_sz,
    const dim3 imHs_sz,
    creal32_T* __restrict__ supwave,
    creal32_T* __restrict__ latentz
);

void backwardLatentZ(
    const creal32_T* __restrict__ latentz,
    const dim3 imLs_sz,
    creal32_T* __restrict__ latentz_record
);






