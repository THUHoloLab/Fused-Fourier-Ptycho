#pragma once

#include "mex.h"
#include "mxGPUArray.h"
#include "kernel_foo.cuh"
#include "addon.h"

void FFT_sample_forward(
    const int sz_h,
    const int sz_w,
    const creal32_T* x,
    creal32_T* y
);

void FFT_sample_backward(
    const dim3 imgSz,
    creal32_T* y
);

void getLatentZ(
    const unsigned z_offset,
    const creal32_T* wavefront1,
    const creal32_T* pupil,
    const int2* ledIdx,
    const dim3 N_BLOCKS,
    const dim3 imLs_sz,
    const dim3 imHs_sz,
    creal32_T* supwave,
    creal32_T* latentz
);

void backwardLatentZ(
    const creal32_T* latentz,
    const dim3 imLs_sz,
    creal32_T* latentz_record
);

// void update_forward(
//     const dim3 imLs_sz,
//     const dim3 imHs_sz,
//     const real32_T * d_obseY,
//     const int2 *d_ledIdx,
//     const creal32_T *wavefront1,
//     const creal32_T *wavefront2,

//     creal32_T * dldw1,
//     creal32_T * dldw2
// );






