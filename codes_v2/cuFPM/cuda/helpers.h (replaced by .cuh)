#include "mex.h"
#include "mxGPUArray.h"
#include "addon.h"

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

void update_forward(
    const dim3 imLs_sz,
    const dim3 imHs_sz,
    const real32_T * d_obseY,
    const int2 *d_ledIdx,
    const creal32_T *wavefront1,
    const creal32_T *wavefront2,

    creal32_T * dldw1,
    creal32_T * dldw2
);






