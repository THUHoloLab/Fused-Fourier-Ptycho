#pragma once

#include "mex.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cufft.h>

__global__ void getSubpupil(
    const unsigned z_offset,
    const creal32_T* wavefront1,
    const creal32_T* wavefront2,
    const int2* ledindex,
    const dim3 imgSz,
    const dim3 imgSzL,
    creal32_T* subwave,
    creal32_T* latentZ
);

__global__ void cufftShift_2D_kernel(
    creal32_T* data, 
    int N
);

__global__ void deconvPIE(
    const unsigned z_offset,
    const creal32_T* x_record,
    const creal32_T* subwave,
    const int2* ledindex,
    const creal32_T* pupil,
    const dim3 imLs_sz,
    const dim3 imHs_sz, 
    // creal32_T* __restrict__ latentz,
    creal32_T* dldw,
    creal32_T* dldp
);

// __global__ void ReduceAddpupil(
//     const creal32_T* __restrict__ x_record,
//     const creal32_T* __restrict__ subwave,
//     const dim3 imgSz,
//     creal32_T* __restrict__ dldw
// );

__global__ void stitch_spectrum(
    const creal32_T* latent,
    const int2* ledindex,
    const dim3 imgSz,
    const dim3 imgSzL, 
    creal32_T* dldw
);

__global__ void clear_spectrum(
    const dim3 imgSzL,
    creal32_T* spect
);

__global__ void ifftCorrection(
    creal32_T* spectrum,
    const dim3 imHs_sz
);

__global__ void differenceKernel(
    const unsigned z_offset,
    const dim3 imLs_sz,
    const int pratio,
    const real32_T* img_Y,
    const creal32_T* latentz,
    creal32_T* out
);

__global__ void clear_data(
    const dim3 imgSzL,
    real32_T * data
);

__global__ void RMSprop_step(
    creal32_T *grad,
    const float beta,
    const float lr,
    const float eps,
    const dim3 imgSz,
    real32_T *mom2,
    creal32_T *wavefront
);

__global__ void setPupilConstrain(
    const real32_T * pupil,
    const float am,
    const dim3 imgSz,
    creal32_T * wavefront
);
