#pragma once

#include "mex.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cufft.h>

__device__ float absC(const creal32_T in, const float ratio){
    float out;
    out = sqrtf(in.re * in.re + in.im * in.im) / ratio;
    return out;
}

__device__ float sign(const float in){
    float out;
    out = copysignf(1.0f,in);
    return out;
}

__global__ void getSubpupil(
    const creal32_T* __restrict__ wavefront1,
    const creal32_T* __restrict__ wavefront2,
    const int2* __restrict__ ledindex,
    const dim3 imgSz,
    const dim3 imgSzL,
    creal32_T* __restrict__ subwave,
    creal32_T* __restrict__ latentZ
);

__global__ void cufftShift_2D_kernel(
    creal32_T* __restrict__ data, 
    int N
);

__global__ void deconvPIE(
    const creal32_T* __restrict__ x_record,
    const creal32_T* __restrict__ subwave,
    const creal32_T* __restrict__ pupil,
    const dim3 imLs_sz,
    creal32_T* __restrict__ latentz,
    creal32_T* __restrict__ dldp
);

__global__ void ReduceAddpupil(
    const creal32_T* __restrict__ x_record,
    const creal32_T* __restrict__ subwave,
    const dim3 imgSz,
    creal32_T* __restrict__ dldw
);

__global__ void stitch_spectrum(
    const creal32_T* __restrict__ latent,
    const int2* __restrict__ ledindex,
    const dim3 imgSz,
    const dim3 imgSzL, 
    creal32_T* __restrict__ dldw
);

__global__ void clear_spectrum(
    const dim3 imgSzL,
    creal32_T* __restrict__ spect
);

__global__ void ifftCorrection(
    creal32_T* __restrict__ spectrum,
    const dim3 imHs_sz
);

__global__ void differenceKernel(
    const unsigned z_offset,
    const dim3 imLs_sz,
    const int pratio,
    const real32_T* __restrict__ img_Y,
    const creal32_T* __restrict__ latentz,
    creal32_T* __restrict__ out
);

__global__ void clear_data(
    const dim3 imgSzL,
    real32_T * data
);

__global__ void RMSprop_step(
    const creal32_T * grad,
    const float beta,
    const float lr,
    const float eps,
    const dim3 imgSz,
    real32_T * mom2,
    creal32_T * wavefront
);

__global__ void setPupilConstrain(
    const real32_T * pupil,
    const float am,
    const dim3 imgSz,
    creal32_T * wavefront
);
