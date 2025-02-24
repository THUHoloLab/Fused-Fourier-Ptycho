#include "helpers.h"

void FFT_sample_forward(
    const int sz_h,
    const int sz_w,
    const creal32_T* __restrict__ x,
    creal32_T* __restrict__ y
){
    dim3 N_THREADS(BLOCK_X,BLOCK_Y,1);
    dim3 N_BLOCKS = {
        (unsigned) ((sz_h + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((sz_w + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) 1
    };

    cufftHandle plan;
    cufftPlan2d(&plan, sz_h, sz_w, CUFFT_C2C);
    cufftExecC2C(plan, (cufftComplex *)&x[0], (cufftComplex *)&y[0],CUFFT_FORWARD);
    cufftDestroy(plan);

    cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(y,sz_h);
}

void FFT_sample_backward(
    const dim3 imgSz,
    creal32_T* __restrict__ y
){
    dim3 N_THREADS(BLOCK_X,BLOCK_Y,1);
    dim3 N_BLOCKS = {
        (unsigned) ((imgSz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imgSz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) 1
    };
    cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(y,(int) imgSz.x);

    cufftHandle plan;
    cufftPlan2d(&plan, imgSz.x, imgSz.y, CUFFT_C2C);
    cufftExecC2C(plan, (cufftComplex *)&y[0], (cufftComplex *)&y[0],CUFFT_INVERSE);
    cufftDestroy(plan);

    ifftCorrection<<<N_BLOCKS, N_THREADS>>>(y,imgSz);
}

void getLatentZ(
    const creal32_T* __restrict__ wavefront1,
    const creal32_T* __restrict__ pupil,
    const int2* __restrict__ ledIdx,
    const dim3 imLs_sz,
    const dim3 imHs_sz,
    creal32_T* __restrict__ supwave,
    creal32_T* __restrict__ latentz
){
    

    dim3 N_BLOCKS = {
        (unsigned) ((imLs_sz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imLs_sz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imLs_sz.z
    };
    dim3 N_THREADS(BLOCK_X,BLOCK_Y,1);

    int inembed[2];
    for (int i{0}; i < 2; i++) {
        inembed[i] = (int) imLs_sz.x;
    }

    getSubpupil<<<N_BLOCKS, N_THREADS>>>(
        wavefront1, 
        pupil,
        ledIdx,
        imLs_sz,
        imHs_sz,
        supwave,
        latentz
    );

    cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(latentz, (unsigned) imLs_sz.x);

    cufftHandle plan;
    cufftPlanMany(
        &plan, 
        2, 
        &inembed[0], // n
        &inembed[0], // inembed
        1,           // istride
        (int) imLs_sz.x * (int) imLs_sz.y, // idist
        &inembed[0], // inembed
        1,           // istride
        (int) imLs_sz.x * (int) imLs_sz.y, // idist
        CUFFT_C2C, 
        (int) imLs_sz.z
    );
    cufftExecC2C(plan, (cufftComplex *)&latentz[0], (cufftComplex *)&latentz[0], CUFFT_INVERSE);
    cufftDestroy(plan);
}



void backwardLatentZ(
    const creal32_T* __restrict__ latentz,
    const dim3 imLs_sz,
    creal32_T* __restrict__ latentz_record
){
    int inembed[2];
    for (int i{0}; i < 2; i++) {
        inembed[i] = (int) imLs_sz.x;
    }

    dim3 N_BLOCKS = {
        (unsigned) ((imLs_sz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imLs_sz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imLs_sz.z
    };
    dim3 N_THREADS(BLOCK_X, BLOCK_Y, 1);

    cufftHandle plan;
    cufftPlanMany(
        &plan, 
        2, 
        &inembed[0], // n
        &inembed[0], // inembed
        1,           // istride
        (int) imLs_sz.x * (int) imLs_sz.y, // idist
        &inembed[0], // inembed
        1,           // istride
        (int) imLs_sz.x * (int) imLs_sz.y, // idist
        CUFFT_C2C, 
        (int) imLs_sz.z
    );
    cufftExecC2C(plan, (cufftComplex *)&latentz[0], (cufftComplex *)&latentz_record[0], CUFFT_FORWARD);
    cufftDestroy(plan);

    cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(latentz_record, (unsigned) imLs_sz.x);
}