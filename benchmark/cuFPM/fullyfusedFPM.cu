#include "mex.h"
#include "cuda/mxGPUArray.h"
#include "cuda/kernel_foo.cu"
#include "cuda/helpers.cu"
#include <cuda_runtime.h>

#include <iostream>
#include <stdio.h>

dim3 size2dim3( const mxGPUArray * in){
    const mwSize *sz = mxGPUGetDimensions(in);
    const int  dim = (int) mxGPUGetNumberOfDimensions(in);
    dim3 imgSz;
    imgSz = {(unsigned) sz[1], (unsigned) sz[0], 1};
    if (dim > 2){
        imgSz.z = (unsigned) sz[2];
    }
    return imgSz;
}

void mexFunction(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const *  __restrict__ prhs[]
){
    // input parames
    const mxGPUArray * wavefront1;
    const mxGPUArray * wavefront2;
    const mxGPUArray * obseY;
    const mxGPUArray * ledIdx;
    // const mxGPUArray *pratio;

    const creal32_T * __restrict__ d_wavefront1;
    const creal32_T * __restrict__ d_wavefront2;
    const real32_T * __restrict__ d_obseY;
    const int2 * __restrict__ d_ledIdx;
    // const int *d_pratio;
    // output parames
    mxGPUArray * dldw1;
    mxGPUArray * dldw2;

    creal32_T * __restrict__ d_dldw1;
    creal32_T * __restrict__ d_dldw2;
    

    mxInitGPU();

    wavefront1  = mxGPUCreateFromMxArray(prhs[0]); // sample wavefront
    wavefront2  = mxGPUCreateFromMxArray(prhs[1]); // pupil wavefront
    obseY       = mxGPUCreateFromMxArray(prhs[2]); // observed intensity
    ledIdx      = mxGPUCreateFromMxArray(prhs[3]); // LED position in pixel
    int pratio  = (int) mxGetPr(prhs[4])[0];


    d_wavefront1 = (const creal32_T * __restrict__ ) mxGPUGetDataReadOnly(wavefront1);
    d_wavefront2 = (const creal32_T * __restrict__ ) mxGPUGetDataReadOnly(wavefront2);
    d_obseY      = (const real32_T * __restrict__ ) mxGPUGetDataReadOnly(obseY);
    d_ledIdx     = (const int2 * __restrict__ ) mxGPUGetDataReadOnly(ledIdx);
    // d_pratio     = (const int *) mxGPUGetDataReadOnly(pratio);
    

    // dldw1 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(wavefront1), mxGPUGetDimensions(wavefront1),
    //                             mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_INITIALIZE_VALUES);

    dldw1 = mxGPUCopyFromMxArray(prhs[0]);

    dldw2 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(wavefront2), mxGPUGetDimensions(wavefront2),
                                mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_INITIALIZE_VALUES);

    d_dldw1 = (creal32_T * __restrict__ ) mxGPUGetData(dldw1);
    d_dldw2 = (creal32_T * __restrict__ ) mxGPUGetData(dldw2);

    // get image size                            
    dim3 imLs_sz = size2dim3(obseY);
    dim3 imHs_sz = size2dim3(wavefront1);

    // allocate temp variables
    creal32_T * __restrict__ latentZ;
    creal32_T * __restrict__ latentZ_copy;
    creal32_T * __restrict__ subwave;
    creal32_T * __restrict__ recordZ;
    // real32_T * __restrict__ dodx;
    // real32_T * __restrict__ dody;
    // real32_T * __restrict__ absO;

    unsigned arraysize = imLs_sz.x * imLs_sz.y * imLs_sz.z;
    cudaMalloc((creal32_T**)&latentZ, (2*arraysize) * sizeof(float));
    cudaMalloc((creal32_T**)&subwave, (2*arraysize) * sizeof(float));
    cudaMalloc((creal32_T**)&recordZ, (2*arraysize) * sizeof(float));
    cudaMalloc((creal32_T**)&latentZ_copy, (2*arraysize) * sizeof(float));

    // cudaMalloc((real32_T**)&dodx, (arraysize) * sizeof(float));
    // cudaMalloc((real32_T**)&dody, (arraysize) * sizeof(float));
    // cudaMalloc((real32_T**)&absO, (arraysize) * sizeof(float));


    // d_dldw1 = fftshift(fft2(d_targt))
    FFT_sample_forward(imHs_sz.x,imHs_sz.y,d_wavefront1,d_dldw1);

    // d_subwave = d_dldw1(kt:tb,kl:kr);
    // d_latentZ = d_subwave .* d_pupil;
    getLatentZ(d_dldw1,d_wavefront2,d_ledIdx,
                imLs_sz,imHs_sz,
                subwave,latentZ);

    dim3 N_BLOCKS = {
        (unsigned) (imLs_sz.x + BLOCK_X - 1) / BLOCK_X,
        (unsigned) (imLs_sz.y + BLOCK_Y - 1) / BLOCK_Y,
        (unsigned) imLs_sz.z
    };

    dim3 N_BLOCKS_L = {
        (unsigned) (imHs_sz.x + BLOCK_X - 1) / BLOCK_X,
        (unsigned) (imHs_sz.y + BLOCK_Y - 1) / BLOCK_Y,
        (unsigned) 1
    };

    dim3 N_THREADS = {BLOCK_X,BLOCK_Y,1};

    
    setConstraint<<<N_BLOCKS, N_THREADS>>>(imLs_sz,pratio,d_obseY,latentZ,latentZ_copy);
    cudaFree(latentZ);
    // d_latent_record = fftshift(fft2(d_latentZ));
    backwardLatentZ(latentZ_copy, imLs_sz, recordZ);

    // d_latentZ = conj(d_pupil) .* d_latent_record ;
    deconvPIE<<<N_BLOCKS, N_THREADS>>>(recordZ, d_wavefront2, imLs_sz, latentZ_copy);

    ReduceAddpupil<<<N_BLOCKS, N_THREADS>>>(recordZ, subwave, imLs_sz, d_dldw2);
    
    clear_spectrum<<<N_BLOCKS_L, N_THREADS>>>(imHs_sz, d_dldw1);


    dim3 N_BLOCKS_test = {
        (unsigned) (imLs_sz.x + BLOCK_X - 1) / BLOCK_X,
        (unsigned) (imLs_sz.y + BLOCK_Y - 1) / BLOCK_Y,
        (unsigned) imLs_sz.z
    };
    
    stitch_spectrum<<<N_BLOCKS_test, N_THREADS>>>(latentZ_copy, d_ledIdx, imLs_sz, imHs_sz, d_dldw1);

    FFT_sample_backward(imHs_sz, d_dldw1);
   

    plhs[0] = mxGPUCreateMxArrayOnGPU(dldw1);
    plhs[1] = mxGPUCreateMxArrayOnGPU(dldw2);
    mxGPUDestroyGPUArray(dldw1);
    mxGPUDestroyGPUArray(dldw2);

    
    cudaFree(subwave);
    cudaFree(recordZ);
    cudaFree(latentZ_copy);
    // cudaFree(dodx);
    // cudaFree(dody);
    // cudaFree(absO);

    mxGPUDestroyGPUArray(wavefront1);
    mxGPUDestroyGPUArray(wavefront2);
    mxGPUDestroyGPUArray(obseY);
    mxGPUDestroyGPUArray(ledIdx);
    // mxGPUDestroyGPUArray(pratio);
}