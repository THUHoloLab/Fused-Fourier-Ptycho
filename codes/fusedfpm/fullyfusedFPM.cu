// This framework is licensed under the BSD 3-clause license.
// If you use it in your research, we would appreciate a citation via
//
// @software{CUDA-fused FPM,
// 	  author = {THU Hololab},
// 	  license = {BSD-3-Clause},
// 	  month = {2},
// 	  title = {{CUDA-fused FPM}},
// 	  url = {https://github.com/THUHoloLab/FusedFourierPtycho},
// 	  version = {1.0},
// 	  year = {2025}
// }

#include "mex.h"
#include "cuda/mxGPUArray.h"
#include "cuda/kernel_foo.cu"
#include "cuda/helpers.cu"
#include <cuda_runtime.h>

#include <iostream>
#include <stdio.h>


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
    

    dldw1 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(wavefront1), mxGPUGetDimensions(wavefront1),
                                mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_INITIALIZE_VALUES);

    dldw2 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(wavefront2), mxGPUGetDimensions(wavefront2),
                                mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_INITIALIZE_VALUES);

    d_dldw1 = (creal32_T * __restrict__ ) mxGPUGetData(dldw1);
    d_dldw2 = (creal32_T * __restrict__ ) mxGPUGetData(dldw2);

    // get image size                            
    const int dim = (int) mxGPUGetNumberOfDimensions(obseY);
    const mwSize *sz = mxGPUGetDimensions(obseY);
    dim3 imLs_sz;
    if (dim == 1){
        imLs_sz = { (unsigned) sz[1], (unsigned) sz[0], 1};
    }else{
        imLs_sz = { (unsigned) sz[1], (unsigned) sz[0], (unsigned) sz[2]};
    }
    sz = mxGPUGetDimensions(wavefront1);
    dim3 imHs_sz = { (unsigned) sz[1], (unsigned) sz[0], 1};

    // allocate temp variables
    creal32_T * __restrict__ latentZ;
    creal32_T * __restrict__ subwave;
    creal32_T * __restrict__ recordZ;
    real32_T * __restrict__ dodx;
    real32_T * __restrict__ dody;
    real32_T * __restrict__ absO;

    unsigned arraysize = imLs_sz.x * imLs_sz.y * imLs_sz.z;
    cudaMalloc((creal32_T**)&latentZ, (2*arraysize) * sizeof(float));
    cudaMalloc((creal32_T**)&subwave, (2*arraysize) * sizeof(float));
    cudaMalloc((creal32_T**)&recordZ, (2*arraysize) * sizeof(float));

    cudaMalloc((real32_T**)&dodx, (arraysize) * sizeof(float));
    cudaMalloc((real32_T**)&dody, (arraysize) * sizeof(float));
    cudaMalloc((real32_T**)&absO, (arraysize) * sizeof(float));


    // d_dldw1 = fftshift(fft2(d_targt))
    FFT_sample_forward(imHs_sz.x,imHs_sz.y,d_wavefront1,d_dldw1);

    // d_subwave = d_dldw1(kt:tb,kl:kr);
    // d_latentZ = d_subwave .* d_pupil;
    getLatentZ(d_dldw1,d_wavefront2,d_ledIdx,
                imLs_sz,imHs_sz,
                subwave,latentZ);

    dim3 N_BLOCKS = {
        (unsigned) ((imLs_sz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imLs_sz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imLs_sz.z
    };

    dim3 N_BLOCKS_L = {
        (unsigned) ((imHs_sz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imHs_sz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) 1
    };

    dim3 N_THREADS(BLOCK_X,BLOCK_Y,1);

    // d_amplitude = ifft2(ifftshift(d_latentZ)) / pratio^2;
    getAmplitude<<<N_BLOCKS, N_THREADS>>>(latentZ, imLs_sz, pratio, absO);

    // [d_dx,d_dy] = grad(d_amplitude - d_obseY);
    getGradients<<<N_BLOCKS, N_THREADS>>>(absO, d_obseY, imLs_sz, dodx, dody);

    // d_amplitude = backward(d_dx,d_dy);
    differenceMap<<<N_BLOCKS, N_THREADS>>>(dodx,dody,imLs_sz,absO);

    // d_latentZ = d_amplitude * sign(d_latentZ) .* d_pratio^2;
    newX<<<N_BLOCKS, N_THREADS>>>(absO, imLs_sz, pratio, latentZ);
    
    // d_latent_record = fftshift(fft2(d_latentZ));
    backwardLatentZ(latentZ, imLs_sz, recordZ);
    // d_latentZ = conj(d_pupil) .* d_latent_record ;
    deconvPIE<<<N_BLOCKS, N_THREADS>>>(recordZ, d_wavefront2, imLs_sz, latentZ);

    ReduceAddpupil<<<N_BLOCKS, N_THREADS>>>(recordZ, subwave, imLs_sz, d_dldw2);
    
    clear_spectrum<<<N_BLOCKS_L, N_THREADS>>>(imHs_sz, d_dldw1);

    stitch_spectrum<<<N_BLOCKS, N_THREADS>>>(latentZ, d_ledIdx, imLs_sz, imHs_sz, d_dldw1);

    FFT_sample_backward(imHs_sz, d_dldw1);
   

    plhs[0] = mxGPUCreateMxArrayOnGPU(dldw1);
    plhs[1] = mxGPUCreateMxArrayOnGPU(dldw2);
    mxGPUDestroyGPUArray(dldw1);
    mxGPUDestroyGPUArray(dldw2);

    cudaFree(latentZ);
    cudaFree(subwave);
    cudaFree(recordZ);
    cudaFree(dodx);
    cudaFree(dody);
    cudaFree(absO);

    mxGPUDestroyGPUArray(wavefront1);
    mxGPUDestroyGPUArray(wavefront2);
    mxGPUDestroyGPUArray(obseY);
    mxGPUDestroyGPUArray(ledIdx);
    // mxGPUDestroyGPUArray(pratio);
}
