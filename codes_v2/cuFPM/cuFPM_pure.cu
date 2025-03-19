#include "mex.h"
#include "cuda/mxGPUArray.h"
#include "cuda/kernel_foo.cu"
#include "cuda/helpers.cu"
#include <cuda_runtime.h>

#include <iostream>
#include <stdio.h>

// convert mwSize to dim3 
static dim3 size2dim3( const mxGPUArray * in){
    const mwSize *sz = mxGPUGetDimensions(in);
    const int dim = (int) mxGPUGetNumberOfDimensions(in);
    dim3 imgSz = {(unsigned) sz[1], (unsigned) sz[0], 1};
    if (dim > 2){
        imgSz.z = (unsigned) sz[2];
    }
    return imgSz;
}

/**
 * @brief cuda implementation of FPM solver, with MATLAB interface
 * @param input  
 * input[0]: mxGPUArray for target, complex 
 * input[1]: mxGPUArray for pupil, complex 
 * input[2]: mxGPUArray for amplitude dataset, real 
 * input[3]: mxGPUArray for LED index, int32  
 * input[4]: mxGPUArray for pupil amplitude, real
 * input[4]: mxGPUArray for pupil amplitude, real
 * input[5]: mxArray pratio, upsampling ratio for reconstruction
 * input[5]: mxArray number of epoch, int32
 * input[6]: mxArray number of batch size, int32
 * input[7]: mxArray number of optimizers parameters, learning rate, beta, eps
 * @param out
 * out[0]: mxGPUArray reconstructed target
 * out[1]: mxGPUArray reconstructed pupil
 *
 * @note This is a CUDA implementation of FPM solver using feature-domain phase retrieval. 
 * The anisotropic image's first gradient is used for feature-domain loss function. 
 * The optimizer is RMSProp for simplifity. 
 * @warning There is no error checking in this implementation.
 * So that the code may araise unpredictable behaviors.
 */

void mexFunction(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const *  __restrict__ prhs[]
){
    // input parames
    const mxGPUArray * obseY;
    const mxGPUArray * ledIdx;
    const mxGPUArray * pupil;

    const real32_T * __restrict__ d_obseY;
    const real32_T * __restrict__ d_pupil;
    const int2 * __restrict__ d_ledIdx;
    
    mxGPUArray * wavefront1;
    mxGPUArray * wavefront2;
    creal32_T * __restrict__ d_wavefront1;
    creal32_T * __restrict__ d_wavefront2;

    // output parames
    mxInitGPU();

    wavefront1  = mxGPUCopyFromMxArray(prhs[0]); // sample wavefront
    wavefront2  = mxGPUCopyFromMxArray(prhs[1]); // pupil wavefront
    obseY       = mxGPUCreateFromMxArray(prhs[2]); // observed intensity
    ledIdx      = mxGPUCreateFromMxArray(prhs[3]); // LED position in pixel
    pupil       = mxGPUCreateFromMxArray(prhs[4]);

    int pratio     = (int) mxGetPr(prhs[5])[0];
    int numEpochs  = (int) mxGetPr(prhs[6])[0];
    int batch_sz   = (int) mxGetPr(prhs[7])[0];

    printf("prhs4: %d, prhs5: %d, prhs6: %d \n",pratio,numEpochs,batch_sz);

    float learning_rate = (float ) mxGetPr(prhs[8])[0];
    float beta = (float ) mxGetPr(prhs[8])[1];
    float eps = (float ) mxGetPr(prhs[8])[2];
    printf("lr: %f, beta: %f, eps: %f \n",learning_rate,beta,eps);

    d_wavefront1 = (creal32_T * __restrict__ ) mxGPUGetDataReadOnly(wavefront1);
    d_wavefront2 = (creal32_T * __restrict__ ) mxGPUGetDataReadOnly(wavefront2);

    d_obseY     = (const real32_T * __restrict__ ) mxGPUGetDataReadOnly(obseY);
    d_pupil     = (const real32_T * __restrict__) mxGPUGetDataReadOnly(pupil);   
    d_ledIdx    = (const int2 * __restrict__ ) mxGPUGetDataReadOnly(ledIdx);
                   
    dim3 imLs_sz = size2dim3(obseY);
    dim3 imHs_sz = size2dim3(wavefront1);

    creal32_T * __restrict__ latentZ;
    creal32_T * __restrict__ recordZ;
    creal32_T * __restrict__ subwave;
    creal32_T * __restrict__ d_dldw1;
    creal32_T * __restrict__ d_dldw2;
    real32_T * __restrict__ mom_w1; // second moment
    real32_T * __restrict__ mom_w2; // second moment

    unsigned arraysize = imLs_sz.x * imLs_sz.y * batch_sz;
    cudaMalloc((creal32_T**)&latentZ, (2*arraysize) * sizeof(float));
    cudaMalloc((creal32_T**)&subwave, (2*arraysize) * sizeof(float));
    cudaMalloc((creal32_T**)&recordZ, (2*arraysize) * sizeof(float));
    cudaMalloc((creal32_T**)&d_dldw1, (2 * imHs_sz.x * imHs_sz.y) * sizeof(float));
    cudaMalloc((creal32_T**)&d_dldw2, (2 * imLs_sz.x * imLs_sz.y) * sizeof(float));
    cudaMalloc((real32_T**)&mom_w1, (imHs_sz.x * imHs_sz.y) * sizeof(float));
    cudaMalloc((real32_T**)&mom_w2, (imLs_sz.x * imLs_sz.y) * sizeof(float));

    dim3 N_BLOCKS_L = {
        (unsigned) (imHs_sz.x + BLOCK_X - 1) / BLOCK_X,
        (unsigned) (imHs_sz.y + BLOCK_Y - 1) / BLOCK_Y,
        (unsigned) 1
    };
    dim3 N_THREADS = {BLOCK_X,BLOCK_Y,1};
    dim3 N_BLOCKS;

    int led_total = (int) imLs_sz.z; 

    N_BLOCKS = {
        (unsigned) (imLs_sz.x + BLOCK_X - 1) / BLOCK_X,
        (unsigned) (imLs_sz.y + BLOCK_Y - 1) / BLOCK_Y,
        (unsigned) 1
    };

    clear_data<<<N_BLOCKS_L, N_THREADS>>>(imHs_sz,mom_w1);
    imLs_sz.z = 1;
    clear_data<<<N_BLOCKS, N_THREADS>>>(imLs_sz,mom_w2);

    imLs_sz.z = batch_sz;
    for(int epoch = 0; epoch < numEpochs; ++ epoch){
        int remain = led_total;
        int curr = 0;
        while(remain > 0){
            remain = remain - batch_sz;
            FFT_sample_forward(imHs_sz.x,imHs_sz.y,d_wavefront1,d_dldw1);
            
            N_BLOCKS.z = (unsigned) min(curr + batch_sz - 1, (led_total-1) ) - curr + 1;
            
            getLatentZ(
                curr, 
                d_dldw1, 
                d_wavefront2, 
                d_ledIdx, 
                N_BLOCKS,
                imLs_sz, 
                imHs_sz, 
                subwave, 
                latentZ
            );

            differenceKernel<<<N_BLOCKS, N_THREADS>>>(
                curr,
                imLs_sz,
                pratio,
                d_obseY,
                latentZ,
                recordZ
            );
            
            backwardLatentZ(recordZ, imLs_sz, latentZ);

            clear_spectrum<<<N_BLOCKS_L, N_THREADS>>>(
                imHs_sz, 
                d_dldw1
            );

            deconvPIE<<<N_BLOCKS, N_THREADS>>>(
                curr,
                latentZ, 
                subwave,
                d_ledIdx, 
                d_wavefront2, 
                imLs_sz, 
                imHs_sz, 
                d_dldw1,
                d_dldw2
            );
            
            FFT_sample_backward(imHs_sz, d_dldw1);
            cudaDeviceSynchronize();

            N_BLOCKS.z = 1;
            RMSprop_step<<<N_BLOCKS_L, N_THREADS>>>(
                d_dldw1,
                beta,
                learning_rate,
                eps,
                imHs_sz,
                mom_w1,
                d_wavefront1
            );
            
            RMSprop_step<<<N_BLOCKS, N_THREADS>>>(
                d_dldw2,
                beta,
                learning_rate,
                eps,
                imLs_sz,
                mom_w2,
                d_wavefront2
            );

            setPupilConstrain<<<N_BLOCKS, N_THREADS>>>(
                d_pupil,
                0.2f,
                imLs_sz,
                d_wavefront2
            );

            curr = curr + batch_sz;
            cudaDeviceSynchronize();
        }
        if (epoch > 12){
            learning_rate = learning_rate * 0.5;
        }
    }

    // function output 
    plhs[0] = mxGPUCreateMxArrayOnGPU(wavefront1);
    plhs[1] = mxGPUCreateMxArrayOnGPU(wavefront2);

    // free memory
    cudaFree(subwave);
    cudaFree(recordZ);
    cudaFree(latentZ);    
    cudaFree(d_dldw1);
    cudaFree(d_dldw2);
    cudaFree(mom_w1);
    cudaFree(mom_w2);
    
    // free mxGPUArray
    mxGPUDestroyGPUArray(wavefront1);
    mxGPUDestroyGPUArray(wavefront2);
    mxGPUDestroyGPUArray(obseY);
    mxGPUDestroyGPUArray(ledIdx);
}