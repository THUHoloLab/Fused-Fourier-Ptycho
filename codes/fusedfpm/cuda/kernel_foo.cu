#include "kernel_foo.cuh"
#include <cufft.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>

namespace cg = cooperative_groups;

__global__ void getSubpupil(
    const creal32_T* __restrict__ wavefront1,
    const creal32_T* __restrict__ wavefront2,
    const int2* __restrict__ ledindex,
    const dim3 imgSz,
    const dim3 imgSzL,
    creal32_T* __restrict__ subwave,
    creal32_T* __restrict__ latentZ
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 
    unsigned idz = block.group_index().z * block.group_dim().z + block.thread_index().z; 

    
    const bool inside = (idx < imgSz.x) && (idy < imgSz.y) && (idz < imgSz.z);

    unsigned pixL_x = (unsigned) ledindex[idz].x;
    unsigned pixL_y = (unsigned) ledindex[idz].y;

    if (inside) {
        unsigned page_id = idz * (imgSz.x * imgSz.y);
        unsigned pix_id = idx * imgSz.y + idy;

        unsigned pix_id_large = (idx + pixL_x - 1) * imgSzL.y + (idy + pixL_y - 1);


        creal32_T temp = wavefront1[pix_id_large];

        subwave[pix_id + page_id] = temp;

        float a = temp.re;
        float b = temp.im;

        float c = wavefront2[pix_id].re;
        float d = wavefront2[pix_id].im;

        latentZ[pix_id + page_id].re = a * c - b * d; 
        latentZ[pix_id + page_id].im = a * d + c * b;
    }
}


__global__ void cufftShift_2D_kernel(
    creal32_T* __restrict__ data, 
    int N
){
    // 2D Slice & 1D Line
    int sLine = N;
    int sSlice = N * N;

    // Transformations Equations
    int sEq1 = (sSlice + sLine) / 2;
    int sEq2 = (sSlice - sLine) / 2;

    auto block = cg::this_thread_block();
    unsigned xIndex =
        block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned yIndex =
        block.group_index().y * block.group_dim().y + block.thread_index().y; 
    unsigned zIndex =
        block.group_index().z * block.group_dim().z + block.thread_index().z; 

    // Thread Index Converted into 1D Index
    int index = (yIndex * N) + xIndex + sSlice * zIndex;

    creal32_T regTemp;

    if (xIndex < N / 2)
    {
        if (yIndex < N / 2)
        {
            regTemp = data[index];

            // First Quad
            data[index] = data[index + sEq1];

            // Third Quad
            data[index + sEq1] = regTemp;
        }
    }
    else
    {
        if (yIndex < N / 2)
        {
            regTemp = data[index];

            // Second Quad
            data[index] = data[index + sEq2];

            // Fourth Quad
            data[index + sEq2] = regTemp;
        }
    }
}

__global__ void getGradients(
    const real32_T* __restrict__ x1,
    const real32_T* __restrict__ x2,
    const dim3 imgSz,
    real32_T* __restrict__ ox,
    real32_T* __restrict__ oy
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 
    unsigned idz = block.group_index().z * block.group_dim().z + block.thread_index().z; 

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y) && (idz < imgSz.z);

    if (inside) {
        unsigned idxZ = idz * (imgSz.x * imgSz.y);
        unsigned pix_id;
        
        unsigned pix_id_x2;
        unsigned pix_id_y2;
        float x1_x2;

        pix_id = idx * imgSz.y + idy + idxZ;

        if (idx == (imgSz.x - 1)){
            pix_id_x2 = idy + idxZ;
        }else{
            pix_id_x2 = (idx + 1) * imgSz.y + idy + idxZ;
        }

        if (idy == (imgSz.y - 1)){
            pix_id_y2 = idx * imgSz.y + idxZ;
        }else{
            pix_id_y2 = idx * imgSz.y + (idy + 1) + idxZ;
        }
    
        x1_x2 = x1[pix_id] - x2[pix_id];
        float temp_ox = (x1[pix_id_x2] - x2[pix_id_x2]) - x1_x2;
        float temp_oy = (x1[pix_id_y2] - x2[pix_id_y2]) - x1_x2;

        float temp = sqrtf(temp_ox * temp_ox + temp_oy * temp_oy);

        ox[pix_id] = temp_ox / (temp + 0.001);
        oy[pix_id] = temp_oy / (temp + 0.001);
    }
}

__global__ void differenceMap(
    const real32_T* __restrict__ ox,
    const real32_T* __restrict__ oy,
    const dim3 imgSz,
    real32_T* __restrict__ diffmap
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 
    unsigned idz = block.group_index().z * block.group_dim().z + block.thread_index().z; 

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y) && (idz < imgSz.z);

    if (inside) {
        unsigned idxZ = idz * (imgSz.x * imgSz.y);
        unsigned pix_id;
        
        unsigned pix_id_x2;
        unsigned pix_id_y2;

        pix_id = idx * imgSz.y + idy + idxZ;

        if (idx == 0){
            pix_id_x2 = (imgSz.x - 1) * imgSz.y + idy + idxZ;
        }else{
            pix_id_x2 = (idx - 1) * imgSz.y + idy + idxZ;
        }

        if (idy == 0){
            pix_id_y2 = idx * imgSz.y + (imgSz.y - 1) + idxZ;
        }else{
            pix_id_y2 = idx * imgSz.y + (idy - 1) + idxZ;
        }
        diffmap[pix_id] = (ox[pix_id_x2] - ox[pix_id]) + (oy[pix_id_y2] - oy[pix_id]);
    }
}

__global__ void newX(
    const real32_T* __restrict__ diffmap,
    const dim3 imgSz,
    const int pratio,
    creal32_T* __restrict__ latentZ
){  
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 
    unsigned idz = block.group_index().z * block.group_dim().z + block.thread_index().z; 

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y) && (idz < imgSz.z);

    if (inside){
        unsigned page_id = idz * (imgSz.x * imgSz.y);
        unsigned pix_id = idx * imgSz.y + idy;

        creal32_T tempZ = latentZ[pix_id + page_id];

        float pratio2 = (float) pratio * pratio;
        float ang = atan2f(tempZ.im,tempZ.re);

        float amp = diffmap[pix_id + page_id];

        float a = cosf(ang) * amp * pratio2;
        float b = sinf(ang) * amp * pratio2;

        latentZ[pix_id + page_id].re = a;
        latentZ[pix_id + page_id].im = b;
    }
}

__global__ void deconvPIE(
    const creal32_T* __restrict__ latentz_record,
    const creal32_T* __restrict__ pupil,
    const dim3 imLs_sz,
    creal32_T* __restrict__ latentz
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 
    unsigned idz = block.group_index().z * block.group_dim().z + block.thread_index().z; 

    const bool inside = (idx < imLs_sz.x) && (idy < imLs_sz.y) && (idz < imLs_sz.z);    
    if (inside){
        unsigned idxZ = idz * (imLs_sz.x * imLs_sz.y);
        unsigned pix_id = idx * imLs_sz.y + idy;
        
        creal32_T tempX = latentz_record[pix_id + idxZ];
        creal32_T tempP = pupil[pix_id];

        float a = tempX.re;
        float b = tempX.im;

        float c = tempP.re;
        float d = tempP.im;

        latentz[pix_id + idxZ].re = a * c + b * d; 
        latentz[pix_id + idxZ].im = c * b - a * d;
    }
}

__global__ void getAmplitude(
    creal32_T* __restrict__ latentz,
    const dim3 imLs_sz,
    const int pratio,
    real32_T* __restrict__ amplitude
){
    auto block = cg::this_thread_block();
    unsigned idx =
        block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy =
        block.group_index().y * block.group_dim().y + block.thread_index().y; 
    unsigned idz =
        block.group_index().z * block.group_dim().z + block.thread_index().z; 

    const bool inside = (idx < imLs_sz.x) && (idy < imLs_sz.y) && (idz < imLs_sz.z);    
    if (inside){
        unsigned idxZ = idz * (imLs_sz.x * imLs_sz.y);
        unsigned pix_id = idx * imLs_sz.y + idy + idxZ;

        float ratio = (float) (imLs_sz.x * imLs_sz.y * pratio * pratio);

        float a = latentz[pix_id].re / ratio;
        float b = latentz[pix_id].im / ratio;

        float o = sqrtf(a * a + b * b);

        amplitude[pix_id] = o;
        latentz[pix_id].re = a;
        latentz[pix_id].im = b;
    }
}

__global__ void ReduceAddpupil(
    const creal32_T* __restrict__ x_record,
    const creal32_T* __restrict__ subwave,
    const dim3 imgSz,
    creal32_T* __restrict__ dldw
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 
    unsigned idz = block.group_index().z * block.group_dim().z + block.thread_index().z; 

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y) && (idz < imgSz.z);

    if (inside) {
        unsigned idxZ = idz * (imgSz.x * imgSz.y);
        unsigned pix_id = idx * imgSz.y + idy;

        float a = x_record[pix_id + idxZ].re;
        float b = x_record[pix_id + idxZ].im;

        float c = subwave[pix_id + idxZ].re;
        float d = subwave[pix_id + idxZ].im;

        float pupil_re = a * c + b * d;
        float pupil_im = c * b - a * d;

        float *temp = (float *) dldw;

        atomicAdd(temp + 2 * pix_id,     pupil_re);
        atomicAdd(temp + 2 * pix_id + 1, pupil_im);
    }
}

__global__ void stitch_spectrum(
    const creal32_T* __restrict__ latent,
    const int2* __restrict__ ledindex,
    const dim3 imgSz,
    const dim3 imgSzL, 
    creal32_T* __restrict__ dldw
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 
    unsigned idz = block.group_index().z * block.group_dim().z + block.thread_index().z; 

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y) && (idz < imgSz.z);

    unsigned pixL_x = (unsigned) ledindex[idz].x;
    unsigned pixL_y = (unsigned) ledindex[idz].y;

    if (inside) {
        unsigned page_id = idz * (imgSz.x * imgSz.y);
        unsigned pix_id = idx * imgSz.y + idy;

        unsigned pix_id_large = (idx + pixL_x - 1) * imgSzL.y + (idy + pixL_y - 1);

        float a = latent[pix_id + page_id].re;
        float b = latent[pix_id + page_id].im;

        float *temp = (float *) dldw;
        atomicAdd(temp + pix_id_large * 2,     a);
        atomicAdd(temp + pix_id_large * 2 + 1, b);
    }
}

__global__ void clear_spectrum(
    const dim3 imgSzL,
    creal32_T* __restrict__ spect
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 

    const bool inside = (idx < imgSzL.x) && (idy < imgSzL.y);
    if(inside){
        unsigned pix_id = idx * imgSzL.y + idy;
        spect[pix_id].re = 0.0;
        spect[pix_id].im = 0.0;
    }
}


__global__ void ifftCorrection(
    creal32_T* __restrict__ spectrum,
    const dim3 imHs_sz
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 

    const bool inside = (idx < imHs_sz.x) && (idy < imHs_sz.y);    
    if (inside){
        unsigned pix_id = idx * imHs_sz.y + idy;

        float ratio = (float) (imHs_sz.x * imHs_sz.y);
        float a = spectrum[pix_id].re;
        float b = spectrum[pix_id].im;
        spectrum[pix_id].re = a / ratio;
        spectrum[pix_id].im = b / ratio;
    }
}