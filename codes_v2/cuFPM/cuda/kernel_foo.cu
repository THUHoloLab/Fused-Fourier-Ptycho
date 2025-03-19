#include "kernel_foo.cuh"
#include "addon.h"
#include <cooperative_groups.h>
#include <iostream>


namespace cg = cooperative_groups;

__global__ void getSubpupil(
    const unsigned z_offset,
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
    unsigned idz = block.group_index().z; 

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y) && (idz < imgSz.z);

    unsigned pixL_x = (unsigned) ledindex[z_offset + idz].x;
    unsigned pixL_y = (unsigned) ledindex[z_offset + idz].y;

    __shared__ creal32_T this_pupil[BLOCK_SIZE];

    unsigned tr = block.thread_rank();
    unsigned pix_id = idx * imgSz.y + idy;

    this_pupil[tr] = wavefront2[pix_id];

    block.sync();

    if (inside) {
        unsigned page_id = idz * (imgSz.x * imgSz.y);
        unsigned pix_id_large = (idx + pixL_x - 1) * imgSzL.y + (idy + pixL_y - 1);

        creal32_T temp = wavefront1[pix_id_large];

        subwave[pix_id + page_id] = temp;

        float a = temp.re;
        float b = temp.im;

        float c = this_pupil[tr].re;
        float d = this_pupil[tr].im;

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
    unsigned xIndex = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned yIndex = block.group_index().y * block.group_dim().y + block.thread_index().y; 
    unsigned zIndex = block.group_index().z; 

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

__global__ void deconvPIE(
    const unsigned z_offset,
    const creal32_T* __restrict__ x_record,
    const creal32_T* __restrict__ subwave,
    const int2* __restrict__ ledindex,
    const creal32_T* __restrict__ pupil,
    const dim3 imLs_sz,
    const dim3 imHs_sz, 
    // creal32_T* __restrict__ latentz,
    creal32_T* __restrict__ dldw,
    creal32_T* __restrict__ dldp
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 
    unsigned idz = block.group_index().z; 

    const bool inside = (idx < imLs_sz.x) && (idy < imLs_sz.y) && (idz < imLs_sz.z);    

    float a;
    float b;
    float c;
    float d;
    creal32_T tempX;
    creal32_T tempP;

    if (inside){
        unsigned pixL_x = (unsigned) ledindex[idz + z_offset].x;
        unsigned pixL_y = (unsigned) ledindex[idz + z_offset].y;

        unsigned idxZ = idz * (imLs_sz.x * imLs_sz.y);
        unsigned pix_id = idx * imLs_sz.y + idy;
        
        tempX = x_record[pix_id + idxZ];
        tempP = pupil[pix_id];

        a = tempX.re;
        b = tempX.im;

        c = tempP.re;
        d = tempP.im;

        tempX.re = a * c + b * d;
        tempX.im = c * b - a * d;

        // latentz[pix_id + idxZ] = tempX; 
        const bool inside_FP = ((idx + pixL_x - 1) < imHs_sz.x) && ((idy + pixL_y - 1) < imHs_sz.y);
        if(inside_FP){
            float *temp1 = (float *) dldw;
            unsigned pix_id_large = (idx + pixL_x - 1) * imHs_sz.y + (idy + pixL_y - 1); 
            atomicAdd(temp1 + pix_id_large * 2 + 0, tempX.re);
            atomicAdd(temp1 + pix_id_large * 2 + 1, tempX.im);
        }

        tempP = subwave[pix_id + idxZ];
        c = tempP.re;
        d = tempP.im;

        tempX.re = a * c + b * d;
        tempX.im = c * b - a * d;

        float *temp = (float *) dldp;
        atomicAdd(temp + 2 * pix_id + 0, tempX.re);
        atomicAdd(temp + 2 * pix_id + 1, tempX.im);
    }
    block.sync();
}

__global__ void stitch_spectrum(
    const unsigned z_offset,
    const creal32_T* __restrict__ latent,
    const int2* __restrict__ ledindex,
    const dim3 imgSz,
    const dim3 imgSzL, 
    creal32_T* __restrict__ dldw
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 
    unsigned idz = block.group_index().z; 

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y) && (idz < imgSz.z);

    if (inside) {
        unsigned pixL_x = (unsigned) ledindex[idz + z_offset].x;
        unsigned pixL_y = (unsigned) ledindex[idz + z_offset].y;

        unsigned page_id = idz * (imgSz.x * imgSz.y);
        unsigned pix_id = idx * imgSz.y + idy;

        float a = latent[pix_id + page_id].re;
        float b = latent[pix_id + page_id].im;

        const bool inside_FP = ((idx + pixL_x - 1) < imgSzL.x) && ((idy + pixL_y - 1) < imgSzL.y);

        if(inside_FP){
            float *temp = (float *) dldw;
            unsigned pix_id_large = (idx + pixL_x - 1) * imgSzL.y + (idy + pixL_y - 1); 
            atomicAdd(temp + pix_id_large * 2 + 0, a);
            atomicAdd(temp + pix_id_large * 2 + 1, b);
        }
    }
}   

// __global__ void ReduceAddpupil(
//     const creal32_T* __restrict__ x_record,
//     const creal32_T* __restrict__ subwave,
//     const dim3 imgSz,
//     creal32_T* __restrict__ dldp
// ){
//     auto block = cg::this_thread_block();
//     unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
//     unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 
//     unsigned idz = block.group_index().z; 

//     const bool inside = (idx < imgSz.x) && (idy < imgSz.y) && (idz < imgSz.z);

//     if (inside) {
//         unsigned idxZ = idz * (imgSz.x * imgSz.y);
//         unsigned pix_id = idx * imgSz.y + idy;

//         float a = x_record[pix_id + idxZ].re;
//         float b = x_record[pix_id + idxZ].im;

//         float c = subwave[pix_id + idxZ].re;
//         float d = subwave[pix_id + idxZ].im;

//         float pupil_re = a * c + b * d;
//         float pupil_im = c * b - a * d;

//         float *temp = (float *) dldp;

//         atomicAdd(temp + 2 * pix_id + 0, pupil_re);
//         atomicAdd(temp + 2 * pix_id + 1, pupil_im);
//     }
// }

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

__global__ void differenceKernel(
    const unsigned z_offset,
    const dim3 imLs_sz,
    const int pratio,
    const real32_T* __restrict__ img_Y,
    const creal32_T* __restrict__ latentz,
    creal32_T* __restrict__ out
){
    auto block = cg::this_thread_block();
    unsigned tr_x = block.thread_index().x;
    unsigned tr_y = block.thread_index().y;
    unsigned idx = block.group_index().x * block.group_dim().x + tr_x;
    unsigned idy = block.group_index().y * block.group_dim().y + tr_y; 
    // unsigned idz = block.group_index().z; 

    const bool inside = (idx < imLs_sz.x) && (idy < imLs_sz.y) && (block.group_index().z < imLs_sz.z);

    unsigned pix_A = idx * imLs_sz.y + idy;
    unsigned pix_X = (idx < (imLs_sz.x - 1))? ((idx + 1) * imLs_sz.y + idy) : idy;
    unsigned pix_Y = (idy < (imLs_sz.y - 1))? (idx * imLs_sz.y + idy + 1) : (idx * imLs_sz.y);
    unsigned page_id = block.group_index().z * (imLs_sz.x * imLs_sz.y);
    unsigned page_Y = (z_offset + block.group_index().z) * (imLs_sz.x * imLs_sz.y);

    __shared__ float blk_x[BLOCK_X + 1][BLOCK_Y];
    __shared__ float blk_y[BLOCK_X][BLOCK_Y + 1];

    creal32_T lat_A = latentz[pix_A + page_id];

    float ratio = (float) (imLs_sz.x * imLs_sz.y * pratio * pratio);

    float tempB = img_Y[pix_A + page_Y];
    float tempA = absC(lat_A,ratio) - tempB;

    // forward difference
    blk_x[tr_x + 1][tr_y] = sign(absC(latentz[pix_X + page_id],ratio) - img_Y[pix_X + page_Y] - tempA);
    blk_y[tr_x][tr_y + 1] = sign(absC(latentz[pix_Y + page_id],ratio) - img_Y[pix_Y + page_Y] - tempA);
    // wait untill all threads collect the pixels
    block.sync();
    
    unsigned test_p2 = 0;
    if (tr_x == 0){
        test_p2 = (idx == 0)? ((imLs_sz.x - 1) * imLs_sz.y + idy) : ((idx - 1) * imLs_sz.y + idy);
        tempB = absC(latentz[test_p2 + page_id],ratio) - img_Y[test_p2 + page_Y];
        blk_x[0][tr_y] = sign(tempA - tempB);
    }

    if (tr_y == 0){
        test_p2 = (idy == 0)? (idx * imLs_sz.y + imLs_sz.y - 1) : (idx * imLs_sz.y + idy - 1);
        tempB = absC(latentz[test_p2 + page_id],ratio) - img_Y[test_p2 + page_Y];
        blk_y[tr_x][0] = sign(tempA - tempB);
    }
    // wait untill all threads collect the pixels
    block.sync();

    // backward difference
    tempA = (blk_x[tr_x][tr_y] - blk_x[tr_x + 1][tr_y] + 
             blk_y[tr_x][tr_y] - blk_y[tr_x][tr_y + 1]) * ((float) pratio * pratio);

    float ang = atan2f(lat_A.im,lat_A.re);

    if (inside){
        out[pix_A + page_id].re = cosf(ang) * tempA;
        out[pix_A + page_id].im = sinf(ang) * tempA;
    }
}

__global__ void RMSprop_step(
    creal32_T * __restrict__ grad,
    const float beta,
    const float lr,
    const float eps,
    const dim3 imgSz,
    real32_T * __restrict__ mom2,
    creal32_T * __restrict__ wavefront
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    unsigned pixID = idx * imgSz.y + idy;

    __shared__ float mom_b[BLOCK_SIZE];
    unsigned tr = block.thread_rank();
    mom_b[tr] = mom2[pixID];
    block.sync();

    if(inside){
        creal32_T this_grad = grad[pixID];
        creal32_T this_wave =  wavefront[pixID];

        float mu = this_grad.re * this_grad.re + this_grad.im * this_grad.im;
    
        mu = beta * mom_b[tr] + (1 - beta) * mu;

        float sq_m = lr / (sqrtf(mu) + eps);

        this_wave.re = this_wave.re - this_grad.re * sq_m;
        this_wave.im = this_wave.im - this_grad.im * sq_m;

        wavefront[pixID] = this_wave;
        mom2[pixID] = mu;

        this_grad.re = 0;
        this_grad.im = 0;
        grad[pixID] = this_grad;
    }

}

__global__ void setPupilConstrain(
    const real32_T * pupil,
    const float am,
    const dim3 imgSz,
    creal32_T * wavefront
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    unsigned pixID = idx * imgSz.y + idy;
    if(inside){
        creal32_T this_pix = wavefront[pixID];
        real32_T this_pupil = pupil[pixID];
        float ang = atan2f(this_pix.im,this_pix.re);
        float abs = absC(this_pix, 1.0f);

        abs = this_pupil * max(min(abs, this_pupil + am), this_pupil - am);
        
        this_pix.re = cosf(ang) * abs;
        this_pix.im = sinf(ang) * abs;
        wavefront[pixID] = this_pix;
    }
}

__global__ void clear_data(
    const dim3 imgSzL,
    real32_T * data
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 

    const bool inside = (idx < imgSzL.x) && (idy < imgSzL.y);
    if(inside){
        unsigned pix_id = idx * imgSzL.y + idy;
        data[pix_id] = 0.0f;
    }
}