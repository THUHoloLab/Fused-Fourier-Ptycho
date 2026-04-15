#include "addon.h"
#include <cooperative_groups.h>
#include <iostream>

inline __device__ float absC(const creal32_T in, const float ratio){
    float out;
    out = sqrtf(in.re * in.re + in.im * in.im) / ratio;
    return out;
}

inline __device__ float sign(const float in){
    float out;
    out = copysignf(1.0f,in);
    return out;
}

namespace cg = cooperative_groups;

__global__ void fused_getSubpupil_shifted(
    const unsigned z_offset,
    const creal32_t* __restrict__ wavefront1,
    const creal32_t* __restrict__ wavefront2,
    const int2* __restrict__ ledindex,
    const dim3 imgSz,
    const dim3 imgSzL,
    creal32_t* __restrict__ subwave,
    creal32_t* __restrict__ latentZ
){
    const unsigned idy = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idx = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned idz = blockIdx.z;

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y) && (idz < imgSz.z);
    if(!inside){
        return;
    }
   
    const unsigned pixL_x = (unsigned) ledindex[z_offset + idz].x;
    const unsigned pixL_y = (unsigned) ledindex[z_offset + idz].y;
    const unsigned page_id = idz * (imgSz.x * imgSz.y);
    const unsigned pix_id = idx * imgSz.y + idy;

    const unsigned crop_x = idx + pixL_x - 1;
    const unsigned crop_y = idy + pixL_y - 1;
    const unsigned shifted_x = (crop_x + imgSzL.x / 2) % imgSzL.x;
    const unsigned shifted_y = (crop_y + imgSzL.y / 2) % imgSzL.y;
    const unsigned pix_id_large = shifted_x * imgSzL.y + shifted_y;

    creal32_t temp = wavefront1[pix_id_large];
    subwave[pix_id + page_id] = temp;
    creal32_t pupil_val = wavefront2[pix_id];

    const float a = temp.re;
    const float b = temp.im;
    const float c = pupil_val.re;
    const float d = pupil_val.im;

    creal32_t latent;
    latent.re = a * c - b * d;
    latent.im = a * d + c * b;

    const unsigned shifted_pix_id = ((idx + imgSz.x / 2) % imgSz.x) * imgSz.y
                                  + ((idy + imgSz.y / 2) % imgSz.y);
    latentZ[shifted_pix_id + page_id] = latent;
}

__global__ void fused_deconvPIE_shifted(
    const unsigned z_offset,
    const creal32_t* __restrict__ x_record,
    const creal32_t* __restrict__ subwave,
    const int2* __restrict__ ledindex,
    const creal32_t* __restrict__ pupil,
    const dim3 imLs_sz,
    const dim3 imHs_sz,
    creal32_t* __restrict__ dldw,
    creal32_t* __restrict__ dldp
){
    const unsigned idy = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idx = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned idz = blockIdx.z;

    const bool inside = (idx < imLs_sz.x) && (idy < imLs_sz.y) && (idz < imLs_sz.z);
    if(!inside){
        return;
    }

    unsigned pixL_x = (unsigned) ledindex[idz + z_offset].x;
    unsigned pixL_y = (unsigned) ledindex[idz + z_offset].y;

    unsigned idxZ = idz * (imLs_sz.x * imLs_sz.y);
    unsigned pix_id = idx * imLs_sz.y + idy;
    unsigned shifted_pix_id = ((idx + imLs_sz.x / 2) % imLs_sz.x) * imLs_sz.y
                                + ((idy + imLs_sz.y / 2) % imLs_sz.y);

    creal32_t tempX = x_record[shifted_pix_id + idxZ];
    creal32_t tempP = pupil[pix_id];

    float a = tempX.re;
    float b = tempX.im;
    float c = tempP.re;
    float d = tempP.im;

    creal32_t gradW;
    gradW.re = a * c + b * d;
    gradW.im = c * b - a * d;

    unsigned crop_x = idx + pixL_x - 1;
    unsigned crop_y = idy + pixL_y - 1;
    unsigned shifted_x = (crop_x + imHs_sz.x / 2) % imHs_sz.x;
    unsigned shifted_y = (crop_y + imHs_sz.y / 2) % imHs_sz.y;

    float *temp1 = (float *) dldw;
    unsigned pix_id_large = shifted_x * imHs_sz.y + shifted_y;
    const float ifft_ratio = 1.0f / (float) (imHs_sz.x * imHs_sz.y);
    atomicAdd(temp1 + pix_id_large * 2 + 0, gradW.re * ifft_ratio);
    atomicAdd(temp1 + pix_id_large * 2 + 1, gradW.im * ifft_ratio);

    tempP = subwave[pix_id + idxZ];
    c = tempP.re;
    d = tempP.im;

    creal32_t gradP;
    gradP.re = a * c + b * d;
    gradP.im = c * b - a * d;

    float *temp = (float *) dldp;
    atomicAdd(temp + 2 * pix_id + 0, gradP.re);
    atomicAdd(temp + 2 * pix_id + 1, gradP.im);
}

__global__ void differenceKernel(
    const unsigned z_offset,
    const dim3 imLs_sz,
    const int pratio,
    const real32_T* __restrict__ img_Y,
    const creal32_t* __restrict__ latentz,
    creal32_t* __restrict__ out
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

    creal32_t lat_A = latentz[pix_A + page_id];

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

__global__ void differenceKernel_acc(
    const unsigned z_offset,
    const dim3 imLs_sz,
    const int pratio,
    const real32_T* __restrict__ img_Y,
    const creal32_t* __restrict__ latentz,
    creal32_t* __restrict__ out
){
    const unsigned tr_x = threadIdx.x;
    const unsigned tr_y = threadIdx.y;
    const unsigned idx = blockIdx.x * blockDim.x + tr_x;
    const unsigned idy = blockIdx.y * blockDim.y + tr_y;
    const unsigned idz = blockIdx.z;

    const bool inside = (idx < imLs_sz.x) && (idy < imLs_sz.y) && (idz < imLs_sz.z);

    __shared__ float blk_x[BLOCK_X + 1][BLOCK_Y];
    __shared__ float blk_y[BLOCK_X][BLOCK_Y + 1];

    const unsigned plane = imLs_sz.x * imLs_sz.y;
    const unsigned pix_A = idx * imLs_sz.y + idy;
    const unsigned pix_X = (idx < (imLs_sz.x - 1)) ? ((idx + 1) * imLs_sz.y + idy) : idy;
    const unsigned pix_Y = (idy < (imLs_sz.y - 1)) ? (idx * imLs_sz.y + idy + 1) : (idx * imLs_sz.y);
    const unsigned page_id = idz * plane;
    const unsigned page_Y = (z_offset + idz) * plane;

    const float ratio = (float) (plane * pratio * pratio);
    const float grad_scale = (float) (pratio * pratio);

    creal32_t lat_A;
    lat_A.re = 0.0f;
    lat_A.im = 0.0f;
    float res_A = 0.0f;

    if (inside) {
        lat_A = latentz[pix_A + page_id];
        res_A = absC(lat_A, ratio) - img_Y[pix_A + page_Y];
        blk_x[tr_x + 1][tr_y] = sign((absC(latentz[pix_X + page_id], ratio) - img_Y[pix_X + page_Y]) - res_A);
        blk_y[tr_x][tr_y + 1] = sign((absC(latentz[pix_Y + page_id], ratio) - img_Y[pix_Y + page_Y]) - res_A);
    } else {
        blk_x[tr_x + 1][tr_y] = 0.0f;
        blk_y[tr_x][tr_y + 1] = 0.0f;
    }
    __syncthreads();

    if (inside && tr_x == 0){
        const unsigned left_pix = (idx == 0) ? ((imLs_sz.x - 1) * imLs_sz.y + idy) : ((idx - 1) * imLs_sz.y + idy);
        const float res_left = absC(latentz[left_pix + page_id], ratio) - img_Y[left_pix + page_Y];
        blk_x[0][tr_y] = sign(res_A - res_left);
    } else if (tr_x == 0) {
        blk_x[0][tr_y] = 0.0f;
    }

    if (inside && tr_y == 0){
        const unsigned up_pix = (idy == 0) ? (idx * imLs_sz.y + imLs_sz.y - 1) : (idx * imLs_sz.y + idy - 1);
        const float res_up = absC(latentz[up_pix + page_id], ratio) - img_Y[up_pix + page_Y];
        blk_y[tr_x][0] = sign(res_A - res_up);
    } else if (tr_y == 0) {
        blk_y[tr_x][0] = 0.0f;
    }
    __syncthreads();

    if (inside) {
        const float tempA = (
            blk_x[tr_x][tr_y] - blk_x[tr_x + 1][tr_y] +
            blk_y[tr_x][tr_y] - blk_y[tr_x][tr_y + 1]
        ) * grad_scale;

        const float mag = sqrtf(lat_A.re * lat_A.re + lat_A.im * lat_A.im);
        if (mag > 0.0f) {
            const float inv_mag = 1.0f / mag;
            out[pix_A + page_id].re = lat_A.re * inv_mag * tempA;
            out[pix_A + page_id].im = lat_A.im * inv_mag * tempA;
        } else {
            out[pix_A + page_id].re = 0.0f;
            out[pix_A + page_id].im = 0.0f;
        }
    }
}

__global__ void RMSprop_step_both(
    creal32_t * __restrict__ grad1,
    creal32_t * __restrict__ grad2,
    const float beta,
    const float lr,
    const float eps,
    const dim3 imgSz1,
    const dim3 imgSz2,
    real32_T * __restrict__ mom2_1,
    real32_T * __restrict__ mom2_2,
    creal32_t * __restrict__ wavefront1,
    creal32_t * __restrict__ wavefront2
){
    unsigned idy = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx = blockIdx.y * blockDim.y + threadIdx.y;

    const bool inside1 = (idx < imgSz1.x) && (idy < imgSz1.y);
    const bool inside2 = (idx < imgSz2.x) && (idy < imgSz2.y);

    unsigned pixID1 = idx * imgSz1.y + idy;
    unsigned pixID2 = idx * imgSz2.y + idy;

    if (inside1){
        creal32_t this_grad = grad1[pixID1];
        creal32_t this_wave = wavefront1[pixID1];

        float mu = this_grad.re * this_grad.re + this_grad.im * this_grad.im;
        mu = beta * mom2_1[pixID1] + (1.0f - beta) * mu;

        float sq_m = lr / (sqrtf(mu) + eps);

        this_wave.re -= this_grad.re * sq_m;
        this_wave.im -= this_grad.im * sq_m;

        wavefront1[pixID1] = this_wave;
        mom2_1[pixID1] = mu;

        this_grad.re = 0.0f;
        this_grad.im = 0.0f;
        grad1[pixID1] = this_grad;
    }

    if (inside2){
        creal32_t this_grad = grad2[pixID2];
        creal32_t this_wave = wavefront2[pixID2];

        float mu = this_grad.re * this_grad.re + this_grad.im * this_grad.im;
        mu = beta * mom2_2[pixID2] + (1.0f - beta) * mu;

        float sq_m = lr / (sqrtf(mu) + eps);

        this_wave.re -= this_grad.re * sq_m;
        this_wave.im -= this_grad.im * sq_m;

        wavefront2[pixID2] = this_wave;
        mom2_2[pixID2] = mu;

        this_grad.re = 0.0f;
        this_grad.im = 0.0f;
        grad2[pixID2] = this_grad;
    }
}

__global__ void constrained_RMSprop_both(
    creal32_t * __restrict__ grad1,
    creal32_t * __restrict__ grad2,
    const float beta,
    const float lr,
    const float eps,
    const dim3 imgSz1,
    const dim3 imgSz2,
    real32_T * __restrict__ mom2_1,
    real32_T * __restrict__ mom2_2,
    creal32_t * __restrict__ wavefront1,
    creal32_t * __restrict__ wavefront2,
    const real32_T * __restrict__ pupil,
    const float am
){
    const unsigned idy = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idx = blockIdx.y * blockDim.y + threadIdx.y;

    const bool inside1 = (idx < imgSz1.x) && (idy < imgSz1.y);
    const bool inside2 = (idx < imgSz2.x) && (idy < imgSz2.y);

    const unsigned pixID1 = idx * imgSz1.y + idy;
    const unsigned pixID2 = idx * imgSz2.y + idy;

    if (inside1){
        creal32_t this_grad = grad1[pixID1];
        creal32_t this_wave = wavefront1[pixID1];

        float mu = this_grad.re * this_grad.re + this_grad.im * this_grad.im;
        mu = beta * mom2_1[pixID1] + (1.0f - beta) * mu;

        const float sq_m = lr / (sqrtf(mu) + eps);

        this_wave.re -= this_grad.re * sq_m;
        this_wave.im -= this_grad.im * sq_m;

        wavefront1[pixID1] = this_wave;
        mom2_1[pixID1] = mu;

        this_grad.re = 0.0f;
        this_grad.im = 0.0f;
        grad1[pixID1] = this_grad;
    }

    if (inside2){
        creal32_t this_grad = grad2[pixID2];
        creal32_t this_wave = wavefront2[pixID2];

        float mu = this_grad.re * this_grad.re + this_grad.im * this_grad.im;
        mu = beta * mom2_2[pixID2] + (1.0f - beta) * mu;

        const float sq_m = lr / (sqrtf(mu) + eps);

        this_wave.re -= this_grad.re * sq_m;
        this_wave.im -= this_grad.im * sq_m;

        const real32_T this_pupil = pupil[pixID2];
        const float ang = atan2f(this_wave.im, this_wave.re);
        float abs = absC(this_wave, 1.0f);
        abs = this_pupil * max(min(abs, this_pupil + am), this_pupil - am);

        this_wave.re = cosf(ang) * abs;
        this_wave.im = sinf(ang) * abs;

        wavefront2[pixID2] = this_wave;
        mom2_2[pixID2] = mu;

        this_grad.re = 0.0f;
        this_grad.im = 0.0f;
        grad2[pixID2] = this_grad;
    }
}

__global__ void setPupilConstrain(
    const real32_T * pupil,
    const float am,
    const dim3 imgSz,
    creal32_t * wavefront
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    unsigned pixID = idx * imgSz.y + idy;
    if(inside){
        creal32_t this_pix = wavefront[pixID];
        real32_T this_pupil = pupil[pixID];
        float ang = atan2f(this_pix.im,this_pix.re);
        float abs = absC(this_pix, 1.0f);

        abs = this_pupil * max(min(abs, this_pupil + am), this_pupil - am);
        
        this_pix.re = cosf(ang) * abs;
        this_pix.im = sinf(ang) * abs;
        wavefront[pixID] = this_pix;
    }
}
