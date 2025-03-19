#include "mex.h"
#include "addon.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void learning_step(
    const creal32_T * grad,
    const float beta,
    const dim3 imgSz,
    const float lr,
    real32_T * mom2,
    creal32_T wavefront
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);

    __shared__ float mom_b[BLOCK_SIZE];
    unsigned tr = block.thread_rank();
    unsigned pixID = idx * imgSz.y + idy;
    
    mom_b[tr] = mom2[pixID];

    block.sync();

    if(inside){
        float a = grad[pixID].re;
        float b = grad[pixID].im;
        float m = a * a + b * b;
        

        creal32_T this_wave =  wavefront[pixID];

        mom2[pixID] = beta * mom_b[tr] + (1 - beta) * m;

        float sq_m = 1 / (sqrtf(mom2[pixID]) + 0.0000001);
        float2 update = {a * sq_m, b * sq_m};

        this_wave.re = this_wave.re - lr * update.x;
        this_wave.im = this_wave.im - lr * update.y;

        wavefront[pixID] = this_wave;
    }

}


