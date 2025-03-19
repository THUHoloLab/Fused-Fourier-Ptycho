#include <cuda_runtime.h>
#include <cufft.h>

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
