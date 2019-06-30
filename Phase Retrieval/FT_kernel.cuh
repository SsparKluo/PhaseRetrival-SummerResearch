#ifndef FT_KERNEL_CUH
#define FT_KERNEL_CUH

#include"FT_kernel.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>

double** fourierFilter(double** image, int** imageFilter, int height, int width, int u0, int v0);

#endif // !FT_KERNEL_CUH
