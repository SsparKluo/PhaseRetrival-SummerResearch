#include"FT_kernel.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>


cufffComplex** fourierFilter(double** image, int** imageFilter, int height, int width, int u0, int v0) {

	float* inputData = trans2Dto1D(image, height, width);

	cufftReal* input = inputData;

	cufftComplex* imageFFT = FFT();
	cufftComplex* shiftedFFT;
	cufftComplex* filterdShiftedFFT;
	cufftComplex* filterdShiftedFFTBaseband;
	cufftComplex* filterdFFTBaseband;
	cufftReal* filterdBaseband;

	cufftHandle FFTPlan;
	cufftHandle IFFTPlan;



	cufftPlan2d(&FFTPlan, height, width, CUFFT_R2C);
	cufftPlan2d(&IFFTPlan, height, (width / 2) + 1, CUFFT_C2R);

}


cufftComplex* filtering() {

}

 cufftComplex* fftShift(cufftComplex* inputFFT, int height, int width) {

}

 

float* trans2Dto1D(double** array2D, int height, int width) {
	float* array1D;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			array1D[i * width + j] = array2D[i][j];
		}
	}
	return array1D;
}

double** trans1Dto2D(float* array1D, int height, int width) {
	double** array2D;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			array2D[i][j] = array1D[i * width + j];
		}
	}
	return array2D;
}



