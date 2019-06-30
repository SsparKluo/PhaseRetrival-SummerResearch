#include<iostream>
#include<fstream>
#include"tiffio.h"
#include<math.h>
#include<algorithm>
#include"FT_kernel.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>


typedef unsigned char BYTE;

using namespace std;

class image
{
public:
	image() {

		float tempMax = 0, tempMin = 1;

		cin >> filename;
		TIFF* tif = TIFFOpen(filename, "r");
		if (!tif) {
			cout << "File open error!" << endl;
		}
		else {
			TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
			TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
			imagePixels = height * width;

			imageData = new float*[height];

			rawImage = new uint8 * [height];
			for (int i = 0; i < height; i++) {
				rawImage[i] = new uint8[width];
				imageData[i] = new float[width];
				TIFFReadScanline(tif, rawImage[i], i);
			}

			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					imageData[i][j] = ((float)rawImage[i][j] / 256);
					if (tempMax < imageData[i][j])
						tempMax = imageData[i][j];
					if (tempMin > imageData[i][j])
						tempMin = imageData[i][j];
				}
			}
			maxIntensity = tempMax;
			minIntensity = tempMin;
		}
		TIFFClose(tif);
	}

	int getInfo(char i) {
		if (i == 'w')
			return width;
		if (i == 'h')
			return height;
		if (i == 'p')
			return imagePixels;
		return 0;
	}

	int2 findPosition(double target) {

	}



	cufftReal** imageData;
	cufftComplex** FFTData;
	cufftReal** absFFTData;

private:
	uint8** rawImage;
	char filename[50];
	unsigned int imagePixels;
	int width;
	int height;
	float maxIntensity;
	float minIntensity;
};



double findMax(double** inputData, int w, int h);
double findMin(double** inputData, int w, int h);
double** arrayAbs(double** inputData, int w, int h);
int* findPosition(double** inputData, int w, int h, double target);
cufftComplex** fourierFilter(double** image, int** imageFilter, int height, int width, int u0, int v0);

int main() {

	const double cambits = 8;
	const double lambda = 0.632;

	cout << "Please input the address and filename of the calibration image(with .tif): ";
	image calib;
	
	cout << "Please input the address and filename of the testing image(with .tif): ";
	image test;

	int width = calib.getInfo('w');
	int height = calib.getInfo('h');

	int** imageFilter = new int* [height];
	for (int u = 0; u < height; u++) {
		imageFilter[u] = new int[width];
	}

	double** FFTCalib;// wait for editting;
	double** absFFTCalib = arrayAbs(FFTCalib, width, height);// wait for editting;

	int2 position = findPosition(calibImage, width, height, findMax(absFFTCalib, width, height)); // wait for editting;

	int u0 = position.x - (int)(height / 2);
	int	v0 = int(position.y + (int)(width * 1.6 / 3) - 1 - width / 2);
	int D0 = 80;


	for (int u = 0; u < height; u++) {
		for (int v = 0; v < width; v++) {
			double cutoff = sqrt((u - (height / 2 + u0)) ^ 2 + (v - (width / 2 + v0)) ^ 2);
			if (cutoff <= D0)
				imageFilter[u][v] = 1;
			else
				imageFilter[u][v] = 0;
		}
	}

	double** check1 = new double* [height];
	for (int i = 0; i < height; i++) {
		check1[i] = new double[width];
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			check1[i][j] = FFTCalib[i][j] * imageFilter[i][j]; // wait for editting
		}
	}



	return 0;

}

/*
	abs计算可以采用cuda加速。
*/

cufftComplex** fourierFilter(double** image, int** imageFilter, int height, int width, int u0, int v0) {

	cufftReal* input = inputData;

	cufftComplex* imageFFT;
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

__global__ void vectorAdd(const float* a, const float* b, float* c, int numElements) {
	int i = blockDim.x * blockIdx.x * threadIdx.x;
	if (i < numElements) {
		c[i] = a[i] + b[i];
	}
}

__global__ void vectorminus(const float* a, const float* b, float* c, int numElements) {
	int i = blockDim.x * blockIdx.x * threadIdx.x;
	if (i < numElements) {
		c[i] = a[i] - b[i];
	}
}

__global__ void vectorNumMultiple(const float* a, const float* b, float* c, int numElements) {
	int i = blockDim.x * blockIdx.x * threadIdx.x;
	if (i < numElements) {
		c[i] = a[i] * b[i];
	}
}

__global__ void vectorNumdevide(const float* a, const float* b, float* c, int numElements) {
	int i = blockDim.x * blockIdx.x * threadIdx.x;
	if (i < numElements) {
		c[i] = a[i] / b[i];
	}
}