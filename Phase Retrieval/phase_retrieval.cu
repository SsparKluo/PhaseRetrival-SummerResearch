#include<iostream>
#include<fstream>
#include"tiffio.h"
#include<cmath>
#include<algorithm>
#include"FT_kernel.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include<cuda.h>
#include"myUnwrap.h"
#include"matrix.h"


#define lambda 0.632
#define pi 3.1415926
#define TILE_WIDTH 16

typedef unsigned char BYTE;

using namespace std;

class image {
public:
	char filename[50];
	float maxIntensity, minIntensity;
	int height, width;
	int imagePixels;
	uint8** rawImage;
	float2* imageData;
	int2 fftMaxPosition;
	float2* filteredBaseband;
	float* imageFilter;

	image() {
		rawImage = (uint8 * *)malloc(sizeof(uint8*) * 960);
		for (int i = 0; i < 960; i++) {
			rawImage[i] = (uint8*)malloc(sizeof(uint8) * 1280);
		}
		filteredBaseband = (float2*)malloc(sizeof(float2) * 1280 * 960);
		imageFilter = new float[960 * 1280];
	}
};

bool getImageInfo(image* targetImage);// Used for get the useful data in tiff image, and return false when the filename of the target image is invalid.
float* phaseRetrieval(image* calibImage, image* testImage);
void fourierFilterForCalib(image* calibImage);
int2 findMaxPoint(float* input);
void imageFileWrite(float* input, char* filename);
void complexWrite(const char* title, float2* input, int width, const char* filename);
void realWrite(const char* title, float* input, int width, const char* filename);
void errorHandle(int input);
float* phaseUnwrapping(float* wMatrix);

__global__ void vectorAdd(float* a, float* b, float* c, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements)
		c[i] = a[i] + b[i];
}

__global__ void vectorNumMultiple(float* a, float* b, float* c, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements)
		c[i] = a[i] * b[i];
}

__global__ void numMultipleForComplex(float2* a, float* b, float2* c, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x + threadIdx.y * gridDim.x;
	if (i < numElements) {
		c[i].x = a[i].x * b[i];
		c[i].y = a[i].y * b[i];
	}
}

__global__ void vectorMatDivide(float2* dividend, float2* divisor, float2* output, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		output[i].x = dividend[i].x / divisor[i].x;
		output[i].y = dividend[i].y / divisor[i].y;
	}
}


__global__ void vectorNumdivide(float2* dividend, int divisor, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		dividend[i].x = dividend[i].x / divisor;
		dividend[i].y = dividend[i].y / divisor;
	}
}

__global__ void getAbsOfComplexMatric(cufftComplex* input, cufftReal* output, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements)
		output[i] = sqrt(input[i].x * input[i].x + input[i].y * input[i].y);
}

__global__ void createFilter(int padding, int2 maxPoint, float2* input, float2* output, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		if ((x - 639) * (x - 639) + (y - 479) * (y - 479) <= padding * padding) {
			output[i].x = input[i].x;
			output[i].y = input[i].y;
		}
		else {
			output[i].x = 0;
			output[i].y = 0;
		}
	}
}

__global__ void FFTShift2D(cufftComplex* input, cufftComplex* output, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	int halfX = 640;
	int halfY = 480;
	if (i < numElements) {
		if (y < halfY) {
			if (x < halfX) {
				output[i].x = input[(x + halfX) * blockDim.y + y + halfY].x;
				output[i].y = input[(x + halfX) * blockDim.y + y + halfY].y;
			}
			else {
				output[i].x = input[(x - halfX) * blockDim.y + y + halfY].x;
				output[i].y = input[(x - halfX) * blockDim.y + y + halfY].y;
			}
		}
		else {
			if (x < halfX) {
				output[i].x = input[(x + halfX) * blockDim.y + y - halfY].x;
				output[i].y = input[(x + halfX) * blockDim.y + y - halfY].y;
			}
			else {
				output[i].x = input[(x - halfX) * blockDim.y + y - halfY].x;
				output[i].y = input[(x - halfX) * blockDim.y + y - halfY].y;
			}
		}
	}
}

__global__ void IFFTShift2D(cufftComplex* input, cufftComplex* output, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	int halfY = blockDim.y / 2;
	int halfX = gridDim.x / 2;
	int preX, preY;
	if (i < numElements) {
		if (y < halfY) {
			if (x < halfX) {
				output[i].x = input[(x + halfX) * blockDim.y + y + halfY].x;
				output[i].y = input[(x + halfX) * blockDim.y + y + halfY].y;
			}
			else {
				output[i].x = input[(x - halfX) * blockDim.y + y + halfY].x;
				output[i].y = input[(x - halfX) * blockDim.y + y + halfY].y;
			}
		}
		else {
			if (x < halfX) {
				output[i].x = input[(x + halfX) * blockDim.y + y - halfY].x;
				output[i].y = input[(x + halfX) * blockDim.y + y - halfY].y;
			}
			else {
				output[i].x = input[(x - halfX) * blockDim.y + y - halfY].x;
				output[i].y = input[(x - halfX) * blockDim.y + y - halfY].y;
			}
		}
	}
}

__global__ void circShift2D(cufftComplex* input, int2 maxPoint, cufftComplex* output, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	int preX = x - 639 + maxPoint.x;
	int preY = y - 479 + maxPoint.y;
	if (i < numElements) {
		/*
		if (preX < 0)
			preX = 1280 + preX;
		if (preX >= 1280)
			preX = preX - 1280;
		if (preY < 0 || preY > 480) {
			preX = 1279 - preX;
			if (preY < 0)
				preY = -preY;
			else
				preY = 480 - preY;
		}
		else {
			output[y + x * blockDim.y].x = input[preY + preX * 481].x;
			output[y + x * blockDim.y].y = input[preY + preX * 481].y;
		}
		*/
		if (preX < 0)
			preX = 1280 + preX;
		if (preX >= 1280)
			preX = preX - 1280;
		if (preY < 0)
			preY = 960 + preY;
		if (preY >= 960)
			preY = preY - 960;
		output[y + x * blockDim.y].x = input[preY + preX * blockDim.y].x;
		output[y + x * blockDim.y].y = input[preY + preX * blockDim.y].y;
	}
}

__global__ void phaseCalculate(cufftComplex* input, cufftReal* output, int numElements) {

	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		output[i] = atan2(input[i].y, input[i].x);
	}
}

__global__ void createXConfVec(float xConf, float vecStep, cufftReal* output, int numElements) {
	int i = threadIdx.x;
	if (i < numElements) {
		output[i] = xConf - i * vecStep;
	}
}

__global__ void forPhaseImage(float mean2, cufftReal* xConfVec, cufftReal* input, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		input[x + y * gridDim.x] = input[x + y * gridDim.x] + xConfVec[x] - mean2;
		if (input[x + y * gridDim.x] < 0)
			input[x + y * gridDim.x] = -input[x + y * gridDim.x];
	}
}

__global__ void calHeight(cufftReal* input, float mean2, cufftReal* output, int numElements) {
	int dn = 0.075;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		output[i] = (input[i] * lambda / 2 / pi / dn - mean2) / 4 * 3;
	}
}

__global__ void calOutputImage(cufftReal* input, float* output, int numElements) {
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		output[i] = ((input[i] + 15) / 30) * 256 - 1;
	}
}

__global__ void matrixMultiple(float* output, float* matrixL, float* matrixR , int size ,int numElements) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * (blockDim.y * gridDim.y) + (threadIdx.y + blockIdx.y * blockDim.y);
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockDim.y + threadIdx.y;
	if (i < numElements) {
		float data;
		for (int a = 0; a < size; a++) {
			data += matrixL[a * size + y] * matrixR[a + x * size];
		}
		output[i] = data;
	}
}

__global__ void DCTMatrixL(int height, float* DCTMatrixL) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * (blockDim.y * gridDim.y) + (threadIdx.y + blockIdx.y * blockDim.y);
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockDim.y + threadIdx.y;
	DCTMatrixL[i] = cospi((2 * x + 1) * y / 2 * height);
}

__global__ void DCTMatrixR(int width, float* DCTMatrixR) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * (blockDim.y * gridDim.y) + (threadIdx.y + blockIdx.y * blockDim.y);
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockDim.y + threadIdx.y;
	DCTMatrixR[i] = cospi((2 * y + 1) * x / 2 * width);
}

__global__ void matrixModify(float* input ,int width, int height) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * (blockDim.y * gridDim.y) + (threadIdx.y + blockIdx.y * blockDim.y);
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockDim.y + threadIdx.y;
	input[i] = input[i] / (2 * (cospi(x / width) + cospi(y / height) - 2));
}

__global__ void matrixModify(float* input, int width, int height) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * (blockDim.y * gridDim.y) + (threadIdx.y + blockIdx.y * blockDim.y);
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockDim.y + threadIdx.y;
	input[i] = input[i] / (2 * (cospi((float)x / width) + cospi((float)y / height) - 2));
	if (x == 0)
		if (y == 0)
			input[i] = input[i] / (height * width);
		else
			input[i] = input[i] * 2 / (height * width);
	else
		if (y == 0)
			input[i] = input[i] / height * width;
		else
			input[i] = input[i] * 2 / (height * width);
}

__global__ normalizer(float* input , int height , int width) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * (blockDim.y * gridDim.y) + (threadIdx.y + blockIdx.y * blockDim.y);
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockDim.y + threadIdx.y;
	if (x == 0)
		if (y == 0)
			input[i] = input[i] * sqrt((float)1 / (height * width));
		else
			input[i] = input[i] * sqrt((float)2 / (height * width));
	else
		if (y == 0)
			input[i] = input[i] / sqrt((float)height * width);
		else
			input[i] = input[i] * sqrt((float)2 / (height * width));
}

__global__ void gradCal(float* input, float* output, int height, int width) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * (blockDim.y * gridDim.y) + (threadIdx.y + blockIdx.y * blockDim.y);
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockDim.y + threadIdx.y;
	if (x < width - 1 && y < height - 1 && x > 0 && y > 0) {
		output[i] = input[i + height] + input[i - height] - 4 * input[i] + input[i + 1] + input[i - 1];
	}
	else if (x == 0) {
		if (y == 0)
			output[i] = input[i + height] - 4 * input[i] + input[i + 1];
		else if (y == width - 1)
			output[i] = input[i + height] - 4 * input[i] + input[i - 1];
		else
			output[i] = input[i + height] - 4 * input[i] + input[i + 1] + input[i - 1];
	}
	else if (x == width - 1) {
		if (y == 0)
			output[i] = input[i - height] - 4 * input[i] + input[i + 1];
		else if (y == width - 1)
			output[i] = input[i - height] - 4 * input[i] + input[i - 1];
		else
			output[i] = input[i - height] - 4 * input[i] + input[i + 1] + input[i - 1];
	}
}

/*
__global__ void MatrixMultiple(int m, int n, int k, float* A, float* B, float* C)
{
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;		int by = blockIdx.y;
	int tx = threadIdx.x;		int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	double Cvalue = 0;

	for (int t = 0; t < (n - 1) / TILE_WIDTH + 1; ++t)
	{
		
		if (Row < m && t * TILE_WIDTH + tx < n)		//越界处理，满足任意大小的矩阵相乘（可选）
			//ds_A[tx][ty] = A[t*TILE_WIDTH + tx][Row];
			ds_A[tx][ty] = A[Row * n + t * TILE_WIDTH + tx];//以合并的方式加载瓦片
		else
			ds_A[tx][ty] = 0.0;

		if (t * TILE_WIDTH + ty < n && Col < k)
			//ds_B[tx][ty] = B[Col][t*TILE_WIDTH + ty];
			ds_B[tx][ty] = B[(t * TILE_WIDTH + ty) * k + Col];
		else
			ds_B[tx][ty] = 0.0;

		//保证tile中所有的元素被加载
		__syncthreads();

		for (int i = 0; i < TILE_WIDTH; ++i)
			Cvalue += ds_A[i][ty] * ds_B[tx][i];//从shared memory中取值

		//确保所有线程完成计算后，进行下一个阶段的计算
		__syncthreads();

		if (Row < m && Col < k)
			C[Row * k + Col] = Cvalue;
	}
}
*/


int main() {
	image calibImage, testImage;
	do
	{
		cout << "please input the filename and the address of the calibration image: ";
		cin >> calibImage.filename;
	} while (!getImageInfo(&calibImage));
	do
	{
		cout << "please input the filename and the address of the test image: ";
		cin >> testImage.filename;
	} while (!getImageInfo(&testImage));
	fourierFilterForCalib(&calibImage);
	float* result = phaseRetrieval(&calibImage, &testImage);
}

bool getImageInfo(image* targetImage) {

	TIFF* tif = TIFFOpen(targetImage->filename, "r");
	if (tif) {
		TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &targetImage->height);
		TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &targetImage->width);
		targetImage->imagePixels = targetImage->height * targetImage->width;
		targetImage->imageData = new float2[targetImage->imagePixels];
		targetImage->rawImage = new uint8 * [targetImage->height];
		for (int i = 0; i < targetImage->height; i++) {
			targetImage->rawImage[i] = (uint8*)malloc(sizeof(uint8) * targetImage->width);
		}
		for (int i = 0; i < targetImage->height; i++) {
			TIFFReadScanline(tif, targetImage->rawImage[i], i);
		}
		float tempMax = 0;
		float tempMin = 10000;
		for (int i = 0; i < targetImage->width; i++) {
			for (int j = 0; j < targetImage->height; j++) {
				targetImage->imageData[i * targetImage->height + j].x = ((float)targetImage->rawImage[j][i] / 255);
				targetImage->imageData[i * targetImage->height + j].y = 0;
				//if (tempMax < targetImage->imageData[i * targetImage->width + j]) 
				//	tempMax = targetImage->imageData[i * targetImage->width + j];
				//if (tempMin > targetImage->imageData[i * targetImage->width + j])
				//	tempMin = targetImage->imageData[i * targetImage->width + j];
			}
		}
		targetImage->maxIntensity = tempMax;
		targetImage->minIntensity = tempMin;
		TIFFClose(tif);
		return true;
	}
	else {
		cout << "File Open Error! please input a valid filename" << endl;
		return false;
	}

	return true;
}

int2 findMaxPoint(float* input) {
	int2 tempPoint = { 0,0 };
	float tempMax = 0;
	for (int i = 0; i < 960; i++) {
		for (int j = 700; j < 1280; j++) {
			if (input[i + j * 960] > tempMax) {
				tempMax = input[i + j * 960];
				tempPoint.y = i;
				tempPoint.x = j;
			}
		}
	}
	return tempPoint;
}

void fourierFilterForCalib(image* calibImage) {
	cout << "Part: fourier filter for calib image" << endl;


	cudaEvent_t FFStart;
	cudaEventCreate(&FFStart);
	cudaEvent_t FFStop;
	cudaEventCreate(&FFStop);
	cudaEventRecord(FFStart, NULL);


	int imageSizeL = 1280 * 960;
	dim3 blockSizeL(1, 960, 1), gridSize(1280, 1, 1);
//	float2* tempComplex = new float2[imageSizeL];
	float* calibAbsImage = new float[imageSizeL];

	//complexWrite("input for fourierFiltered", calibImage->imageData, 960, "../Debug/input_FF.csv");

	cufftReal* dev_calibABSFFTShifted;
	cufftComplex* dev_calibFFT, * dev_circCalibFFT, * dev_calibFilteredBaseband, * dev_calibCircFilteredFFT, * dev_filteredCalibFFT, * dev_calibFFTShifted, * dev_calibImage;
	int n = cudaMalloc((void**)& dev_calibImage, sizeof(float2) * calibImage->imagePixels);
//	if (cudaSuccess != n)
//		cout << "cuda malloc error1!" << endl;
//	cout << n << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_calibFilteredBaseband, sizeof(float2) * imageSizeL))
		cout << "cuda malloc error2!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_calibCircFilteredFFT, sizeof(float2) * imageSizeL))
		cout << "cuda malloc error3!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_calibABSFFTShifted, sizeof(float) * imageSizeL))
		cout << "cuda malloc error4!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_calibFFT, sizeof(float2) * imageSizeL))
		cout << "cuda malloc error5!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_circCalibFFT, sizeof(float2) * imageSizeL))
		cout << "cuda malloc error8!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_filteredCalibFFT, sizeof(float2) * imageSizeL))
		cout << "cuda malloc error8!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_calibFFTShifted, sizeof(float2) * imageSizeL))
		cout << "cuda malloc error8!" << endl;

	cufftHandle FFT;
	errorHandle(cufftPlan2d(&FFT, 1280, 960, CUFFT_C2C));

	if (cudaSuccess != cudaMemcpy(dev_calibImage, calibImage->imageData, calibImage->imagePixels * sizeof(cufftComplex), cudaMemcpyHostToDevice))
		cout << "cuda memory cpy error!" << endl;

	cudaEvent_t fftStart;
	cudaEventCreate(&fftStart);
	cudaEvent_t fftStop;
	cudaEventCreate(&fftStop);
	cudaEventRecord(fftStart, NULL);

	errorHandle(cufftExecC2C(FFT, dev_calibImage, dev_calibFFT, CUFFT_FORWARD));

	cudaEventRecord(fftStop, NULL);
	cudaEventSynchronize(fftStop);
	float msecFFT = 0.0f;
	cudaEventElapsedTime(&msecFFT, fftStart, fftStop);
	cout << "total runtime of FFT: " << msecFFT << " ms" << endl;

//	float2* tempOut = (float2*)malloc(sizeof(float2) * 1280 * 960);
//	int a = cudaMemcpy((void*)tempOut, (void*)dev_calibFFT, imageSizeL * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
//	if (cudaSuccess != a)
//		cout << "cuda memory cpy error!" << endl;
//	cout << a << endl;
//	complexWrite("temp dubug info", tempOut, 960, "../Debug/calib_FFT.csv");

	cudaEvent_t fftShiftStart;
	cudaEventCreate(&fftShiftStart);
	cudaEvent_t fftShiftStop;
	cudaEventCreate(&fftShiftStop);
	cudaEventRecord(fftShiftStart, NULL);

	FFTShift2D << < gridSize, blockSizeL >> > (dev_calibFFT, dev_calibFFTShifted, imageSizeL);

	cudaEventRecord(fftShiftStop, NULL);
	cudaEventSynchronize(fftShiftStop);
	float msecFFTShift = 0.0f;
	cudaEventElapsedTime(&msecFFTShift, fftShiftStart, fftShiftStop);
	cout << "total runtime of FFT: " << msecFFTShift << " ms" << endl;

//	if (cudaSuccess != cudaGetLastError())
//		printf("FFTShift error!\n");
//	if (cudaSuccess != cudaMemcpy(tempComplex, dev_calibFFTShifted, imageSizeL * sizeof(cufftComplex), cudaMemcpyDeviceToHost))
//		cout << "cuda memory cpy error!" << endl;
//	complexWrite("fft after shift", tempComplex, 960, "../Debug/calib_FFT_Shifted.csv");


	cudaEvent_t absStart;
	cudaEventCreate(&absStart);
	cudaEvent_t absStop;
	cudaEventCreate(&absStop);
	cudaEventRecord(absStart, NULL);

	cudaEvent_t absStart;
	cudaEventCreate(&absStart);
	cudaEvent_t absStop;
	cudaEventCreate(&absStop);
	cudaEventRecord(absStart, NULL);

	getAbsOfComplexMatric << < gridSize, blockSizeL >> > (dev_calibFFTShifted, dev_calibABSFFTShifted, imageSizeL);


	cudaEventRecord(absStop, NULL);
	cudaEventSynchronize(absStop);
	float msecABS = 0.0f;
	cudaEventElapsedTime(&msecABS	, absStart, absStop);
	cout << "total runtime of ABS: " << msecABS << " ms" << endl;

//	if (cudaSuccess != cudaGetLastError())
//		printf("get abs error!\n");

	int b = cudaMemcpy(calibAbsImage, dev_calibABSFFTShifted, imageSizeL * sizeof(cufftReal), cudaMemcpyDeviceToHost);
//	if (cudaSuccess != b)
//		cout << "cuda memory cpy error!" << endl;
//	cout << b << endl;
//	realWrite("calib abs image", calibAbsImage, 960, "../Debug/calib_abs_image.csv");

	calibImage->fftMaxPosition = findMaxPoint(calibAbsImage);
//	cout << "Xmax= " << calibImage->fftMaxPosition.x << ", Ymax= " << calibImage->fftMaxPosition.y << endl;

	cudaEvent_t filterStart;
	cudaEventCreate(&filterStart);
	cudaEvent_t filterStop;
	cudaEventCreate(&filterStop);
	cudaEventRecord(filterStart, NULL);

	circShift2D << <gridSize, blockSizeL >> > (dev_calibFFTShifted, calibImage->fftMaxPosition, dev_circCalibFFT, imageSizeL);

//	if (cudaSuccess != cudaGetLastError())
//		printf("circle shift error!\n");
//	if (cudaSuccess != cudaMemcpy(tempComplex, dev_circCalibFFT, imageSizeL * sizeof(cufftComplex), cudaMemcpyDeviceToHost))
//		cout << "cuda memory cpy error!" << endl;
//	complexWrite("fft after circshift", tempComplex, 960, "../Debug/circFFT.csv");

	createFilter << <gridSize, blockSizeL >> > (80, calibImage->fftMaxPosition, dev_circCalibFFT, dev_calibCircFilteredFFT, imageSizeL);



//	if (cudaSuccess != cudaGetLastError())
//		printf("filter create error!\n");
//	if (cudaSuccess != cudaMemcpy(tempComplex, dev_calibCircFilteredFFT, imageSizeL * sizeof(cufftComplex), cudaMemcpyDeviceToHost))
//		cout << "cuda memory cpy error!" << endl;
//	complexWrite("fft after circshift", tempComplex, 960, "../Debug/calib_filtered.csv");

	IFFTShift2D << <gridSize, blockSizeL >> > (dev_calibCircFilteredFFT, dev_filteredCalibFFT, imageSizeL);
//	if (cudaSuccess != cudaGetLastError())
//		printf("IFFT shift error!\n");

	cudaEventRecord(filterStop, NULL);
	cudaEventSynchronize(filterStop);
	float msecFilter = 0.0f;
	cudaEventElapsedTime(&msecFilter, filterStart, filterStop);
	cout << "total runtime of creating filter: " << msecFilter << " ms" << endl;

//	if (cudaSuccess != cudaMemcpy(tempComplex, dev_filteredCalibFFT, imageSizeL * sizeof(cufftComplex), cudaMemcpyDeviceToHost))
//		cout << "cuda memory cpy error!" << endl;
//	complexWrite("fft after circshift", tempComplex, 960, "../Debug/ifft_shifted.csv");


	cudaEvent_t ifftStart;
	cudaEventCreate(&ifftStart);
	cudaEvent_t ifftStop;
	cudaEventCreate(&ifftStop);
	cudaEventRecord(ifftStart, NULL);
	
	cufftExecC2C(FFT, dev_filteredCalibFFT, dev_calibFilteredBaseband, CUFFT_INVERSE);

	cudaEventRecord(ifftStop, NULL);
	cudaEventSynchronize(ifftStop);
	float msecIFFT = 0.0f;
	cudaEventElapsedTime(&msecIFFT, ifftStart, ifftStop);
	cout << "total runtime of FFT: " << msecIFFT << " ms" << endl;

	vectorNumdivide << <gridSize, blockSizeL >> > (dev_calibFilteredBaseband, imageSizeL, imageSizeL);
	if (cudaSuccess != cudaMemcpy(calibImage->filteredBaseband, dev_calibFilteredBaseband, (calibImage->imagePixels) * sizeof(float2), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error!" << endl;

//	complexWrite("calib filtered baseband", calibImage->filteredBaseband, 960, "../Debug/calib_filtered_baseband.csv");

	if (cudaSuccess != cudaFree(dev_calibImage))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_calibABSFFTShifted))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_circCalibFFT))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_calibFilteredBaseband))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_calibFFT))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_filteredCalibFFT))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_calibCircFilteredFFT))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_calibFFTShifted))
		cout << "cude memory free error!" << endl;


	cudaEventRecord(FFStop, NULL);
	cudaEventSynchronize(FFStop);
	float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, FFStart, FFStop);
	cout << "total runtime of part fourier filter: " << msecTotal << " ms" << endl;
}

float* phaseRetrieval(image* calibImage, image* testImage) {
	cout << "Part: phase retrieval" << endl;

	int imageSizeL = 1280 * 960;
	dim3 blockSizeL(1, 960, 1), gridSize(1280, 1, 1);
	float2* tempComplex = new float2[imageSizeL];
	float* testAbsImage = (float*)malloc(sizeof(float) * imageSizeL);

	cufftReal* dev_testABSFFTShifted;
	cufftComplex* dev_testFFT, * dev_circTestFFT, * dev_testFilteredBaseband, * dev_testCircFilteredFFT, * dev_filteredTestFFT, * dev_testFFTShifted, * dev_testImage;
	int n = cudaMalloc((void**)& dev_testImage, sizeof(float2) * testImage->imagePixels);
//	if (cudaSuccess != n)
//		cout << "cuda malloc error1!" << endl;
//	cout << n << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_testFilteredBaseband, sizeof(float2) * testImage->imagePixels))
		cout << "cuda malloc error2!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_testCircFilteredFFT, sizeof(float2) * imageSizeL))
		cout << "cuda malloc error3!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_testABSFFTShifted, sizeof(float) * imageSizeL))
		cout << "cuda malloc error4!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_testFFT, sizeof(float2) * imageSizeL))
		cout << "cuda malloc error5!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_circTestFFT, sizeof(float2) * testImage->imagePixels))
		cout << "cuda malloc error8!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_filteredTestFFT, sizeof(float2) * testImage->imagePixels))
		cout << "cuda malloc error8!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_testFFTShifted, sizeof(float2) * imageSizeL))
		cout << "cuda malloc error8!" << endl;

	cufftHandle FFT;
	errorHandle(cufftPlan2d(&FFT, 1280, 960, CUFFT_C2C));

	if (cudaSuccess != cudaMemcpy(dev_testImage, testImage->imageData, testImage->imagePixels * sizeof(float2), cudaMemcpyHostToDevice))
		cout << "cuda memory cpy error!" << endl;

	errorHandle(cufftExecC2C(FFT, dev_testImage, dev_testFFT, CUFFT_FORWARD));
	FFTShift2D << < gridSize, blockSizeL >> > (dev_testFFT, dev_testFFTShifted, imageSizeL);
//	if (cudaSuccess != cudaGetLastError())
//		printf("FFT shift Error!\n");

	circShift2D << <gridSize, blockSizeL >> > (dev_testFFTShifted, calibImage->fftMaxPosition, dev_circTestFFT, imageSizeL);
//	if (cudaSuccess != cudaGetLastError())
//		printf("circ shift Error!\n");
	createFilter << <gridSize, blockSizeL >> > (80, calibImage->fftMaxPosition, dev_circTestFFT, dev_testCircFilteredFFT, imageSizeL);
//	if (cudaSuccess != cudaGetLastError())
//		printf("filter create Error!\n");
	IFFTShift2D << <gridSize, blockSizeL >> > (dev_testCircFilteredFFT, dev_filteredTestFFT, imageSizeL);
	if (cudaSuccess != cudaGetLastError())
		printf("IFFT shift Error!\n");
	errorHandle(cufftExecC2C(FFT, dev_filteredTestFFT, dev_testFilteredBaseband, CUFFT_INVERSE));
	vectorNumdivide << <gridSize, blockSizeL >> > (dev_testFilteredBaseband, imageSizeL, imageSizeL);
	if (cudaSuccess != cudaMemcpy(testImage->filteredBaseband, dev_testFilteredBaseband, (testImage->imagePixels) * sizeof(float2), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error" << endl;
//	complexWrite("test filtered baseband", testImage->filteredBaseband, 960, "../Debug/test_filtered_baseband.csv");

	if (cudaSuccess != cudaFree(dev_testImage))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_testABSFFTShifted))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_circTestFFT))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_testFFT))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_filteredTestFFT))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_testCircFilteredFFT))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_testFFTShifted))
		cout << "cude memory free error!" << endl;

	float* phaseImage = (float*)malloc(sizeof(float) * imageSizeL);
	
	cufftReal* dev_phaseImage;
	cufftComplex* dev_calibFilteredBaseband, * dev_finalImage;
	if (cudaSuccess != cudaMalloc((void**)& dev_phaseImage, sizeof(float) * imageSizeL))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_calibFilteredBaseband, sizeof(float2) * imageSizeL))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_finalImage, sizeof(float2) * imageSizeL))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMemcpy(dev_calibFilteredBaseband, calibImage->filteredBaseband, imageSizeL * sizeof(float2), cudaMemcpyHostToDevice))
		cout << "cuda memory cpy error!" << endl;

	vectorMatDivide << <gridSize, blockSizeL >> > (dev_testFilteredBaseband, dev_calibFilteredBaseband, dev_finalImage, calibImage->imagePixels);
	if (cudaSuccess != cudaGetLastError())
		printf("divide Error!\n");
	phaseCalculate << <gridSize, blockSizeL >> > (dev_finalImage, dev_phaseImage, imageSizeL);
	if (cudaSuccess != cudaGetLastError())
		printf("phase calculate Error!\n");
	if (cudaSuccess != cudaMemcpy(phaseImage, dev_phaseImage, imageSizeL * sizeof(float), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error!" << endl;
	realWrite("phase image", phaseImage, 960, "../Debug/phase_image1.csv");
	
	float* UnwrappedImage = new float[imageSizeL];

	phaseUnwrapping(phaseImage, UnwrappedImage);

	realWrite("phase image after unwrapping", UnwrappedImage, 1280, "..\ouput_text\phase_image2.csv");
	
	if (cudaSuccess != cudaMemcpy(dev_phaseImage, phaseImage, testImage->imagePixels * sizeof(float), cudaMemcpyHostToDevice))
		cout << "cuda memory cpy error!" << endl;

	float xConf, yConf, xSum = 0, ySum = 0, vecStep, sum2 = 0, mean2;
	for (int i = 0; i < 960; i++) {
		xSum += phaseImage[i * testImage->width + 29] - phaseImage[i * testImage->width + 1279 - 30];
	}
	//for (int i = 0; i < 1280; i++) {
	//	ySum += phaseImage[29 * testImage.width + i] - phaseImage[929 * testImage.width + i];
	//}
	xConf = -1 * (xSum / 960);
	vecStep = xConf / 1279;
	for (int x = 0; x < 100; x++) {
		for (int y = 0; y < 100; y++) {
			sum2 += phaseImage[x + y * testImage->width];
		}
	}
	mean2 = sum2 / 10000;

	cufftReal* dev_xConfVec;
	if (cudaSuccess != cudaMalloc((void**)& dev_xConfVec, sizeof(float) * 1280))
		cout << "cuda malloc error!" << endl;
	createXConfVec << <(1, 1, 1), (1280, 1, 1) >> > (xConf, vecStep, dev_xConfVec, 1280);
	if (cudaSuccess != cudaGetLastError())
		printf("xConf vec create Error!\n");

	forPhaseImage << <gridSize, blockSizeL >> > (mean2, dev_xConfVec, dev_phaseImage, testImage->imagePixels);
	cudaFree(dev_xConfVec);
	cudaThreadSynchronize();
	if (cudaSuccess != cudaMemcpy(phaseImage, dev_phaseImage, testImage->imagePixels * sizeof(float), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error!" << endl;
	
	float* height1 = (float*)malloc(sizeof(float) * testImage->imagePixels);
	cufftReal* dev_height;
	cudaMalloc((void**)& dev_height, sizeof(float) * testImage->imagePixels);
	calHeight << <gridSize, blockSizeL >> > (dev_phaseImage, mean2, dev_height, testImage->imagePixels);
	if (cudaSuccess != cudaGetLastError())
		printf("cal height Error!\n");
	cudaThreadSynchronize();
	if (cudaSuccess != cudaMemcpy(height1, dev_height, testImage->imagePixels * sizeof(float), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error!" << endl;

	float* outputImage = (float*)malloc(sizeof(float) * testImage->imagePixels);
	float* dev_output;
	if (cudaSuccess != cudaMalloc((void**)& dev_output, sizeof(float) * testImage->imagePixels))
		cout << "cuda malloc error!" << endl;
	calOutputImage << <gridSize, blockSizeL >> > (dev_height, dev_output, testImage->imagePixels);
	if (cudaSuccess != cudaGetLastError())
		printf("output image create Error!\n");
	cudaThreadSynchronize();
	if (cudaSuccess != cudaMemcpy(outputImage, dev_output, testImage->imagePixels * sizeof(float), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error!" << endl;

	cudaFree(dev_output);
	cudaFree(dev_height);
	
	return height1;

}

float* phaseUnwrapping(float* wMatrix, float* result) {
	dim3 blockSize(32, 32, 1), gridSize(40, 30, 1);
	int imageSize = 1280 * 960;
	int width = 960;
	int height = 1280;

	float* dev_wMatrix;
	float* dev_GradMatrix, * dev_matrixS, * dev_matrixL, * dev_temp, * dev_unwrapC, * dev_result;
	if (cudaSuccess != cudaMalloc((void**)& dev_GradMatrix, sizeof(float) * imageSize))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_matrixS, sizeof(double) * imageSize))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_matrixL, sizeof(double) * imageSize))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_temp, sizeof(double) * imageSize))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_unwrapC, sizeof(double) * imageSize))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_unwrapC, sizeof(double) * imageSize))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_wMatrix, sizeof(float) * imageSize))
		cout << "cuda malloc error!" << endl;

	if (cudaSuccess != cudaMemcpy(dev_wMatrix, wMatrix, 1280 * 960 * sizeof(float), cudaMemcpyHostToDevice))
		cout << "cuda memory cpy error!" << endl;
	gradCal << <gridSize, blockSize >> > (dev_wMatrix, dev_GradMatrix, 960, 1280);

	DCTMatrixL<<<gridSize , blockSize >>>(height, dev_matrixS);
	DCTMatrixR<<<gridSize , blockSize >>>(width, dev_matrixL);

	matrixMultiple << <gridSize, blockSize >> > (dev_temp, dev_matrixS, dev_GradMatrix, 960, imageSize);
	matrixMultiple << <gridSize, blockSize >> > (dev_unwrapC, dev_temp, dev_matrixL, 1280, imageSize);
	matrixModify << <gridSize, blockSize >> > (dev_unwrapC, 1280, 960);

	matrixMultiple << <gridSize, blockSize >> > (dev_temp, dev_matrixS, dev_unwrapC, 960, imageSize);
	matrixMultiple << <gridSize, blockSize >> > (dev_result, dev_temp, dev_matrixL, 1280, imageSize);

	if (cudaSuccess != cudaMemcpy(result, dev_result, imageSize * sizeof(float), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error!" << endl;
}

void imageFileWrite(float* input, char* filename) {

	TIFF* tif = TIFFOpen(filename, "w");
	if (tif) {
		TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, 960);
		TIFFSetField(tif, TIFFTAG_IMAGELENGTH, 1280);
		uint8** tempData = new uint8 * [960];
		for (int i = 0; i < 960; i++) {
			tempData[i] = new uint8[1280];
			for (int j = 0; j < 1280; j++) {
				tempData[i][j] = (uint8)input[j + i * 1280];
			}
			TIFFWriteScanline(tif, tempData[i], i);
		}
	}
	else
		cout << filename << " can not be opened!" << endl;

}

void realWrite(const char* title, float* input, int height, const char* filename) {

	ofstream outFile;
	outFile.open(filename);
	outFile.setf(ios::fixed, ios::floatfield);
	outFile.precision(7);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < 1280; x++) {
			outFile << input[x * height + y] << ",";
		}
		outFile << " " << endl;
	}

}

void complexWrite(const char* title, float2* input, int height, const char* filename) {
	ofstream outFile;
	outFile.open(filename);
	outFile.setf(ios::fixed, ios::floatfield);
	outFile.precision(7);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < 1280; x++) {
			outFile << input[x * height + y].x << "+" << input[x * height + y].y << "i" << ",";
		}
		outFile << " " << endl;
	}
}

void errorHandle(int input) {
	switch (input)
	{
	case CUFFT_ALLOC_FAILED:
		cout << "The allocation of GPU resources for the plan failed." << endl;
	case CUFFT_INVALID_VALUE:
		cout << "One or more invalid parameters were passed to the API." << endl;
	case CUFFT_INTERNAL_ERROR:
		cout << "An internal driver error was detected." << endl;
	case CUFFT_SETUP_FAILED:
		cout << "cuFFT library initialize fail." << endl;
	case CUFFT_INVALID_SIZE:
		cout << "One or more of the nx, ny, or nz parameters is not a supported size." << endl;
	case CUFFT_INVALID_PLAN:
		cout << "The plan parameter is not a valid handle." << endl;
	case CUFFT_EXEC_FAILED:
		cout << "cuFFT failed to execute the transform on the GPU." << endl;
	case CUFFT_SUCCESS:
		break;
	}
}

/*
cudaEvent_t start1;
cudaEventCreate(&start1);
cudaEvent_t stop1;
cudaEventCreate(&stop1);
cudaEventRecord(start1, NULL);
// 需要测时间的内核函数kernel;
cudaEventRecord(stop1, NULL);
cudaEventSynchronize(stop1);
float msecTotal1 = 0.0f;
cudaEventElapsedTime(&msecTotal1, start1, stop1);
*/

