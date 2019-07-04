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

typedef unsigned char BYTE;

using namespace std;

class image {
public:
	char filename[50];
	float maxIntensity, minIntensity;
	int height, width;
	int imagePixels;
	uint8** rawImage;
	float* imageData;
	int2 fftMaxPosition;
	float2* filteredBaseband;
	float* imageFilter;

	image() {
		rawImage = (uint8 * *)malloc(sizeof(uint8*) * 960);
		for (int i = 0; i < 960; i++) {
			rawImage[i] = (uint8*)malloc(sizeof(uint8) * 1280);
		}
		filteredBaseband =(float2*)malloc(sizeof(float)*2*1280*960);
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


__global__ void vectorAdd( float* a,  float* b, float* c, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements)
		c[i] = a[i] + b[i];
}

__global__ void vectorNumMultiple( float* a,  float* b, float* c, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements)
		c[i] = a[i] * b[i];
}

__global__ void numMultipleForComplex( float2* a,  float* b, float2* c, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x + threadIdx.y * gridDim.x;
	if (i < numElements) {
		c[i].x = a[i].x * b[i];
		c[i].y = a[i].y * b[i];
	}
}

__global__ void vectorNumdivide( float2* dividend,  float2* divisor, float2* output, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		output[i].x = dividend[i].x / divisor[i].x;
		output[i].y = dividend[i].y / divisor[i].y;
	}

}

__global__ void getAbsOfComplexMatric( cufftComplex* input, cufftReal* output, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements)
		output[i] = sqrt(input[i].x * input[i].x + input[i].y * input[i].y);
}

__global__ void createFilter( int padding,  int2 maxPoint, float2* input, float2* output, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		if ((x * x + y * y) <= padding * padding) {
			output[i].x = input[i].x;
			output[i].y = input[i].y;
		}
		else {
			output[i].x = 0;
			output[i].y = 0;
		}
	}
}

__global__ void FFTShift2D( cufftComplex* input, cufftComplex* output, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	int half = gridDim.x / 2;
	if (i < numElements ) {
		if (y >= half / 2) {
			output[(y - half) * gridDim.x + x].x = input[y * gridDim.x + x].x;
			output[(y - half) * gridDim.x + x].y = input[y * gridDim.x + x].y;
		}
		else {
			output[(y + half) * gridDim.x + x].x = input[y * gridDim.x + x].x;
			output[(y + half) * gridDim.x + x].y = input[y * gridDim.x + x].y;
		}
	}
}

__global__ void IFFTShift2D( cufftComplex* input, cufftComplex* output, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	int halfY = blockDim.y / 2;
	int halfX = gridDim.x / 2;
	int preX, preY;
	if (i < numElements) {
		if (y >= halfY)
			preY = y - halfY;
		else
			preY = y + halfY;
		if (x >= halfX)
			preX = x - halfX;
		else
			preX = x + halfX;
		output[x + gridDim.x * y].x = input[preX + gridDim.x * preY].x;
		output[x + gridDim.x * y].y = input[preX + gridDim.x * preY].y;
	}
}

__global__ void circShift2D( cufftComplex* input,  int2 maxPoint, cufftComplex* output, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	int preX = x - 640 + maxPoint.x;
	int preY = x - 480 + maxPoint.y;
	if (i < numElements) {
		if (preX < 0)
			preX = 1280 + preX;
		if (preY < 0 || preY > 480) {
			output[x + gridDim.x + y].x = 0;
			output[x + gridDim.x + y].y = 0;
		}
		else {
			output[x + gridDim.x * y].x = input[preX + gridDim.x * preY].x;
			output[x + gridDim.x * y].y = input[preX + gridDim.x * preY].y;
		}
	}
}

__global__ void phaseCalculate( cufftComplex* input, cufftReal* output, int numElements) {

	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		output[i] = atan2(input[i].y, input[i].x);
	}
}

__global__ void createXConfVec( float xConf,  float vecStep, cufftReal* output, int numElements) {
	int i = threadIdx.x;
	if (i < numElements) {
		output[i] = xConf - i * vecStep;
	}
}

__global__ void forPhaseImage( float mean2,  cufftReal* xConfVec, cufftReal* input, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		input[x + y * gridDim.x] = input[x + y * gridDim.x] + xConfVec[x] - mean2;
		if (input[x + y * gridDim.x] < 0)
			input[x + y * gridDim.x] = -input[x + y * gridDim.x];
	}
}

__global__ void calHeight( cufftReal* input, float mean2, cufftReal* output, int numElements) {
	int dn = 0.075;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		output[i] = (input[i] * lambda / 2 / pi / dn - mean2) / 4 * 3;
	}
}

__global__ void calOutputImage( cufftReal* input, float* output, int numElements) {
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		output[i] = ((input[i] + 15) / 30) * 256 - 1;
	}
}


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
	float* result = phaseRetrieval(&calibImage,&testImage);
}

bool getImageInfo( image* targetImage ) {
	
	TIFF* tif = TIFFOpen(targetImage->filename, "r");
	if (tif) {
		TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &targetImage->height);
		TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &targetImage->width);
		targetImage->imagePixels = targetImage->height * targetImage->width;
		targetImage->imageData = new float[targetImage->imagePixels];
		targetImage->rawImage = new uint8 * [targetImage->height];
		for (int i = 0; i < targetImage->height; i++) {
			targetImage->rawImage[i] =(uint8*)malloc(sizeof(uint8)*targetImage->width);
		}
		for (int i = 0; i < targetImage->height; i++) {
			TIFFReadScanline(tif, targetImage->rawImage[i], i);
		}
		float tempMax = 0;
		float tempMin = 10000;
		for (int i = 0; i < targetImage->width; i++) {
			for (int j = 0; j < targetImage->height; j++) {
				targetImage->imageData[i*targetImage->height + j] = ((float)targetImage->rawImage[j][i] / 256);
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
	int2 tempPoint = {0,0};
	float tempMax = 0;
	for (int i = 0; i < 481; i++) {
		for (int j = 0; j < 1280; j++) {
			if (j > 600 && j < 680)
				continue;
			if (input[i + j * 481] > tempMax) {
				tempMax = input[i + j * 481];
				tempPoint.y = i;
				tempPoint.x = j;
			}
		}
	}
	return tempPoint;
}

void fourierFilterForCalib(image* calibImage) {
	cout << "Part: fourier filter for calib image" << endl;
	int imageSizeS = 1280 * 481;
	int imageSizeL = 1280 * 960;
	dim3 blockSizeL(1, 960, 1), gridSize(1280, 1, 1), blockSizeS(1, 481, 1);

	float* calibAbsImage = new float[imageSizeS];

	realWrite("input for fourierFiltered", calibImage->imageData, 1280, "../Debug/input_FF.txt");

	cufftReal* dev_calibImage, * dev_calibABSFFTShifted;
	cufftComplex* dev_calibFFT, * dev_circCalibFFT, * dev_calibFilteredBaseband, * dev_calibCircFilteredFFT, * dev_filteredCalibFFT;
	int n = cudaMalloc((void**)& dev_calibImage, sizeof(float) * calibImage->imagePixels);
	if (cudaSuccess != n)
		cout << "cuda malloc error1!" << endl;
	cout << n << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_calibFilteredBaseband, sizeof(float2) * imageSizeL))
		cout << "cuda malloc error2!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_calibCircFilteredFFT, sizeof(float2) * imageSizeL))
		cout << "cuda malloc error3!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_calibABSFFTShifted, sizeof(float2) * imageSizeS))
		cout << "cuda malloc error4!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_calibFFT, sizeof(float2) * imageSizeS))
		cout << "cuda malloc error5!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_circCalibFFT, sizeof(float2) * imageSizeL))
		cout << "cuda malloc error8!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_filteredCalibFFT, sizeof(float2) * imageSizeL))
		cout << "cuda malloc error8!" << endl;

	cufftHandle FFT;
	cufftHandle IFFT;
	errorHandle(cufftPlan2d(&FFT, 1280, 960, CUFFT_R2C));
	errorHandle(cufftPlan2d(&IFFT, 1280, 960, CUFFT_C2C));

	if (cudaSuccess != cudaMemcpy(dev_calibImage, calibImage->imageData, calibImage->imagePixels * sizeof(cufftReal), cudaMemcpyHostToDevice))
		cout << "cuda memory cpy error!" << endl;

	errorHandle(cufftExecR2C(FFT, dev_calibImage, dev_calibFFT));
	float2* tempOut = (float2*)malloc(sizeof(float2) * 1280 * 481);
	int a = cudaMemcpy((void*)tempOut, (void*)dev_calibFFT, imageSizeS * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	if (cudaSuccess != a)
		cout << "cuda memory cpy error!" << endl;
	cout << a << endl;
	complexWrite("temp dubug info", tempOut, 1280, "../Debug/tempout.txt");

	FFTShift2D <<< gridSize, blockSizeS >>> (dev_calibFFT, dev_calibFFTShifted, imageSizeS);
	if (cudaSuccess != cudaGetLastError())
		printf("FFTShift error!\n");
	
	getAbsOfComplexMatric <<< gridSize, blockSizeS >>> (dev_calibFFTShifted, dev_calibABSFFTShifted, imageSizeS);
	if (cudaSuccess != cudaGetLastError())
		printf("get abs error!\n");


	int b = cudaMemcpy((void*)calibAbsImage, (void*)dev_calibABSFFTShifted, imageSizeS * sizeof(cufftReal), cudaMemcpyDeviceToHost);
	if (cudaSuccess != b)
		cout << "cuda memory cpy error!" << endl;
	cout << b << endl;

	realWrite("calib abs image", calibAbsImage, 640, "..\ouput_text\calib_abs_image.txt");

	calibImage->fftMaxPosition = findMaxPoint(calibAbsImage);

	circShift2D <<<gridSize, blockSizeL >>> (dev_calibFFT, calibImage->fftMaxPosition, dev_circCalibFFT, imageSizeL);
	if (cudaSuccess != cudaGetLastError())
		printf("circle shift error!\n");
	createFilter << <gridSize, blockSizeL >> > (80, calibImage->fftMaxPosition, dev_circCalibFFT, dev_calibCircFilteredFFT, imageSizeL);//???
	if (cudaSuccess != cudaGetLastError())
		printf("filter create error!\n");
	IFFTShift2D <<<gridSize, blockSizeL >>> (dev_calibCircFilteredFFT, dev_filteredCalibFFT, imageSizeL);
	if (cudaSuccess != cudaGetLastError())
		printf("IFFT shift error!\n");
	cufftExecC2C(IFFT, dev_filteredCalibFFT, dev_calibFilteredBaseband,CUFFT_INVERSE);
	cudaThreadSynchronize();
	
	if (cudaSuccess != cudaMemcpy(calibImage->filteredBaseband, dev_calibFilteredBaseband, (calibImage->imagePixels ) * sizeof(float), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error!" << endl;

	complexWrite("calib filtered baseband", calibImage->filteredBaseband, 1280, "..\ouput_text\calib_filtered_baseband.txt");

	if (cudaSuccess != cudaFree(dev_calibImage))
		cout<<"cude memory free error!"<<endl;
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

}

float* phaseRetrieval(image* calibImage, image* testImage) {
	cout << "Part: phase retrieval" << endl;
	int imageSizeS = 1280 * 481;
	int imageSizeL = 1280 * 960;
	dim3 blockSizeL(1, 960, 1), gridSize(1280, 1, 1), blockSizeS(1, 641, 1);

	float* testAbsImage = (float*)malloc(sizeof(float) * imageSizeS);

	cufftReal* dev_testImage, * dev_testABSFFTShifted;
	cufftComplex* dev_testFFT, * dev_circTestFFT, * dev_testFilteredBaseband, * dev_testCircFilteredFFT, * dev_filteredTestFFT;
	int n = cudaMalloc((void**)& dev_testImage, sizeof(float) * testImage->imagePixels);
	if (cudaSuccess != n)
		cout << "cuda malloc error1!" << endl;
	cout << n << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_testFilteredBaseband, sizeof(float2) * testImage->imagePixels))
		cout << "cuda malloc error2!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_testCircFilteredFFT, sizeof(float2) * imageSizeL))
		cout << "cuda malloc error3!" << endl; 
	if (cudaSuccess != cudaMalloc((void**)& dev_testABSFFTShifted, sizeof(float2) * imageSizeS))
		cout << "cuda malloc error4!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_testFFT, sizeof(float2) * imageSizeS))
		cout << "cuda malloc error5!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_circTestFFT, sizeof(float2) * testImage->imagePixels))
		cout << "cuda malloc error8!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_filteredTestFFT, sizeof(float2) * testImage->imagePixels))
		cout << "cuda malloc error8!" << endl;

	cufftHandle FFT;
	errorHandle(cufftPlan2d(&FFT, 1280, 960, CUFFT_R2C));
	cufftHandle IFFT;
	errorHandle(cufftPlan2d(&IFFT, 1280, 960, CUFFT_C2C));

	if (cudaSuccess != cudaMemcpy(dev_testImage, testImage->imageData, testImage->imagePixels * sizeof(float), cudaMemcpyHostToDevice))
		cout << "cuda memory cpy error!" << endl;
	//
	errorHandle(cufftExecR2C(FFT, dev_testImage, dev_testFFT));
	FFTShift2D <<< gridSize, blockSizeS >>> (dev_testFFT, dev_testFFTShifted, imageSizeS);
	if (cudaSuccess != cudaGetLastError())
		printf("FFT shift Error!\n");
	cudaThreadSynchronize();
	
	circShift2D <<<gridSize, blockSizeL >>> (dev_filteredTestFFT, testImage->fftMaxPosition, dev_circTestFFT, imageSizeL);
	if (cudaSuccess != cudaGetLastError())
		printf("circ shift Error!\n");
	createFilter << <gridSize, blockSizeL >> > (80, calibImage->fftMaxPosition, dev_circTestFFT, dev_testCircFilteredFFT, imageSizeL);
	if (cudaSuccess != cudaGetLastError())
		printf("filter create Error!\n");
	IFFTShift2D <<<gridSize, blockSizeL >>> (dev_testCircFilteredFFT, dev_filteredTestFFT, imageSizeL);
	if (cudaSuccess != cudaGetLastError())
		printf("IFFT shift Error!\n");
	errorHandle(cufftExecC2C(IFFT, dev_filteredTestFFT, dev_testFilteredBaseband,CUFFT_INVERSE));
	cudaThreadSynchronize();
	if (cudaSuccess != cudaMemcpy(testImage->filteredBaseband, dev_testFilteredBaseband, (testImage->imagePixels) * sizeof(float), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error" << endl;

	complexWrite("test filtered baseband", testImage->filteredBaseband, 1280, "..\ouput_text\test_filtered_baseband.txt");

	if (cudaSuccess != cudaFree(dev_testImage))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_testABSFFTShifted))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_circTestFFT))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_testFilteredBaseband))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_testFFT))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_filteredTestFFT))
		cout << "cude memory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_testCircFilteredFFT))
		cout << "cude memory free error!" << endl;

	float* phaseImage = (float*)malloc(sizeof(float) * testImage->imagePixels);
	cufftReal* dev_phaseImage;
	cufftComplex* dev_calibFilteredBaseband,* dev_finalImage;
	if (cudaSuccess != cudaMalloc((void**)& dev_phaseImage, sizeof(float) * calibImage->imagePixels))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_calibFilteredBaseband, sizeof(float2) * calibImage->imagePixels))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_finalImage, sizeof(float2) * calibImage->imagePixels))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMemcpy(dev_calibFilteredBaseband, calibImage->filteredBaseband, testImage->imagePixels * sizeof(float2), cudaMemcpyHostToDevice))
		cout << "cuda memory cpy error!" << endl;

	vectorNumdivide <<<gridSize,blockSizeL>>> (dev_testFilteredBaseband, dev_calibFilteredBaseband, dev_finalImage, calibImage->imagePixels) ;
	if (cudaSuccess != cudaGetLastError())
		printf("divide Error!\n");
	phaseCalculate <<<gridSize, blockSizeL >>> (dev_finalImage, dev_phaseImage, calibImage->imagePixels);
	if (cudaSuccess != cudaGetLastError())
		printf("phase calculate Error!\n");
	if (cudaSuccess != cudaMemcpy(phaseImage, dev_phaseImage, testImage->imagePixels * sizeof(float), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error!" << endl;
	realWrite("phase image", phaseImage, 1280, "..\ouput_text\phase_image1.txt");
	/*
	if (!myUnwrapInitialize()) {
		cout << "matlab unwrap function initialize error" << endl;
	}
	mwArray matlabInput(960, 1280, mxSINGLE_CLASS);
	mwArray matlabOutput(960, 1280, mxSINGLE_CLASS);
	matlabInput.SetData(phaseImage,960*1280);
	myUnwrap(matlabInput);
	matlabOutput.GetData(phaseImage, 960 * 1280);
	//别忘了调整matlab\extern的地址
	realWrite("phase image after unwrapping", phaseImage, 1280, "..\ouput_text\phase_image2.txt");
	*/
	if (cudaSuccess != cudaMemcpy(dev_phaseImage, phaseImage, testImage->imagePixels * sizeof(float), cudaMemcpyHostToDevice))
		cout << "cuda memory cpy error!" << endl;

	float xConf, yConf, xSum = 0, ySum = 0, vecStep, sum2=0, mean2;
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
	createXConfVec <<<1,1280 >>> (xConf,vecStep,dev_xConfVec,1280);
	if (cudaSuccess != cudaGetLastError())
		printf("xConf vec create Error!\n");

	forPhaseImage<<<gridSize,blockSizeL>>>(mean2, dev_xConfVec, dev_phaseImage, testImage->imagePixels);
	cudaFree(dev_xConfVec);
	cudaThreadSynchronize();
	if (cudaSuccess != cudaMemcpy(phaseImage, dev_phaseImage, testImage->imagePixels * sizeof(float), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error!" << endl;
	//
	float* height1 = (float*)malloc(sizeof(float) * testImage->imagePixels);
	cufftReal* dev_height;
	cudaMalloc((void**)& dev_height, sizeof(float) * testImage->imagePixels);
	calHeight <<<gridSize, blockSizeL >>> (dev_phaseImage,mean2,dev_height,testImage->imagePixels);
	if (cudaSuccess != cudaGetLastError())
		printf("cal height Error!\n");
	cudaThreadSynchronize();
	if (cudaSuccess != cudaMemcpy(height1, dev_height, testImage->imagePixels * sizeof(float), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error!" << endl;

	float* outputImage = (float*)malloc(sizeof(float) * testImage->imagePixels);
	float* dev_output;
	if (cudaSuccess != cudaMalloc((void**)& dev_output, sizeof(float)* testImage->imagePixels))
		cout << "cuda malloc error!" << endl;
	calOutputImage<<<gridSize, blockSizeL >>>(dev_height, dev_output, testImage->imagePixels);
	if (cudaSuccess != cudaGetLastError())
		printf("output image create Error!\n");
	cudaThreadSynchronize();
	if (cudaSuccess != cudaMemcpy(outputImage, dev_output, testImage->imagePixels * sizeof(float), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error!" << endl;

	cudaFree(dev_output);
	cudaFree(dev_height);

	return height1;
}

void imageFileWrite(float* input, char* filename) {
	/*
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
		*/
}

void realWrite(const char* title, float* input , int width, const char* filename) {

	ofstream outFile;
	outFile.open(filename);
	outFile << title << ": " << endl;
	outFile.setf(ios::fixed, ios::floatfield);
	outFile.precision(7);
	outFile << "line 0: ";
	for (int i = 0; i < width; i++) {
		outFile << i << " | ";
	}
	outFile << endl;
	for (int y = 0, int sum = 0; y < 960; y++) {
		outFile << "line " << y << ": ";
		for (int x = 0; x < width; x++) {
			outFile << input[sum] << " | ";
			sum++;
		}
		outFile << endl;
	}

}

void complexWrite(const char* title, float2* input, int width, const char* filename) {
	ofstream outFile;
	outFile.open(filename);
	outFile << title << ": " << endl;
	outFile.setf(ios::fixed, ios::floatfield);
	outFile.precision(7);
	outFile << "line 0: ";
	for (int i = 0; i < width; i++) {
		outFile << i << "|" << i << "i"<< " | ";
	}
	outFile << endl;
	for (int y = 0, int sum = 0; y < 960; y++) {
		outFile << "line " << y << ": ";
		for (int x = 0; x < width; x++) {
			outFile << input[sum].x << "|" << input[sum].y << "i" << " | ";
			sum++;
		}
		outFile << endl;
	}
}

void errorHandle( int input) {
	switch (input)
	{
	case CUFFT_ALLOC_FAILED :
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
		cout << "Success" << endl;
	}
}