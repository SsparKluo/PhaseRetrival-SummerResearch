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
	double maxIntensity, minIntensity;
	int height, width;
	int imagePixels;
	uint8** rawImage;
	double* imageData;
	int2 fftMaxPosition;
	double2* filteredBaseband;

	image() {
		rawImage = (uint8 * *)malloc(sizeof(uint8*) * 960);
		for (int i = 0; i < 960; i++) {
			rawImage[i] = (uint8*)malloc(sizeof(uint8) * 1280);
		}
		filteredBaseband =(double2*)malloc(sizeof(double)*2*1280*960);
	}
};

bool getImageInfo(image* targetImage);// Used for get the useful data in tiff image, and return false when the filename of the target image is invalid.
double* phaseRetrieval(image* calibImage, image* testImage);
void fourierFilterForCalib(image* calibImage);
int2 findMaxPoint(double* input); 
//void imageFileWrite(double* input, const char* filename, int height);
void complexWrite(const char* title, double2* input, int width, const char* filename);
void realWrite(const char* title, double* input, int width, const char* filename);
void errorHandle(int input);


__global__ void vectorAdd( double* a,  double* b, double* c, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements)
		c[i] = a[i] + b[i];
}

__global__ void vectorNumMultiple( double* a,  double* b, double* c, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements)
		c[i] = a[i] * b[i];
}

__global__ void numMultipleForComplex( double2* a,  double* b, double2* c, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x + threadIdx.y * gridDim.x;
	if (i < numElements) {
		c[i].x = a[i].x * b[i];
		c[i].y = a[i].y * b[i];
	}
}

__global__ void vectorNumdivide( double2* dividend,  double2* divisor, double2* output, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		output[i].x = dividend[i].x / divisor[i].x;
		output[i].y = dividend[i].y / divisor[i].y;
	}

}

__global__ void getAbsOfComplexMatric( cufftDoubleComplex* input, cufftDoubleReal* output, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements)
		output[i] = sqrt(input[i].x * input[i].x + input[i].y * input[i].y);
}

__global__ void createFilter( int padding,  int2 maxPoint, double* imageFilter, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		if ((x - maxPoint.x) * (x - maxPoint.x) + (y - maxPoint.y) * (y - maxPoint.y) <= padding * padding)
			imageFilter[i] = 1;
		else
			imageFilter[i] = 0;
	}
}

__global__ void FFTShift2D( cufftDoubleComplex* input, cufftDoubleComplex* output, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	int half = 640;
	if (i < numElements ) {
		if (x >= half ) {
			output[y + (x - half) * blockDim.y].x = input[y + x * blockDim.y].x;
			output[y + (x - half) * blockDim.y].y = input[y + x * blockDim.y].y;
		}
		else {
			output[y + (x + half) * blockDim.y].x = input[y + x * blockDim.y].x;
			output[y + (x + half) * blockDim.y].y = input[y + x * blockDim.y].y;
		}
	}
}

__global__ void IFFTShift2D( cufftDoubleComplex* input, cufftDoubleComplex* output, int numElements) {
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

__global__ void circShift2D( cufftDoubleComplex* input,  int2 maxPoint, cufftDoubleComplex* output, int numElements) {
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

__global__ void phaseCalculate( cufftDoubleComplex* input, cufftDoubleReal* output, int numElements) {

	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		output[i] = atan2(input[i].y, input[i].x);
	}
}

__global__ void createXConfVec( double xConf,  double vecStep, cufftDoubleReal* output, int numElements) {
	int i = threadIdx.x;
	if (i < numElements) {
		output[i] = xConf - i * vecStep;
	}
}

__global__ void forPhaseImage( double mean2,  cufftDoubleReal* xConfVec, cufftDoubleReal* input, int numElements) {
	int x = blockIdx.x;
	int y = threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		input[x + y * gridDim.x] = input[x + y * gridDim.x] + xConfVec[x] - mean2;
		if (input[x + y * gridDim.x] < 0)
			input[x + y * gridDim.x] = -input[x + y * gridDim.x];
	}
}

__global__ void calHeight( cufftDoubleReal* input, double mean2, cufftDoubleReal* output, int numElements) {
	int dn = 0.075;
	int i = blockIdx.x * blockDim.y + threadIdx.y;
	if (i < numElements) {
		output[i] = (input[i] * lambda / 2 / pi / dn - mean2) / 4 * 3;
	}
}

__global__ void calOutputImage( cufftDoubleReal* input, double* output, int numElements) {
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
	double* result = phaseRetrieval(&calibImage,&testImage);
}

bool getImageInfo( image* targetImage ) {
	
	TIFF* tif = TIFFOpen(targetImage->filename, "r");
	if (tif) {
		TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &targetImage->height);
		TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &targetImage->width);
		targetImage->imagePixels = targetImage->height * targetImage->width;
		targetImage->imageData = new double[targetImage->imagePixels];
		targetImage->rawImage = new uint8 * [targetImage->height];
		for (int i = 0; i < targetImage->height; i++) {
			targetImage->rawImage[i] =(uint8*)malloc(sizeof(uint8)*targetImage->width);
		}
		for (int i = 0; i < targetImage->height; i++) {
			TIFFReadScanline(tif, targetImage->rawImage[i], i);
		}
		double tempMax = 0;
		double tempMin = 10000;
		for (int i = 0; i < targetImage->width; i++) {
			for (int j = 0; j < targetImage->height; j++) {
				targetImage->imageData[i*targetImage->height + j] = ((double)targetImage->rawImage[j][i] / 256);
				//if (tempMax < targetImage->imageData[i * targetImage->width + j]) 
					//tempMax = targetImage->imageData[i * targetImage->width + j];
				//if (tempMin > targetImage->imageData[i * targetImage->width + j])
					//tempMin = targetImage->imageData[i * targetImage->width + j];
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

int2 findMaxPoint(double* input) {
	int2 tempPoint = {0,0};
	double tempMax = 0;
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
	dim3 blockSizeL(1, 960, 1), gridSize(1280, 1, 1), blockSizeS(1, 481, 1);

	double* calibAbsImage = new double[imageSizeS];

	//realWrite("input for fourierFiltered", calibImage->imageData, 1280, "../Debug/input_FF.txt");

	cufftDoubleReal* dev_calibImage, * dev_calibABSFFTShifted, * dev_calibImageFilter;
	cufftDoubleComplex* dev_calibFFT, * dev_calibFFTShifted, * dev_filteredCalibFFT, * dev_filterCircCalibFFT, * dev_calibFilteredBaseband;
	int n = cudaMalloc((void**)& dev_calibImage, sizeof(double) * calibImage->imagePixels);
	if (cudaSuccess != n)
		cout << "cuda malloc error1!" << endl;
	cout << n << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_calibFilteredBaseband, sizeof(double2) * calibImage->imagePixels))
		cout << "cuda malloc error2!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_calibImageFilter, sizeof(double) * imageSizeS))
		cout << "cuda malloc error3!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_calibABSFFTShifted, sizeof(double) * imageSizeS))
		cout << "cuda malloc error4!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_calibFFT, sizeof(double2) * imageSizeS))
		cout << "cuda malloc error5!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_filteredCalibFFT, sizeof(double2) * calibImage->imagePixels))
		cout << "cuda malloc error6!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_calibFFTShifted, sizeof(double2) * imageSizeS))
		cout << "cuda malloc error7!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_filterCircCalibFFT, sizeof(double2) * calibImage->imagePixels))
		cout << "cuda malloc error8!" << endl;

	cufftHandle FFT;
	cufftHandle IFFT;
	errorHandle(cufftPlan2d(&FFT, 1280, 960, CUFFT_R2C));
	errorHandle(cufftPlan2d(&IFFT, 1280, 960, CUFFT_C2C));

	if (cudaSuccess != cudaMemcpy(dev_calibImage, calibImage->imageData, calibImage->imagePixels * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice))
		cout << "cuda memory cpy error!" << endl;

	errorHandle(cufftExecD2Z(FFT, dev_calibImage, dev_calibFFT));
	cudaDeviceSynchronize();
	double2* tempOut = (double2*)malloc(sizeof(double2) * 1280 * 481);
	int a = cudaMemcpy((void*)tempOut, (void*)dev_calibFFT, imageSizeS * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	if (cudaSuccess != a)
		cout << "cuda memory cpy error!" << endl;
	cout << a << endl;
//	complexWrite("temp dubug info", tempOut, 1280, "../Debug/tempout.txt");

	FFTShift2D <<< gridSize, blockSizeS >>> (dev_calibFFT, dev_calibFFTShifted, imageSizeS);
	if (cudaSuccess != cudaGetLastError())
		printf("FFTShift error!\n");
	
	getAbsOfComplexMatric <<< gridSize, blockSizeS >>> (dev_calibFFTShifted, dev_calibABSFFTShifted, imageSizeS);
	if (cudaSuccess != cudaGetLastError())
		printf("get abs error!\n");


	int b = cudaMemcpy((void*)calibAbsImage, (void*)dev_calibABSFFTShifted, imageSizeS * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
	if (cudaSuccess != b)
		cout << "cuda memory cpy error!" << endl;
	cout << b << endl;

	//imageFileWrite( calibAbsImage ,"calib_abs.tif", 481);

	calibImage->fftMaxPosition = findMaxPoint(calibAbsImage);

	createFilter <<<gridSize, blockSizeS >>> (80, calibImage->fftMaxPosition, dev_calibImageFilter, imageSizeS);
	if (cudaSuccess != cudaGetLastError())
		printf("filter create error!\n");
	numMultipleForComplex <<<gridSize, blockSizeS >>> (dev_calibFFT, dev_calibImageFilter, dev_filteredCalibFFT, imageSizeS);
	if (cudaSuccess != cudaGetLastError())
		printf("matrix num multiple error!\n");
	circShift2D <<<gridSize, blockSizeL >>> (dev_filteredCalibFFT, calibImage->fftMaxPosition, dev_filterCircCalibFFT, imageSizeS);
	if (cudaSuccess != cudaGetLastError())
		printf("circle shift error!\n");
	IFFTShift2D <<<gridSize, blockSizeL >>> (dev_filterCircCalibFFT, dev_filteredCalibFFT, imageSizeS);
	if (cudaSuccess != cudaGetLastError())
		printf("IFFT shift error!\n");
	cufftExecZ2Z(IFFT, dev_filteredCalibFFT, dev_calibFilteredBaseband,CUFFT_INVERSE);
	cudaThreadSynchronize();
	
	if (cudaSuccess != cudaMemcpy(calibImage->filteredBaseband, dev_calibFilteredBaseband, (calibImage->imagePixels + 960) * sizeof(double), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error!" << endl;

	//complexWrite("calib filtered baseband", calibImage->filteredBaseband, 1280, "..\ouput_text\calib_filtered_baseband.txt");

	if (cudaSuccess != cudaFree(dev_calibImage))
		cout<<"cude meomory free error!"<<endl;
	if (cudaSuccess != cudaFree(dev_calibABSFFTShifted))
		cout << "cude meomory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_calibImageFilter))
		cout << "cude meomory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_calibFilteredBaseband))
		cout << "cude meomory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_calibFFT))
		cout << "cude meomory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_calibFFTShifted))
		cout << "cude meomory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_filteredCalibFFT))
		cout << "cude meomory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_filterCircCalibFFT))
		cout << "cude meomory free error!" << endl;

}

double* phaseRetrieval(image* calibImage, image* testImage) {
	cout << "Part: phase retrieval" << endl;
	int imageSizeS = 1280 * 481;
	dim3 blockSizeL(1, 960, 1), gridSize(1280, 1, 1), blockSizeS(1, 481, 1);

	double* testAbsImage = (double*)malloc(sizeof(double) * imageSizeS);

	cufftDoubleReal* dev_testImage, * dev_testABSFFTShifted, * dev_testImageFilter;
	cufftDoubleComplex* dev_testFFT, * dev_testFFTShifted, * dev_filteredTestFFT, * dev_filterCircTestFFT, * dev_testFilteredBaseband;
	if (cudaSuccess != cudaMalloc((void**)& dev_testImage, sizeof(double) * testImage->imagePixels))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_testFilteredBaseband, sizeof(double2) * testImage->imagePixels))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_testImageFilter, sizeof(double) * imageSizeS))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_testABSFFTShifted, sizeof(double) * imageSizeS))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_testFFT, sizeof(double2) * imageSizeS))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_filteredTestFFT, sizeof(double2) * testImage->imagePixels))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_testFFTShifted, sizeof(double2) * imageSizeS))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_filterCircTestFFT, sizeof(double2) * testImage->imagePixels))
		cout << "cuda malloc error!" << endl;

	cufftHandle FFT;
	errorHandle(cufftPlan2d(&FFT, 1280, 960, CUFFT_R2C));
	cufftHandle IFFT;
	errorHandle(cufftPlan2d(&IFFT, 1280, 960, CUFFT_C2C));

	if (cudaSuccess != cudaMemcpy(dev_testImage, testImage->imageData, testImage->imagePixels * sizeof(double), cudaMemcpyHostToDevice))
		cout << "cuda memory cpy error!" << endl;
	//
	errorHandle(cufftExecD2Z(FFT, dev_testImage, dev_testFFT));
	FFTShift2D <<< gridSize, blockSizeS >>> (dev_testFFT, dev_testFFTShifted, imageSizeS);
	if (cudaSuccess != cudaGetLastError())
		printf("FFT shift Error!\n");
	getAbsOfComplexMatric <<< gridSize, blockSizeS >>> (dev_testFFTShifted, dev_testABSFFTShifted, imageSizeS);
	if (cudaSuccess != cudaGetLastError())
		printf("ABS Error!\n");
	cudaDeviceSynchronize();

	if (cudaSuccess != cudaMemcpy(testAbsImage, dev_testABSFFTShifted, (testImage->imagePixels / 2 + 960) * sizeof(double), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error!" << endl;

	//realWrite("test abs image", testAbsImage, 640, "..\ouput_text\test_abs_image.txt");

	testImage->fftMaxPosition = findMaxPoint(testAbsImage);
	createFilter <<<gridSize, blockSizeS >>> (80, testImage->fftMaxPosition, dev_testImageFilter, imageSizeS);
	if (cudaSuccess != cudaGetLastError())
		printf("filter create Error!\n");
	numMultipleForComplex <<<gridSize, blockSizeS >>> (dev_testFFT, dev_testImageFilter, dev_filteredTestFFT, imageSizeS);
	if (cudaSuccess != cudaGetLastError())
		printf("multiple Error!\n");
	circShift2D <<<gridSize, blockSizeL >>> (dev_filteredTestFFT, testImage->fftMaxPosition, dev_filterCircTestFFT, imageSizeS);
	if (cudaSuccess != cudaGetLastError())
		printf("circ shift Error!\n");
	IFFTShift2D <<<gridSize, blockSizeL >>> (dev_filterCircTestFFT, dev_filteredTestFFT, imageSizeS);
	if (cudaSuccess != cudaGetLastError())
		printf("IFFT shift Error!\n");
	errorHandle(cufftExecZ2Z(IFFT, dev_filteredTestFFT, dev_testFilteredBaseband,CUFFT_INVERSE));
	cudaThreadSynchronize();
	cudaMemcpy(testImage->filteredBaseband, dev_testFilteredBaseband, (testImage->imagePixels) * sizeof(double), cudaMemcpyDeviceToHost);

	complexWrite("test filtered baseband", testImage->filteredBaseband, 1280, "../Debug/test_filtered_baseband.txt");

	if (cudaSuccess != cudaFree(dev_testImage))
		cout << "cude meomory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_testABSFFTShifted))
		cout << "cude meomory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_testImageFilter))
		cout << "cude meomory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_testFFT))
		cout << "cude meomory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_testFFTShifted))
		cout << "cude meomory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_filteredTestFFT))
		cout << "cude meomory free error!" << endl;
	if (cudaSuccess != cudaFree(dev_filterCircTestFFT))
		cout << "cude meomory free error!" << endl;

	double* phaseImage = (double*)malloc(sizeof(double) * testImage->imagePixels);
	cufftDoubleReal* dev_phaseImage;
	cufftDoubleComplex* dev_calibFilteredBaseband,* dev_finalImage;
	if (cudaSuccess != cudaMalloc((void**)& dev_phaseImage, sizeof(double) * calibImage->imagePixels))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_calibFilteredBaseband, sizeof(double2) * calibImage->imagePixels))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMalloc((void**)& dev_finalImage, sizeof(double2) * calibImage->imagePixels))
		cout << "cuda malloc error!" << endl;
	if (cudaSuccess != cudaMemcpy(dev_calibFilteredBaseband, calibImage->filteredBaseband, testImage->imagePixels * sizeof(double2), cudaMemcpyHostToDevice))
		cout << "cuda memory cpy error!" << endl;

	vectorNumdivide <<<gridSize,blockSizeL>>> (dev_testFilteredBaseband, dev_calibFilteredBaseband, dev_finalImage, calibImage->imagePixels) ;
	if (cudaSuccess != cudaGetLastError())
		printf("divide Error!\n");
	phaseCalculate <<<gridSize, blockSizeL >>> (dev_finalImage, dev_phaseImage, calibImage->imagePixels);
	if (cudaSuccess != cudaGetLastError())
		printf("phase calculate Error!\n");
	if (cudaSuccess != cudaMemcpy(phaseImage, dev_phaseImage, testImage->imagePixels * sizeof(double), cudaMemcpyDeviceToHost))
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
	if (cudaSuccess != cudaMemcpy(dev_phaseImage, phaseImage, testImage->imagePixels * sizeof(double), cudaMemcpyHostToDevice))
		cout << "cuda memory cpy error!" << endl;

	double xConf, yConf, xSum = 0, ySum = 0, vecStep, sum2=0, mean2;
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

	cufftDoubleReal* dev_xConfVec;
	dim3 tempGrid(1, 1, 1), tempBlock(1280, 1, 1);
	if (cudaSuccess != cudaMalloc((void**)& dev_xConfVec, sizeof(double) * 1280))
		cout << "cuda malloc error!" << endl;
	createXConfVec <<< tempGrid,tempBlock >>> (xConf,vecStep,dev_xConfVec,1280);
	if (cudaSuccess != cudaGetLastError())
		printf("xConf vec create Error!\n");

	forPhaseImage<<<gridSize,blockSizeL>>>(mean2, dev_xConfVec, dev_phaseImage, testImage->imagePixels);
	cudaFree(dev_xConfVec);
	cudaThreadSynchronize();
	if (cudaSuccess != cudaMemcpy(phaseImage, dev_phaseImage, testImage->imagePixels * sizeof(double), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error!" << endl;
	//
	double* height1 = (double*)malloc(sizeof(double) * testImage->imagePixels);
	cufftDoubleReal* dev_height;
	cudaMalloc((void**)& dev_height, sizeof(double) * testImage->imagePixels);
	calHeight <<<gridSize, blockSizeL >>> (dev_phaseImage,mean2,dev_height,testImage->imagePixels);
	if (cudaSuccess != cudaGetLastError())
		printf("cal height Error!\n");
	cudaThreadSynchronize();
	if (cudaSuccess != cudaMemcpy(height1, dev_height, testImage->imagePixels * sizeof(double), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error!" << endl;

	double* outputImage = (double*)malloc(sizeof(double) * testImage->imagePixels);
	double* dev_output;
	if (cudaSuccess != cudaMalloc((void**)& dev_output, sizeof(double)* testImage->imagePixels))
		cout << "cuda malloc error!" << endl;
	calOutputImage<<<gridSize, blockSizeL >>>(dev_height, dev_output, testImage->imagePixels);
	if (cudaSuccess != cudaGetLastError())
		printf("output image create Error!\n");
	cudaThreadSynchronize();
	if (cudaSuccess != cudaMemcpy(outputImage, dev_output, testImage->imagePixels * sizeof(double), cudaMemcpyDeviceToHost))
		cout << "cuda memory cpy error!" << endl;

	cudaFree(dev_output);
	cudaFree(dev_height);

	return height1;
}
/*
void imageFileWrite(double* input, char* filename, int height) {

	TIFF* tif = TIFFOpen(filename, "w");
	if (tif) {
		TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, height);
		TIFFSetField(tif, TIFFTAG_IMAGELENGTH, 1280);
		uint8** tempData = new uint8 * [960];
		for (int i = 0; i < height; i++) {
			tempData[i] = new uint8[1280];
			for (int j = 0; j < 1280; j++) {
				tempData[i][j] = (uint8)input[j * height + i];
			}
			TIFFWriteScanline(tif, tempData[i], i);
		}
	}
	else
		cout << filename << " can not be opened!" << endl;

}
*/
void realWrite(const char* title, double* input , int height, const char* filename) {

	ofstream outFile;
	outFile.open(filename);
	outFile << title << ": " << endl;
	outFile.setf(ios::fixed, ios::floatfield);
	outFile.precision(7);
	outFile << "line 0: ";
	for (int i = 0; i < 1280; i++) {
		outFile << i << " | ";
	}
	outFile << endl;
	for (int y = 0; y < height; y++) {
		outFile << "line " << y+1 << ": ";
		for (int x = 0; x < 960; x++) {
			outFile << input[ y + x * height] << " | ";
		}
		outFile << endl;
	}

}

void complexWrite(const char* title, double2* input, int height, const char* filename) {
	ofstream outFile;
	outFile.open(filename);
	outFile << title << ": " << endl;
	outFile.setf(ios::fixed, ios::floatfield);
	outFile.precision(7);
	outFile << "line 0: ";
	for (int i = 0; i < 1280; i++) {
		outFile << i << "|" << i << "i"<< " | ";
	}
	outFile << endl;
	for (int y = 0; y < height; y++) {
		outFile << "line " << y+1 << ": ";
		for (int x = 0; x < 1280; x++) {
			outFile << input[y + x * height].x << "|" << input[y + x * height].y << "i" << " | ";
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