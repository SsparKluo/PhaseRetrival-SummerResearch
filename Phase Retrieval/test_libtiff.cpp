
#include<iostream>

#include"..\libtiff\tiffio.h"

#include<fstream>

typedef unsigned char BYTE;

using namespace std;

int main()
{	
	char filename[10];
	ofstream outFile;
	outFile.open("test.txt");
	cin >> filename;

    TIFF* tif = TIFFOpen( filename , "r");
    if (tif) {
		int dircount = 0;
		do {
			dircount++;
		} while (TIFFReadDirectory(tif));
		printf("%d directories in %s\n", dircount, filename);
    }

	uint32 width, height, bps, spp , bit;
	uint16 photom;
	size_t npixels;
	unsigned int raster;

	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photom);

	npixels = width * height;

	cout << "Height= " << height << endl;
	cout << "width= " << width << endl;
	cout << photom << endl;

	int stripSize = TIFFStripSize(tif);
	cout << "strip size: " << stripSize << endl;

	uint32* image;
	image = (uint32*)malloc(sizeof(uint32) * npixels);
	TIFFReadRGBAImage(tif, width, height, image, 1);

	BYTE* RImage = new BYTE[npixels];
	uint32* rowPointerToSrc = image + (height - 1) * width;
	BYTE* rowPointerToR = RImage;

	for (int y = height - 1; y >= 0; y--) {
		uint32* colPointerToSrc = rowPointerToSrc;
		BYTE* colPointerToR = rowPointerToR;
		for (int x = 0 ; x <= width; x++) {
			colPointerToR[0] = (int)TIFFGetA(colPointerToSrc[0]);
			//TIFFGetB(colPointerToSrc[0]);//获取B通道
			//TIFFGetG(colPointerToSrc[0]);//获取G通道
			colPointerToR++;
			colPointerToSrc++;
		}
		rowPointerToR += width;
		rowPointerToSrc -= width;
	}

	int** imageNum = new int* [height];
	for (int i = 0; i < height; i++) {
		imageNum[i] = new int[width];
	}

	uint8** imageLine = new uint8 * [height];
	for (int i = 0; i < height; i++) {
		imageLine[i] = new uint8[width];
	}

	for (int i = 0; i < height; i++) {
		TIFFReadScanline(tif, imageLine[i], i);
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			imageNum[i][j] = imageLine[i][j];
		}
	}
	for (int i = 0; i < height; i++) {
		outFile << i << " line:" << endl;
		for (int j = 0; j < width; j++) {
			outFile << imageNum[i][j] << " ";
		}
		outFile << endl;
	}
	//
	for (int i = 0; i < height; i++) {
		outFile << i << " line: " << endl;
		for (int j =0 ; j < width; j++) {
			outFile << image[i*width+j] << " ";
		}
		outFile <<"\n"<< endl ;
	}
	//

	// 现在尝试输出

	TIFF* tiff;
	unsigned char* buf;
	tstrip_t strip;

	tiff = TIFFOpen("test_output.tif", "w");

	if (tiff)
	{
		TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, width);
		TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, height);

		TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 8);
		TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
		TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, 1);

		TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 1);
		TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, height);

		TIFFSetField(tiff, TIFFTAG_XRESOLUTION, 72);
		TIFFSetField(tiff, TIFFTAG_YRESOLUTION, 72);
		TIFFSetField(tiff, TIFFTAG_RESOLUTIONUNIT, RESUNIT_INCH);

		TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
		TIFFSetField(tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);

		buf = (BYTE*)malloc(sizeof(uint32) * width * height);

		for (int i = 0; i < height; i++) {
			TIFFWriteScanline(tiff, imageLine[i], i);
		}
		
		TIFFClose(tiff);
		TIFFClose(tif);

		free(buf);
		printf("Done!\n");
	}


}

