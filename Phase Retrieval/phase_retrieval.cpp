#include"phase retrieval.h"

double** calPhase(double** testFilteredBaseband, double** calibFilteredBaseband , int height , int width) {
	double** finalImage = new double* [height];
	for (int n = 0; n < height; n++) {
		finalImage[n] = new double[width];
	}
	for (int i = 0; i < height; i++) 
		for (int j = 0; j < width; j++) 
			finalImage[i][j] = testFilteredBaseband[i][j] / calibFilteredBaseband[i][j];
		
	
}