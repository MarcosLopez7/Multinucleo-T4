#include<iostream>
#include"stdio.h"
#include<cmath>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <cuda.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int suma(int *);

__global__ void laplaceadno1D(int *src, int *dst, int w, int h) {
	
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int filtro[9] = {1 , 1, 1, 1, -8, 1, 1, 1, 1};

    if(i > 0 && h - 1> i)
    	for(int j = 1; j < w - 1; j++) {
		int suma = src[(i - 1) * w + j - 1] * filtro[0] + src[i * w + j - 1] * filtro[1] + src[(i + 1) * w + j - 1] * filtro[2];
		suma += src[(i - 1) * w + j] * filtro[3] + src[i * w + j] * filtro[4] + src[(i + 1) * w + j] * filtro[5]; 
		suma += src[(i - 1) * w + j + 1] * filtro[6] + src[i * w + j + 1] * filtro[7] + src[(i + 1) * w + j + 1] * filtro[8];	    
		dst[i * w + j] = suma;
    	}

}

__global__ void laplaceadno2D(int *src, int *dst, int w, int h) {

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int filtro[9] = {1 , 1, 1, 1, -8, 1, 1, 1, 1};

    if(i > 0 && h - 1 > i)
        if(j > 0 && j < w - 1) {
                int suma = src[(i - 1) * h + j - 1] * filtro[0] + src[i * h + j - 1] * filtro[1] + src[(i + 1) * h + j - 1] * filtro[2];
                suma += src[(i - 1) * h + j] * filtro[3] + src[i * h + j] * filtro[4] + src[(i + 1) * h + j] * filtro[5];
                suma += src[(i - 1) * h + j + 1] * filtro[6] + src[i * h + j + 1] * filtro[7] + src[(i + 1) * h + j + 1] * filtro[8];
                dst[i * w + j] = suma;

        }

}

int main( int argc, char** argv )  {

    Mat src, gray, dst;
    src = imread( "salon.jpg" );
    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT ); /// Remove noise by blurring with a Gaussian filter
    cvtColor( src, gray, CV_RGB2GRAY );
    dst = gray.clone();
    int *srcA = (int *) malloc(src.rows * src.cols * sizeof(int));
    int *dstA = (int *) malloc(src.rows * src.cols * sizeof(int));
    int *dev_src, *dev_dst;
    cudaMalloc( (void**)&dev_src, src.rows * src.cols * sizeof(int) );
    cudaMalloc( (void**)&dev_dst, src.rows * src.cols * sizeof(int) );

    if (argc != 4){
	cout << "Se esperaba la opcion, el numero de bloques y threads, en ese orden\n";	
	exit(-1);
    }
 
    int opcion = atoi(argv[1]);
    int blocks = atoi(argv[2]);    
    int threads = atoi(argv[3]);

    for(int i = 0; i < src.rows; i++) 
	for (int j = 0; j < src.cols; j++)
	    srcA[i * src.cols + j] = src.at<uchar>(i, j);

    cudaMemcpy( dev_src, srcA, src.rows * src.cols * sizeof(int), cudaMemcpyHostToDevice );

    if (opcion == 1) 
	laplaceadno1D<<<blocks, threads>>>(dev_src, dev_dst, src.cols, src.rows);
    else if(opcion == 2)
	laplaceadno2D<<<blocks, threads>>>(dev_src, dev_dst, src.cols, src.rows);    	

    cudaMemcpy( dstA, dev_dst, src.rows * src.cols * sizeof(int), cudaMemcpyDeviceToHost );

    for(int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++){
	    if (dstA[i * src.cols + j] > 255)
		dst.at<uchar>(i, j) = 255; 
	    else if (0 > dstA[i * src.cols + j])
		dst.at<uchar>(i, j) = 0; 
	    else
		dst.at<uchar>(i, j) = dstA[i * src.cols + j]; 
	 }
   }

    imshow("Original", src);
    imshow( "Resultado", dst );

    waitKey(0);
    free(srcA);
    free(dstA);
    return 0;
}

