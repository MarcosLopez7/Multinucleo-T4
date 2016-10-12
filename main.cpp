#include<iostream>
#include"stdio.h"
#include<cmath>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int suma(int *);

int filtro[9] = {0 , 1, 0, 1, -4, 1, 0, 1, 0};

int main( int argc, char** argv )  {

    Mat src, gray, dst;
    src = imread( "salon.jpg" );
    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT ); /// Remove noise by blurring with a Gaussian filter
    cvtColor( src, gray, CV_RGB2GRAY );
    dst = gray.clone();
    int *pedazoDeMatriz = (int *) malloc(3 * 3 * sizeof(int));

    for (int i = 1; i < gray.rows - 1; ++i)
        for (int j = 1; j < gray.cols - 1; ++j) {
            pedazoDeMatriz[0] = gray.at<uchar>(i-1, j-1);
            pedazoDeMatriz[1] = gray.at<uchar>(i, j-1);
            pedazoDeMatriz[2] = gray.at<uchar>(i+1, j-1);
            pedazoDeMatriz[3] = gray.at<uchar>(i-1, j);
            pedazoDeMatriz[4] = gray.at<uchar>(i, j);
            pedazoDeMatriz[5] = gray.at<uchar>(i+1, j);
            pedazoDeMatriz[6] = gray.at<uchar>(i-1, j+1);
            pedazoDeMatriz[7] = gray.at<uchar>(i, j+1);
            pedazoDeMatriz[8] = gray.at<uchar>(i+1, j+1);
            dst.at<uchar>(i, j) = suma(pedazoDeMatriz);
        }

    imshow("Original", src);
    imshow( "Resultado", dst );
    free(pedazoDeMatriz);
    waitKey(0);
    return 0;
}

int suma(int *m) {

    int res = 0;

    for (int i = 0; i < 9; ++i)
        res += m[i] * filtro[i];

    if (res > 255)
        return 255;
    else if (res < 0)
        return 0;
    else
        return res;
}
