#include<iostream>
#include"stdio.h"
#include<cmath>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int suma(int *); //Función que va ejecutar la suma de los puntos con el filtro, regresando el resultado de la suma, y recibiendo 
//un arreglo con los puntos que van hacer la suma

int filtro[9] = {1 , 1, 1, 1, -8, 1, 1, 1, 1}; //Filtro o máscara para la suma de puntos laplaceana

int main( int argc, char** argv )  {

    Mat src, gray, dst; //Variable Mat que guarda el src, los puntos de la imagen original, gray para transformar a grises la imagen y dst como la imagen resultado de la suma laplaceana
    src = imread( "salon.jpg" ); //Carga de la imagen
    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT ); /// Quitando el 'ruido' haciendo un filtro gausseano de OpenCV, esto para facilitar la detección de puntos
    cvtColor( src, gray, CV_RGB2GRAY ); //conversión de la imagen a grises
    dst = gray.clone(); //clonar puntos para dst para que tenga el tamañao de la imagen original
    int *pedazoDeMatriz = (int *) malloc(3 * 3 * sizeof(int)); //Un arreglo que va guardar los puntos de alrededor del punto para la suma

    //For para hacer la suma en todos los puntos
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
            dst.at<uchar>(i, j) = suma(pedazoDeMatriz); //Llamar función suma con la matriz, regresar y guardarla en los puntos de la imagen resultado
        }

    imshow("Original", src); //carga de la imagen original
    imshow( "Resultado", dst ); //carga de la imagen resultado
    free(pedazoDeMatriz); //liberar el arreglo temporal
    waitKey(0);
    return 0;
}

//Función de la suma laplaceana
int suma(int *m) {

    int res = 0; //inicialización del resultado

    //suma de los puntos aplicando el filtro laplaceano
    for (int i = 0; i < 9; ++i)
        res += m[i] * filtro[i];

    if (res > 255)
        return 255; //Unidad máxima para rgb 255, si es más devolver el valor máximo
    else if (res < 0)
        return 0; //UNidad mínima para rgb 0, si es menor devolver 0
    else
        return res; //retorno del valor de la suma si no fue mayor a 255 y no fue menor de 0
}
