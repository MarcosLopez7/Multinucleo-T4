#include<iostream>
#include"stdio.h"
#include<cmath>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <cuda.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//Kernel de CUDA para la función de la suma laplaceana em 1 dimensión, recibe una matriz con los puntos de la imagen general, la matriz de resultado, w como el ancho de la imagen, h como el largo de la imagen
__global__ void laplaceadno1D(int *src, int *dst, int w, int h) {
	
    int i = blockIdx.x*blockDim.x + threadIdx.x; //el id del thread en el que estamos trabajando en el kernel
    int filtro[9] = {1 , 1, 1, 1, -8, 1, 1, 1, 1}; //filtro laplaceano para la suma de puntos

    //Checar que el thread este dentro de los puntos permitidos en lo alto para que no de segmentation fault
    if(i > 0 && h - 1> i)
    	for(int j = 1; j < w - 1; j++) { //for para iterar sobre lo ancho de la imagen y obtener los puntos conforme pase el ciclo
		//variable de suma que va guardar el resultado de la suma con el filtro laplaceano
		int suma = src[(i - 1) * w + j - 1] * filtro[0] + src[i * w + j - 1] * filtro[1] + src[(i + 1) * w + j - 1] * filtro[2];
		suma += src[(i - 1) * w + j] * filtro[3] + src[i * w + j] * filtro[4] + src[(i + 1) * w + j] * filtro[5]; 
		suma += src[(i - 1) * w + j + 1] * filtro[6] + src[i * w + j + 1] * filtro[7] + src[(i + 1) * w + j + 1] * filtro[8];	    
		dst[i * w + j] = suma; //guardar resultado en la matriz de resultados
    	}

}


//Kernel de CUDA para la función de la suma laplaceana em 2 dimensión, recibe una matriz con los puntos de la imagen general, la matriz de resultado, w como el ancho de la imagen, h como el largo de la imagen
__global__ void laplaceadno2D(int *src, int *dst, int w, int h) {

    int i = blockIdx.x*blockDim.x + threadIdx.x; //el id del thread en el que estamos trabajando en el kernel en la dimensión X
    int j = blockIdx.y*blockDim.y + threadIdx.y; //el id del thread en el que estamos trabajando en el kernel en la dimensión Y
    int filtro[9] = {1, 1, 1, 1, -8, 1, 1, 1, 1};

//    printf("i: %d, \n", i);

    //Checar que el thread este dentro de los puntos permitidos en lo alto para que no de segmentation fault
    if(i > 0 && h - 1 > i)
	//Checar que el thread este dentro de los puntos permitidos en lo alto para que no de segmentation fault
        if(j > 0 && j < w - 1) {
                int suma = 0;//variable de suma que va guardar el resultado de la suma con el filtro laplaceano
		suma += (src[(i - 1) * w + j - 1] * filtro[0]) + (src[i * w + j - 1] * filtro[1]) + (src[(i + 1) * w + j - 1] * filtro[2]);
                suma += (src[(i - 1) * w + j] * filtro[3]) + (src[i * w + j] * filtro[4]) + (src[(i + 1) * w + j] * filtro[5]);
                suma += (src[(i - 1) * w + j + 1] * filtro[6]) + (src[i * w + j + 1] * filtro[7]) + (src[(i + 1) * w + j + 1] * filtro[8]);
                dst[i * w + j] = suma; //guardar resultado en la matriz de resultados
        }

}

int main( int argc, char** argv )  {

    float tiempo1, tiempo2;
    cudaEvent_t inicio1, fin1, inicio2, fin2;

    cudaEventCreate(&inicio1);
    cudaEventCreate(&fin1);
    cudaEventCreate(&inicio2);
    cudaEventCreate(&fin2);
    cudaEventRecord( inicio1, 0 );

    Mat src, gray, dst;//Variable Mat que guarda el src, los puntos de la imagen original, gray para transformar a grises la imagen y dst como la imagen resultado de la suma laplaceana
    src = imread( "salon.jpg" );//Carga de la imagen
    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT ); /// Quitando el 'ruido' haciendo un filtro gausseano de OpenCV, esto para facilitar la detección de puntos
    cvtColor( src, gray, CV_RGB2GRAY );//conversión de la imagen a grises
    dst = gray.clone();//clonar puntos para dst para que tenga el tamañao de la imagen original
    int *srcA = (int *) malloc(src.rows * src.cols * sizeof(int)); //inicialización de matriz que va copiar los puntos de la imagen original para luego pasarlo a un arreglo de CUDA
    int *dstA = (int *) malloc(src.rows * src.cols * sizeof(int)); //Inicialización de la matriz que va contener los resultados de la suma de la imagen y va recibir la copia del arreglo de CUDA
    int *dev_src, *dev_dst; //inicialización de los arreglos que va cargar los puntos pero en los kernel de CUDA
    //inicialización de la memoria
    cudaMalloc( (void**)&dev_src, src.rows * src.cols * sizeof(int) );
    cudaMalloc( (void**)&dev_dst, src.rows * src.cols * sizeof(int) );

    //verificar que haya 3 argumentos a la hora de ejcutar el programa
    if (argc != 4){
	cout << "Se esperaba la opcion, el numero de bloques y threads, en ese orden\n";	
	exit(-1);
    }
 
    //Variables que toman los parametros de ejecución del programa
    int opcion = atoi(argv[1]);
    int blocks = atoi(argv[2]);    
    int threads = atoi(argv[3]);

    //Copia de los puntos de la imagen original al arreglo
    for(int i = 0; i < gray.rows; i++) 
	for (int j = 0; j < gray.cols; j++)
	    srcA[i * gray.cols + j] = gray.at<uchar>(i, j);

    //Copia de los puntos de la imagen que están en el arreglo al arreglo de CUDA
    cudaMemcpy( dev_src, srcA, src.rows * src.cols * sizeof(int), cudaMemcpyHostToDevice );

    cudaEventRecord( inicio2, 0 );
    if (opcion == 1) //Ejecución de la función de la suma en 1 dimensión
	laplaceadno1D<<<blocks, threads>>>(dev_src, dev_dst, gray.cols, gray.rows);
    else if(opcion == 2){ //Ejecución de la función de la suma en 2 dimensiones
	laplaceadno2D<<<blocks, threads>>>(dev_src, dev_dst, gray.cols, gray.rows);    	
    }
    cudaEventRecord( fin2, 0); // Se toma el tiempo final.
    cudaEventSynchronize( fin2 ); // Se sincroniza
    cudaEventElapsedTime( &tiempo2, inicio2, fin2 );

    //Copia del arreglo de resultado de la suma de CUDA al arreglo 
    cudaMemcpy( dstA, dev_dst, src.rows * src.cols * sizeof(int), cudaMemcpyDeviceToHost );

    //For que va copiar los resultados de los puntos del arreglo a una variable Mat para que se puedra mostrar en la imagen
    for(int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++){ //Vamos a evaluar los resultados que arroja la matriz y ver que no se pase del límite de la escala RGB de 0 a 255
	    if (dstA[i * gray.cols + j] > 255) //Unidad máxima para rgb 255, si es más devolver el valor máximo
		dst.at<uchar>(i, j) = 255; 
	    else if (0 > dstA[i * gray.cols + j]) //UNidad mínima para rgb 0, si es menor devolver 0
		dst.at<uchar>(i, j) = 0; 
	    else //retorno del valor de la suma si no fue mayor a 255 y no fue menor de 0
		dst.at<uchar>(i, j) = dstA[i * gray.cols + j];

	 }
   }

    imshow("Original", src); //Mostrar imagen original
    imshow( "Resultado", dst ); //MOstrar imagen resultado

    waitKey(0); 
    //Liberación de la memoria normal y la de CUDA
    
    cudaFree( dev_src );
    cudaFree( dev_dst );
    free(srcA);
    free(dstA);

    cudaEventRecord( fin1, 0); // Se toma el tiempo final.
    cudaEventSynchronize( fin1 ); // Se sincroniza
    cudaEventElapsedTime( &tiempo1, inicio1, fin1 );

    printf("Tiempo de cálculo: %f , tiempo total: %f\n", tiempo2, tiempo1);

    return 0;
}

