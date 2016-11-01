/* Implementation of Floyd-Warshall Algorithm in CUDA
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include<sys/time.h>
#include "MatUtil.h"
#include "MatUtil.c"


// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);
void shortestPath(int argc, char **argv);
extern "C"
void computeGold(float *reference, float *idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel(int *result, int N, int k)
{
	//this part needs to be changed
	const int num = 100;
	__shared__ int ds_element_ik;
	__shared__ int ds_Column_k[100];
	__shared__ int ds_row_j[100];
	
	
    int j;
	int Row = blockIdx.x*blockDim.x+threadIdx.x;
	
	ds_element_ik = result[Row*N+k];
	
	for(int tmp = 0;tmp<N;tmp++)
	{
		ds_Column_k[tmp] = result[k*N+tmp];
		ds_row_j[tmp] = result[Row*N+tmp];
	}
	
		for(j = 0;j<N;j++)
		{
			if(ds_element_ik == -1 || ds_Column_k[j] == -1)
				continue;
			else
				if(ds_row_j[j] == -1)
				{
				result[Row*N+j] = ds_element_ik+ds_Column_k[j];
				}
			else
				result[Row*N+j] =  (ds_row_j[j] < ds_element_ik+ds_Column_k[j])? ds_row_j[j]:(ds_element_ik+ds_Column_k[j]);
			
		}

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    shortestPath(argc, argv);
}


void shortestPath(int argc, char **argv)
{
	//bool bTestResult = true;

    printf("%s Starting...\n\n", argv[0]);
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
    	int devID = findCudaDevice(argc, (const char **)argv);

    	StopWatchInterface *timer = 0;
    	sdkCreateTimer(&timer);
    	sdkStartTimer(&timer);
	
	printf("Starting to generate random matrix for input...\n");	
	fflush(stdin);
	//struct timeval tv1,tv2,tv3,tv4;
	//generate a random matrix.
	size_t N = atoi(argv[1]);
	int *mat = (int*)malloc(sizeof(int)*N*N);
	GenMatrix(mat, N);
	printf("Finished generating the test data....\n");	


	int *result = (int*)malloc(sizeof(int)*N*N);
	//compute the reference result.
	int *ref = (int*)malloc(sizeof(int)*N*N);
	memcpy(ref, mat, sizeof(int)*N*N);
	//gettimeofday(&tv1,NULL);
	ST_APSP(ref, N);
	//gettimeofday(&tv2,NULL);
	//fprintf(f,"%ld,", (tv2.tv_sec -tv1.tv_sec)*1000000+tv2.tv_usec-tv1.tv_usec);
	
	unsigned int mem_size = sizeof(int) * N*N;
	unsigned int num_threads = N;
	
	printf("Finished generating all the matrices\n");	
	int *d_mat;
	
	//Allocate memory for data matrix
	checkCudaErrors(cudaMalloc((void **) &d_mat, mem_size));
	
	// copy host memory to device
    checkCudaErrors(cudaMemcpy(d_mat, mat, mem_size,
                               cudaMemcpyHostToDevice));
	
	
	
	//Allocate memory for reference matrix in the device
	int *d_ref;
	
	//Allocate memory for data matrix
	checkCudaErrors(cudaMalloc((void **) &d_ref, mem_size));
	
	// copy host memory to device
    checkCudaErrors(cudaMemcpy(d_ref, ref, mem_size,
                               cudaMemcpyHostToDevice));
	
	//Allocate memory for the result
	int *d_result;
	
	//Allocate memory for data matrix
	checkCudaErrors(cudaMalloc((void **) &d_result, mem_size));
	
	// copy host memory to device
    checkCudaErrors(cudaMemcpy(d_result, mat, mem_size,
                               cudaMemcpyHostToDevice));
	
	 
	// setup execution parameters
    dim3  grid(1);
    dim3  threads(num_threads, 1, 1);
	
	int k = 0;
	for(k = 0;k<N;k++)
	{
	// execute the kernel
    testKernel<<< grid, threads >>>(d_result, N,k);
	}
	
	// copy host memory to device
    checkCudaErrors(cudaMemcpy(result,d_result, mem_size,
                               cudaMemcpyDeviceToHost));
	
	//compare your result with reference result
	if(CmpArray(result, ref, N*N))
		printf("Your result is correct.\n");
	else
		printf("Your result is wrong.\n");
		
	checkCudaErrors(cudaFree(d_result));
	free(result);
	checkCudaErrors(cudaFree(d_ref));
	free(ref);
	free(mat);
}
