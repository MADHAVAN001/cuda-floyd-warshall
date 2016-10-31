////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Template project which demonstrates the basics on how to setup a project
* example application.
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
testKernel(float *g_idata, float *g_odata)
{
    // shared memory
    // the size is determined by the host application
    extern  __shared__  float sdata[];

    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    // read in input data from global memory
    sdata[tid] = g_idata[tid];
    __syncthreads();

    // perform some computations
    sdata[tid] = (float) num_threads * sdata[tid];
    __syncthreads();

    // write data to global memory
    g_odata[tid] = sdata[tid];
}


__global__ void
testKernel(int *result, int N)
{
    int j,k;
	int Row = blockIdx.x*blockDim.x+threadIdx.x;
	
	for(k = 0;k<N;k++)
	{
		for(j = 0;j<N;j++)
		{
			int l = Row*N+j;
			int m = Row*N+k;
			int n = k*N+j;
			if(result[m] == -1 || result[n] == -1)
				continue;
			else
				if(result[l] == -1)
				result[l] = result[m]+result[n];
			else
				result[l] =  (result[l] < result[m] + result[n])? result[l]:(result[m]+result[n]);
			__syncthreads();
		}
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
	
	// execute the kernel
    testKernel<<< grid, threads >>>(d_result, N);
	
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



////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char **argv)
{
    bool bTestResult = true;

    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    unsigned int num_threads = 32;
    unsigned int mem_size = sizeof(float) * num_threads;

    // allocate host memory
    float *h_idata = (float *) malloc(mem_size);

    // initalize the memory
    for (unsigned int i = 0; i < num_threads; ++i)
    {
        h_idata[i] = (float) i;
    }

    // allocate device memory
    float *d_idata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size,
                               cudaMemcpyHostToDevice));

    // allocate device memory for result
    float *d_odata;
    checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

    // setup execution parameters
    dim3  grid(1, 1, 1);
    dim3  threads(num_threads, 1, 1);

    // execute the kernel
    testKernel<<< grid, threads, mem_size >>>(d_idata, d_odata);

    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

    // allocate mem for the result on host side
    float *h_odata = (float *) malloc(mem_size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float) * num_threads,
                               cudaMemcpyDeviceToHost));

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    // compute reference solution
    float *reference = (float *) malloc(mem_size);
    computeGold(reference, h_idata, num_threads);

    // check result
    if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
    {
        // write file for regression test
        sdkWriteFile("./data/regression.dat", h_odata, num_threads, 0.0f, false);
    }
    else
    {
        // custom output handling when no regression test running
        // in this case check if the result is equivalent to the expected solution
        bTestResult = compareData(reference, h_odata, num_threads, 0.0f, 0.0f);
    }

    // cleanup memory
    free(h_idata);
    free(h_odata);
    free(reference);
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits   
    cudaDeviceReset();
    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
