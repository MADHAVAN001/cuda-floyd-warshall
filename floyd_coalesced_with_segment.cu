#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

extern "C" {
#include "MatUtil.h"
}

__device__
int Min(int a, int b) { return a < b ? a : b; }

__global__
void NaiveFloydWarshall(int* mat, int k, int N, int segment_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (segment_size*idx < N*N) {
        for (int offset = 0; offset < segment_size && offset + segment_size*idx < N*N; ++offset) {
            int i = (segment_size*idx + offset)/N;
            int j = segment_size*idx + offset - i*N;
            if (mat[i*N + k] != -1 && mat[k*N + j] != -1) {
                if (mat[i*N+j] == -1) {
                    mat[i*N+j] = mat[i*N + k] + mat[k*N +j];
                } else {
                    mat[i*N+j] = Min(mat[i*N + k] + mat[k*N + j], mat[i*N+j]);
                }
            }
        }
    }
}

// Each thread will access 'segment_size' values to improve coalescing.
// Each block now handles thread_per_block * segment_size values.
// Hence the number of blocks needed is N*N/(segment_size*thread_per_block).
void NaiveFloydWarshallDriver(int* mat, int N, int thread_per_block, int segment_size) {
    int* cuda_mat;
    int size = sizeof(int) * N * N;
    cudaMalloc((void**) &cuda_mat, size);
    cudaMemcpy(cuda_mat, mat, size, cudaMemcpyHostToDevice);
    int num_block = ceil(1.0*N*N/(thread_per_block*segment_size));
    for (int k = 0; k < N; ++k) {
        NaiveFloydWarshall<<<num_block, thread_per_block>>>(cuda_mat, k, N, segment_size);
    }
    cudaMemcpy(mat, cuda_mat, size, cudaMemcpyDeviceToHost);
    cudaFree(cuda_mat);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    if(argc != 5) {
        printf("Usage: test {N} {run_sequential_check: 'T' or 'F'} {thread_per_block} {segment_size}\n");
        exit(-1);
    }
    char run_sequential_check = argv[2][0];
    int thread_per_block = atoi(argv[3]);
    int segment_size = atoi(argv[4]);
    //generate a random matrix.
    size_t N = atoi(argv[1]);
    int *mat = (int*)malloc(sizeof(int)*N*N);
    GenMatrix(mat, N);

    //compute your results
    int *result = (int*)malloc(sizeof(int)*N*N);
    memcpy(result, mat, sizeof(int)*N*N);
    //replace by parallel algorithm
    NaiveFloydWarshallDriver(result, N, thread_per_block, segment_size);
    
    //compare your result with reference result
    if (run_sequential_check == 'T') {
        int *ref = (int*)malloc(sizeof(int)*N*N);
        memcpy(ref, mat, sizeof(int)*N*N);
        ST_APSP(ref, N);
        if(CmpArray(result, ref, N*N))
            printf("Your result is correct.\n");
        else
            printf("Your result is wrong.\n");
    }
}
