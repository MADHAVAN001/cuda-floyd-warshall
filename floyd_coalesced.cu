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
void NaiveFloydWarshall(int* mat, int k, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N*N) {
        int i = idx/N;
        int j = idx - i*N;
        if (mat[i*N + k] != -1 && mat[k*N + j] != -1) {
            if (mat[idx] == -1) {
                mat[idx] = mat[i*N + k] + mat[k*N +j];
            } else {
                mat[idx] = Min(mat[i*N + k] + mat[k*N + j], mat[idx]);
            }
        }
    }
}

void NaiveFloydWarshallDriver(int* mat, int N, int thread_per_block) {
    int* cuda_mat;
    int size = sizeof(int) * N * N;
    cudaMalloc((void**) &cuda_mat, size);
    cudaMemcpy(cuda_mat, mat, size, cudaMemcpyHostToDevice);
    int num_block = ceil(1.0*N*N/(thread_per_block));
    for (int k = 0; k < N; ++k) {
        NaiveFloydWarshall<<<num_block, (thread_per_block)>>>(cuda_mat, k, N);
    }
    cudaMemcpy(mat, cuda_mat, size, cudaMemcpyDeviceToHost);
    cudaFree(cuda_mat);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    if(argc != 4) {
        printf("Usage: test {N} {run_sequential_check: 'T' or 'F'} {thread_per_block}\n");
        exit(-1);
    }
    char run_sequential_check = argv[2][0];
    int thread_per_block = atoi(argv[3]);
    //generate a random matrix.
    size_t N = atoi(argv[1]);
    int *mat = (int*)malloc(sizeof(int)*N*N);
    GenMatrix(mat, N);

    //compute your results
    int *result = (int*)malloc(sizeof(int)*N*N);
    memcpy(result, mat, sizeof(int)*N*N);
    //replace by parallel algorithm
    NaiveFloydWarshallDriver(result, N, thread_per_block);
    
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
