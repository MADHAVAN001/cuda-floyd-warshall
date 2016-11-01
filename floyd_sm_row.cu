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
void SharedMemoryFloydWarshall(int* mat, int k, int N) {
    __shared__ int dist_i_k;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        int dist_i_j = mat[i*N + j];
        int dist_k_j = mat[k*N + j];
        if (threadIdx.y == 0) {
            dist_i_k = mat[i*N + k];
        }
        __syncthreads();
        if (dist_i_k != -1 && dist_k_j != -1) {
            int new_dist = dist_i_k + dist_k_j;
            if (dist_i_j != -1) {
                new_dist = Min(new_dist, dist_i_j);
            }
            mat[i*N + j] = new_dist;
        }
    }
}

void SharedMemoryFloydWarshallDriver(int* mat, int N, dim3 thread_per_block) {
    int* cuda_mat;
    int size = sizeof(int) * N * N;
    cudaMalloc((void**) &cuda_mat, size);
    cudaMemcpy(cuda_mat, mat, size, cudaMemcpyHostToDevice);
    dim3 num_block(ceil(1.0*N/thread_per_block.x),
                   ceil(1.0*N/thread_per_block.y));
    for (int k = 0; k < N; ++k) {
        SharedMemoryFloydWarshall<<<num_block, thread_per_block>>>(cuda_mat, k, N);
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
        printf("Usage: test {N} {run_sequential_check: 'T' or 'F'} {segment_size}\n");
        exit(-1);
    }
    char run_sequential_check = argv[2][0];
    int segment_size = atoi(argv[3]);
    dim3 thread_per_block(1, segment_size);
    //generate a random matrix.
    size_t N = atoi(argv[1]);
    int *mat = (int*)malloc(sizeof(int)*N*N);
    GenMatrix(mat, N);

    //compute your results
    int *result = (int*)malloc(sizeof(int)*N*N);
    memcpy(result, mat, sizeof(int)*N*N);
    //replace by parallel algorithm
    SharedMemoryFloydWarshallDriver(result, N, thread_per_block);
    
    //compare your result with reference result
    if (run_sequential_check == 'T') {
        int *ref = (int*)malloc(sizeof(int)*N*N);
        memcpy(ref, mat, sizeof(int)*N*N);
        ST_APSP(ref, N);
        if(CmpArray(result, ref, N*N))
            printf("Your result is correct.\n");
        else
            printf("Your result is wrong.\n");
#ifdef PRINT_MATRIX
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("%d ", ref[i*N+j]);
            }
            printf("\n");
        }
#endif
    }

#ifdef PRINT_MATRIX
    printf("==RESULT==\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", result[i*N+j]);
        }
        printf("\n");
    }
#endif
}
