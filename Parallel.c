#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "MatUtil.h"

int rank;

void ParallelWarshall(int *weight, const int N, int numProcess);

int main(int argc, char **argv)
{
	
	if(argc != 2)
	{
		printf("Usage: test {N}\n");
		exit(-1);
	}

	//generate a random matrix.
	size_t N = atoi(argv[1]);
	int *mat = (int*)malloc(sizeof(int)*N*N);
	GenMatrix(mat, N);

	//compute the reference result.
	int *ref = (int*)malloc(sizeof(int)*N*N);
	memcpy(ref, mat, sizeof(int)*N*N);
	ParallelWarshall(ref, N, 1);

	//compute your results
	int *result = (int*)malloc(sizeof(int)*N*N);
	memcpy(result, mat, sizeof(int)*N*N);
	//replace by parallel algorithm
	ST_APSP(result, N);
	
	//compare your result with reference result
	
	if(CmpArray(ref, result, N*N))
		printf("Your result is correct.\n");
	else
		printf("Your result is wrong.\n"); 
	
	return 0;
}

void ParallelWarshall(int *weight, const int N, int numProcess)
{
	int rank = 0;
		
	MPI_Status status;
	int *ref = (int*)malloc(sizeof(int)*N*N);
	int *mat = (int*)malloc(sizeof(int)*N*N);
	memcpy(ref, mat, sizeof(int)*N*N);
	memset(ref,0,N*N);

	MPI_Init(NULL,NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcess);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if(rank == 0)
	{
		for(int i = 0;i<N;i++)
			for(int j =0; j<N;j++)
			{
			MPI_Bcast(&weight[i*N+j], 1, MPI_INT, 0,MPI_COMM_WORLD);
			}
	}
	
	int npes = N/numProcess;
	//printf("%d", npes);
	int INF = -1;
	for(int k = 0;k<N;k++)
	{
		for(int i = rank*npes;i<(rank+1)*npes;i++)
			for(int j = 0;j<N;j++)
			{
			if(weight[i*N+k] == INF || weight[k*N+j] == INF)
				continue;
			if(weight[i*N+j] == INF)
				weight[i*N+j] = weight[i*N+k] + weight[k*N+j];
			weight[i*N+j] = weight[i*N+j]<(weight[i*N+k] + weight[k*N+j])?weight[i*N+j]:(weight[i*N+k] + weight[k*N+j]);
			MPI_Reduce(&weight[i*N+j],&ref[i*N+j],1,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);
			//printf("%d\n",weight[i*N+j]);
			}	
	}
	
	weight = ref;	
	
}
