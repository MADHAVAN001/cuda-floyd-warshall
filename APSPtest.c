#include <stdio.h>
#include <stdlib.h>
#include<sys/time.h>
#include "MatUtil.h"

int main(int argc, char **argv)
{
	struct timeval tv1,tv2,tv3,tv4;
	FILE *f = fopen("dynamic.txt", "a");
	if(f == NULL)
		printf("Error opening the file\n");
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
	gettimeofday(&tv1,NULL);
	ST_APSP(ref, N);
	gettimeofday(&tv2,NULL);
	fprintf(f,"%ld,", (tv2.tv_sec -tv1.tv_sec)*1000000+tv2.tv_usec-tv1.tv_usec);



	
	//compute your results
	int *result = (int*)malloc(sizeof(int)*N*N);
	memcpy(result, mat, sizeof(int)*N*N);
	//replace by parallel algorithm
	gettimeofday(&tv3,NULL);
	parallel(result, N);
	gettimeofday(&tv4,NULL);
	fprintf(f,"%ld\n", (tv4.tv_sec -tv3.tv_sec)*1000000+tv4.tv_usec-tv3.tv_usec);


	//compare your result with reference result
	if(CmpArray(result, ref, N*N))
		printf("Your result is correct.\n");
	else
		printf("Your result is wrong.\n");
	fclose(f);
}
