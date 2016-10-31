#include "APSPtest.h"
#include <stdio.h>
#include <stdlib.h>

#include "MatUtil.h"

int main(int argc, char **argv)
{
struct timeval tv1,tv2;
gettimeofday(&tv1,NULL);
APSPSerial(argc, argv);
gettimeofday(&tv2,NULL);
printf("Elasped time = %ld usecs\n", (tv2.tv_sec -tv1.tv_sec)*1000000+tv2.tv_usec-tv1.tv_usec);


return 0;
}
