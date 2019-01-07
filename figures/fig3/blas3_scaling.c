// simple strong-scaling test for BLAS3 (DGEMM)
// reference problem size = 52137
// we are testing 1, 2, 4, 8, & 16 cores on a 16-core node

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "mkl.h"

#define SIZE 52137

int main(void)
{
  MKL_INT size = SIZE;
  size_t n2 = (size_t)SIZE*(size_t)SIZE;
  double *mat1 = (double*)malloc(sizeof(double)*n2);
  double *mat2 = (double*)malloc(sizeof(double)*n2);
  double alpha = 1.0, beta = 0.0;
  char trans = 'N';

  for(size_t i=0 ; i<n2 ; i++)
  {
    mat1[i] = rand();
    mat2[i] = rand();
  }

  double time1 = omp_get_wtime();
  dgemm(&trans,&trans,&size,&size,&size,&alpha,mat1,&size,mat1,&size,&beta,mat2,&size);
  double time2 = omp_get_wtime();
  printf("time = %e s\n",time2-time1);
}

