/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>


int main()
{
  int N = 100000;

  // Input arrays
  double a[N];
  double b[N];
  // Output arrays
  double c[N];

  double sum;

  for (int i=0; i<N; i++){
    a[i]=sin((double)(i+1))*sin((double)(i+1));
    b[i]=cos((double)(i+1))*cos((double)(i+1));
    c[i]=0.0;
 }
  double tstart = omp_get_wtime();
  for (int j = 0; j< N; j++){
    c[j] = a[j] + b[j];
  }

  sum = 0.0;
  for (int i=0; i<N; i++){
    sum += c[i];
  }
  sum = sum/(double)N;
  printf("Runtime is: %lf secs\n", omp_get_wtime()-tstart);

  printf("Final result: %lf\n",sum);

}