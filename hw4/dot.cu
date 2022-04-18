// $ nvcc -arch=sm_61 gpu03.cu -o gpu03 -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

#define Nd  (1024*1024)
#define THREADS_PER_BLOCK 1024


void dot(double* c, const double* a, const double* b) {
  double sum = 0.0;
  
  #pragma omp parallel
  #pragma omp for reduction(+: sum)
  for (long i = 0; i < Nd; i++) {
    sum += a[i]*b[i];    // sum is private
  }
  *c = sum;
}


__global__
void dot_kernel(double* c, const double* a, const double* b) {
  __shared__ double temp[THREADS_PER_BLOCK];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  temp[threadIdx.x] = a[idx] * b[idx];

  __syncthreads();

  if ( 0 == threadIdx.x ) {
    double sum = 0.;
    for ( int i = 0; i < THREADS_PER_BLOCK; i++ ){
      sum += temp[i];
    }
    atomicAdd( c, sum );
  }

} 



void vec_add(double* c, const double* a, const double* b, long N){
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

__global__
void vec_add_kernel(double* c, const double* a, const double* b, long N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] = a[idx] + b[idx];
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}


int main(void) {
  double* a     = (double*) malloc(Nd * sizeof(double));
  double* b     = (double*) malloc(Nd * sizeof(double)); 
  double* c     = (double*) malloc( sizeof(double) );
  double* c_ref = (double*) malloc( sizeof(double) );
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < Nd; i++) {
    a[i] = 1.0;
    b[i] = 1.0;
  }
  *c     = 0.0;
  *c_ref = 0.0;

  double tt = omp_get_wtime();
  dot(c_ref, a, b);
  printf("CPU %f s\n", omp_get_wtime()-tt);

  double *a_d, *b_d, *c_d;
  cudaMalloc(&a_d, Nd*sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc(&b_d, Nd*sizeof(double));
  cudaMalloc(&c_d, sizeof(double));

  tt = omp_get_wtime();
  cudaMemcpy(a_d, a, Nd*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, Nd*sizeof(double), cudaMemcpyHostToDevice);

  double ttinner = omp_get_wtime();
  dot_kernel<<< Nd/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(c_d, a_d, b_d);
  cudaDeviceSynchronize();
  ttinner = omp_get_wtime() - ttinner;
  cudaMemcpy(c, c_d, sizeof(double), cudaMemcpyDeviceToHost);
  printf("GPU %f s, %f s\n", omp_get_wtime()-tt, ttinner);

  printf("c_ref = %f\n", *c_ref );
  printf("c = %f\n", *c );
  printf("Error = %f\n", fabs(*c - *c_ref));
  
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

  free(a);
  free(b);
  free(c);
  free(c_ref);

  return 0;

}


