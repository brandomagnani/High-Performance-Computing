// $ nvcc -arch=sm_61 gpu03.cu -o gpu03 -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

#define THREADS_PER_BLOCK 1024


// we assume all matrices are stored in row major order
// we assume the matrix 'mat' is of size (m x n)
// we assume the vector 'vec' is of size n
// we assume the vector 'out' is of size m

void matvec(double* out, double* mat, double* vec, long m, long n) {

  for ( long i = 0; i < m; i++ ) {
    double sum = 0.0;
    for ( long j = 0; j < n; j++ ) {
      sum += mat[i*n + j] * vec[j];
    }
    out[i] = sum;
  }
}


__global__ 
void matvec_kernel(double* out, double* mat, double* vec, long m, long n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  double sum = 0.0;

  if ( idx < m ) {
    for ( long j = 0; j < n; j++ ) {
      sum += mat[idx*n + j] * vec[j];
    }
    out[idx] = sum;
  }
}


void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}


int main(void) {

  long m = 1024*1024;
  long n = 1000;
  double* mat     = (double*) malloc( m * n * sizeof(double));
  double* vec     = (double*) malloc( n * sizeof(double));
  double* z  	  = (double*) malloc( m * sizeof(double));
  double* z_ref       = (double*) malloc( m * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < m*n; i++) {
    mat[i]   = 1.0;
  }
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < n; i++) {
    vec[i]   = 1.0;
  }
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < m; i++) {
    z[i]       = 0.0;
    z_ref[i]   = 0.0;
  }

  double tt = omp_get_wtime();
  matvec(z_ref, mat, vec, m, n);
  printf("CPU %f s\n", omp_get_wtime()-tt);

  double *mat_d, *vec_d, *z_d;
  cudaMalloc(&mat_d, m * n *sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc(&vec_d, n *sizeof(double));
  cudaMalloc(&z_d, m *sizeof(double));

  tt = omp_get_wtime();
  cudaMemcpy(mat_d, mat, m * n *sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(vec_d, vec, n *sizeof(double), cudaMemcpyHostToDevice);

  double ttinner = omp_get_wtime();
  matvec_kernel<<< m/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(z_d, mat_d, vec_d, m, n);
  cudaDeviceSynchronize();
  ttinner = omp_get_wtime() - ttinner;
  cudaMemcpy(z, z_d, m *sizeof(double), cudaMemcpyDeviceToHost);
  printf("GPU %f s, %f s\n", omp_get_wtime()-tt, ttinner);

  double err = 0;
  for (long i = 0; i < m; i++) err += fabs(z[i]-z_ref[i]);
  printf("Error = %f\n", err);

  //printf("z[0] = %f\n", z[0]);

  cudaFree(mat_d);
  cudaFree(vec_d);
  cudaFree(z_d);

  free(mat);
  free(vec);
  free(z);
  free(z_ref);

  return 0;
}


