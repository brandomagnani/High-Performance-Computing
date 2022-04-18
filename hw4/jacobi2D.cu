// $ nvcc -arch=sm_61 gpu03.cu -o gpu03 -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>


#define BLOCK_SIZE 32



// we assume that u, u_next, f are all stored in *Column Major* order


void init(double* u_next, double* u, double* f, double h, long N) {

  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (N+2)*(N+2); i++) {
    u[i]        = 0.0;
    u_next[i]   = 0.0;
    f[i]        = 1.0;
  }
}



void jacobi2D(double* u_next, double* u, double* f, double h, long N) {

  for ( long i = 1; i < N+1; i++ ) {
    for ( long j = 1; j < N+1; j++ ) {
      u_next[i+j*(N+2)] = 0.25*( h*h*f[i+j*(N+2)] + u[i-1+j*(N+2)]
                             + u[i+(j-1)*(N+2)] + u[i+1+j*(N+2)] + u[i+(j+1)*(N+2)] );
    }
  }
}


__global__
void jacobi2D_kernel(double* u_next, double* u, double* f, double h, long N) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if ( (i < N+1) && (j < N+1) && (i > 0) && (j > 0) ) {
    u_next[i+j*(N+2)] = 0.25*( h*h*f[i+j*(N+2)] + u[i-1+j*(N+2)]
			  + u[i+(j-1)*(N+2)] + u[i+1+j*(N+2)] + u[i+(j+1)*(N+2)] );
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

  long N   = 320;
  long itm = 10000;
  double h = 1.0 / double(N+1);

  double* u          = (double*) malloc((N+2) * (N+2) * sizeof(double));   // u, (N+2) x (N+2) matrix ( includes 2 ghost points per axis)
  double* u_next     = (double*) malloc((N+2) * (N+2) * sizeof(double));   // next u in iteration
  double* u_ref      = (double*) malloc((N+2) * (N+2) * sizeof(double));   // used for reference
  double* f          = (double*) malloc((N+2) * (N+2) * sizeof(double));   // right hand side
  double *temp;    // dummy pointer for pointer swapping

  init(u_next, u, f, h, N);  


  double tt = omp_get_wtime();
  for ( long l = 0; l < itm; l++ ) {
    jacobi2D(u_next, u, f, h, N);
    temp = u;
    u = u_next;
    u_next = temp;
  }
  for ( long i = 0; i < (N+2)*(N+2); i++ ) {
   u_ref[i] = u[i];
  }
  printf("CPU %f s\n", omp_get_wtime()-tt);

  dim3 BlockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 GridDim(N/BLOCK_SIZE, N/BLOCK_SIZE);
  
  double *u_next_d, *u_d, *f_d;
  cudaMalloc(&u_next_d, (N+2) * (N+2) *sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc(&u_d, (N+2) * (N+2) *sizeof(double));
  cudaMalloc(&f_d, (N+2) * (N+2) *sizeof(double));

  tt = omp_get_wtime();
  cudaMemcpy(u_next_d, u_next, (N+2) * (N+2) *sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(u_d, u, (N+2) * (N+2) *sizeof(double), cudaMemcpyHostToDevice);

  double ttinner = omp_get_wtime();
  for ( long l = 0; l < itm; l++ ) {
    jacobi2D_kernel<<<GridDim, BlockDim>>>(u_next, u, f, h, N);
    temp = u;
    u = u_next;
    u_next = temp;
  }

  cudaDeviceSynchronize();
  ttinner = omp_get_wtime() - ttinner;
  cudaMemcpy(u_next, u_next_d, (N+2) * (N+2) *sizeof(double), cudaMemcpyDeviceToHost);
  printf("GPU %f s, %f s\n", omp_get_wtime()-tt, ttinner);

  double err = 0;
  for (long i = 1; i < (N+1)*(N+1); i++) err += fabs(u[i]-u_ref[i]);
  printf("Error = %f\n", err);


  cudaFree(u_next_d);
  cudaFree(u_d);
  cudaFree(f_d);

  free(u_next);
  free(u);
  free(f);
  free(u_ref);






  return 0;

}




/*



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

*/
