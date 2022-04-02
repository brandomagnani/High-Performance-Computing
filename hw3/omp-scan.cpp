#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = A[0];
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
}


void scan_omp(long* prefix_sum, const long* A, long n, long p) {
  if (n == 0) return;

  long *sum_vec; // array containing sum produced by each thread

  #pragma omp parallel num_threads(p) 
  {
    long tid = omp_get_thread_num();
    #pragma omp single
    sum_vec = (long*) malloc((p+1) * sizeof sum_vec);
    sum_vec[0] = 0;

    long sum = 0;  // this will be private, each thread has its own copy
    #pragma omp for schedule(static)   // parallelize for loop, static chunks
    for (long i=0; i<n; i++) {
      sum += A[i];
      prefix_sum[i] = sum;
    }
    sum_vec[tid+1] = sum;    
    
    #pragma omp barrier  // we need to have all threads done, so that sum_vec is correct

    long correction = 0;
    for (long i=0; i<tid+1; i++) {  // compute correction for current thread
      correction += sum_vec[i];
    } 
    
    #pragma omp for schedule(static)  // add correction to entries dealt by current thread 
    for (long i=0; i<n; i++) {
      prefix_sum[i] += correction;
    }
  }
  free(sum_vec);
}



int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));

  long  p  = 4;  // number of threads

  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N, p);
  printf("parallel-scan      = %fs\n", omp_get_wtime() - tt);
  printf("number of threads  = %ld\n", p);
  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
