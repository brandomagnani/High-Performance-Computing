// g++ -std=c++11 -fopenmp -O3 omp_solved2.c -o omp_solved2 && ./omp_solved2

// BUGS: 1- need to make tid private, otherwise threads override each others
// .. 2- since al the threads are already busy, can't use pragma omp for schedule 
// .. inside the parallel region
/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
int nthreads, i, tid;
float total;

/*** Spawn parallel region ***/
#pragma omp parallel private(tid)  // make tid private 
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  total = 0.0;
  //#pragma omp for schedule(dynamic,10)  // threads are already fully taken, so can't use schedule here
  for (i=0; i<1000000; i++) 
     total = total + i*1.0;

  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
}
