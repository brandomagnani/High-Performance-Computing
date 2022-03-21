// g++ -std=c++11 -fopenmp -O3 omp_solved6.c -o omp_solved6

// BUG: sum had to be private inside the reduction region, so moved stuff around
// .. to have this

/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];

float dotprod()
{
int tid;
float sum=0.0;

#pragma omp parallel 
#pragma omp for reduction(+:sum)
  for (size_t i=0; i < VECLEN; i++)
    {
    sum = sum + (a[i]*b[i]);
    tid = omp_get_thread_num();
    printf("  tid= %d i=%d\n",tid,i);
    }
  return sum;
}


int main (int argc, char *argv[]) {
int i,tid;
float s;

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;

s = dotprod();

printf("Sum = %f\n",s);

}
