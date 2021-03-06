// $ g++ -std=c++11 -fopenmp -O0 -march=native jacobi2D-omp.cpp && ./a.out
// $ g++ -std=c++11 -fopenmp -O3 -march=native jacobi2D-omp.cpp && ./a.out

// $ g++ -O3 -std=c++11 jacobi2D-omp.cpp && ./a.out

#include <stdio.h>
#include <cmath>
#include <chrono>
#include <cstring>
#include <stdlib.h>
#include <iostream>
#include <fstream>                // write the output to a file
#include "utils.h"
#include <cstdlib>
#include <omp.h>
using namespace std;



int main(int argc, char** argv) {
   
   //int N    = read_option<long>("-N", argc, argv);
   //int T    = read_option<long>("-T", argc, argv);

   int N = 300;   // grid size
   int T = 1;     // number of threads

   int itm        = 300000;              // max number of iterations
   double h = 1.0 / double(N+1);
   double* u     = (double*) malloc((N+2) * (N+2) * sizeof(double));   // u, (N+2) x (N+2) matrix ( includes 2 ghost points per axis)
   double* u_nxt = (double*) malloc((N+2) * (N+2) * sizeof(double));   // next u in iteration
   double* f     = (double*) malloc((N+2) * (N+2) * sizeof(double));   // right hand side

   // Initialize u, u_nxt, f
   for (int i = 0; i < N+2; i++) {
      for (int j = 0; j < N+2; j++) {
         u[i+j*(N+2)] = 0.0;
         f[i+j*(N+2)] = 1.0;
      }
   }  // done with initialization of u, u_nxt, f
   

   double fac     = pow(10.0,6.0);
   double resInit = 1.0;

   
   Timer t;
   t.tic();
   
   // Jacobi method
   for (int n=0; n < itm; n++) {
      
      // given u, do one iteration of Jacobi method to get u_nxt
      double res_ij  = 0.0;           //   | (Au_n - f)_ij |
      double sup_res_init = 1.0;      // ||Au_0 - f||_sup = 1.0 (with initial guess u_0 = 0)
      double sup_res      = 0.0;      // ||Au_nxt - f||_sup
     
      # pragma omp parallel for collapse(2) num_threads(T)      
      for (int i = 1; i < N+1; i++) {
         for (int j = 1; j < N+1; j++) {
            u_nxt[i+j*(N+2)] = 0.25*( h*h*f[i+j*(N+2)] + u[i-1+j*(N+2)]
                                  + u[i+(j-1)*(N+2)] + u[i+1+j*(N+2)] + u[i+(j+1)*(N+2)] );
         }
      } // done with Jacobi n-th iteration
      
      for (int i = 1; i < N+1; i++) {
         for (int j = 1; j < N+1; j++) {
            res_ij = abs(f[i+j*(N+2)] + (N+1)*(N+1)*( u[i-1+j*(N+2)] + u[i+(j-1)*(N+2)]
                                          -4.0* u[i+j*(N+2)] + u[i+1+j*(N+2)] + u[i+(j+1)*(N+2)] )) ;
            if (res_ij > sup_res){
               sup_res = res_ij;
            }
         }
      } 
      //cout << " Current Residual = " << sup_res << endl;
      if ( sup_res < (sup_res_init / fac) ){  // stopping criterion
         cout << "    "                                        <<                 endl;
         cout << " Jacobi has converged, iterations needed : " << n-1          << endl;
         cout << " Grid Size, N  = "                           << N            << endl;
         cout << " Initial Residual "                          << sup_res_init << endl;
         cout << " Final   Residual "                          << sup_res      << endl;
         cout << "    "                                        <<                 endl;
         break;
      }
      if ( n == itm-1 ) {
         cout << "    "                                                <<                 endl;
         cout << " Jacobi has NOT converged, Max iteration reached = " << n            << endl;         
         cout << " Grid Size, N  = "                                   << N            << endl;
         cout << " Initial Residual "                                  << sup_res_init << endl;
         cout << " Final   Residual "                                  << sup_res      << endl;
         cout << "    "                                                <<                 endl;
         break;

      }
      // at the end of iteration n, set u = u_nxt:
      for (int i = 1; i < N+1; i++) {
         for (int j = 1; j < N+1; j++) {
            u[i+j*(N+2)] = u_nxt[i+j*(N+2)];
         }
      }
   }

   double time = t.toc();
   cout << " time              = " << time << endl;
   cout << " number of threads = " << T << endl;
   cout << "    "                                        <<            endl;
   
   free(u);
   free(u_nxt);
   free(f);
   
   return 0;
}

