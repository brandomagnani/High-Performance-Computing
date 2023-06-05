// $ g++ -O0 -std=c++11 Laplace1D.cpp && ./a.out
// $ g++ -O3 -std=c++11 Laplace1D.cpp && ./a.out

#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cstring>
#include <stdlib.h>
#include <iostream>
#include <fstream>                // write the output to a file
#include "utils.h"
using namespace std;



int main(int argc, char** argv) {
   
   int N    = 2000;
   int itm        = 50000000;              // max number of iterations
   double h = 1.0 / double(N+1);
   double* A     = (double*) malloc((N+2) * (N+2) * sizeof(double));  // (N+2) x (N+2) matrix
   double* u     = (double*) malloc((N+2) * sizeof(double));          // (N+2) x 1 vector (N + 2 ghost points)
   double* u_new = (double*) malloc((N+2) * sizeof(double));          // (N+2) x 1 vector (N + 2 ghost points)
   double* f     = (double*) malloc((N+2) * sizeof(double));          // (N+2) x 1 vector
   double* Au    = (double*) malloc((N+2) * sizeof(double));          // (N+2) x 1 vector
   // Initialize A
   for (int i = 0; i < N+2; i++) {
      for (int j = 0; j < N+2; j++) {
         if (i==j){
            A[i+j*(N+2)] = 2.0 / (h*h);
         }
         else if ( (j == (i+1)) or (j == (i-1)) ) {
            A[i+j*(N+2)] = -1.0 / (h*h);
         }
         else{
            A[i+j*(N+2)] = 0.0;
         }
      }
   }  // done with initialization of A
   

   
   // Initialize u (to the initial guess), u_new and f
   for (int i = 0; i < N+2; i++) {
      u[i]      = 0.0;
      u_new[i]  = 0.0;
      f[i]      = 1.0;
      Au[i]     = 0.0;
   }
   
   double fac     = pow(10.0,6.0);
   double resInit = 1.0;

   
   Timer t;
   t.tic();
/*
   // Jacobi method
   for (int n=0; n < itm; n++) {
      
      // given u, do one iteration of Jacobi method to get u_new
      for (int i = 1; i < N+1; i++) {
         double u_im  = u[i-1];
         double u_ip  = u[i+1];
         double f_i   = f[i];
         u_new[i] = 0.5* ( ( f_i*h*h ) + u_im + u_ip);
         
      }   // done with single iteration of Jacobi method
*/
   // Gauss-Seidel method
   for (int n=0; n < itm; n++) {
      for (int i = 1; i < N+1; i++) {
         double u_kk  = u_new[i-1];
         double u_kii  = u[i+1];
         double f_i   = f[i];
         u_new[i] = 0.5* ( ( f_i*h*h ) + u_kk + u_kii);
      }
         

      
      
      // Compute the residual: sup norm Au-f
      double res = 0.0;
      double res_i;
      
      for (int i=1; i < N+1; i++) {
         double u_im  = u_new[i-1];
         double u_i   = u_new[i];
         double u_ip  = u_new[i+1];
         double f_i   = f[i];
         
         res_i = abs( ((double(N+1)*double(N+1))*(-u_im + (2.0*u_i) - u_ip)) - f_i );
         
         if (res_i > res) {
            res = res_i;
         }
      }   // done with computing residual for u_new
      
      // TO PLOT RESIDUAL AFTER EACH ITERATION, UNCOMMENT THE LINE BELOW
      /* cout << " Current residual = " << res << endl; */
      
      if ( res < (resInit / fac) ) {
         cout << "    "                                              <<            endl;
         cout << " Gauss-Seidel has converged, iterations needed : " << n-1     << endl;
         cout << " Grid Size, N  = "                                 << N       << endl;
         cout << " Iterations needed =  "                            << n-1     << endl;
         cout << " Initial Residual "                                << resInit << endl;
         cout << " Final   Residual "                                << res     << endl;
         cout << "    "                                              <<            endl;
         break;
      }
      if ( n == itm-1 ) {
         cout << " Max iteration reached, Jacobi has NOT converged " << endl;
      }
      
      // set u <-- u_new
      for (int i=1; i < N+1; i++) {
         u[i] = u_new[i];
      }
      
   }
   double time = t.toc();
   cout << " time = " << time << endl;
   cout << "    "                                        <<            endl;
   
   free(A);
   free(u);
   free(u_new);
   free(f);
   free(Au);
   
   return 0;
}
