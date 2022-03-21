all: val_test01_solved val_test02_solved MMult1 omp_solved2 omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 jacobi2D-omp gs2D-omp

val_test01_solved: val_test01_solved.cpp
	g++ -g val_test01_solved.cpp -o val_test01_solved #&& valgrind --leak-check=full 
        
val_test02_solved: val_test02_solved.cpp
	g++ -g val_test02_solved.cpp -o val_test02_solved #&& valgrind --leak-check=full 

MMult1: MMult1.cpp
	g++ -std=c++11 -fopenmp -O3 -march=native MMult1.cpp -o MMult1

omp_solved2: omp_solved2.c
	g++ -fopenmp -O3 omp_solved2.c -o omp_solved2

omp_solved3: omp_solved3.c
	g++ -fopenmp -O3 omp_solved3.c -o omp_solved3

omp_solved4: omp_solved4.c
	g++ -fopenmp -O3 omp_solved4.c -o omp_solved4

omp_solved5: omp_solved5.c
	g++ -fopenmp -O3 omp_solved5.c -o omp_solved5

omp_solved6: omp_solved6.c
	g++ -std=c++11 -fopenmp -O3 omp_solved6.c -o omp_solved6

jacobi2D-omp: jacobi2D-omp.cpp
	g++ -std=c++11 -fopenmp -O3 jacobi2D-omp.cpp -o jacobi2D-omp

gs2D-omp: gs2D-omp.cpp
	g++ -std=c++11 -fopenmp -O3 gs2D-omp.cpp -o gs2D-omp
