all: dot matvec jacobi2D

dot: dot.cu
	nvcc -arch=sm_61 dot.cu -o dot -Xcompiler -fopenmp

matvec: matvec.cu
	nvcc -arch=sm_61 matvec.cu -o matvec -Xcompiler -fopenmp

jacobi2D: jacobi2D.cu
	nvcc -arch=sm_61 jacobi2D.cu -o jacobi2D -Xcompiler -fopenmp
