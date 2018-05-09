#!/usr/bin/python

#Multiplicacion de matrices generica

#acepta cualquier multiplicacion de matrices generica

#para usar, mandar parametros al mandar a llamar a python
#(poner en archivo batch si se usa sbatch)
# Ex: python3 multMatriz.py 20 15 15 20
import pycuda.autoinit
import pycuda.driver as drv
import numpy
import sys
from pycuda.compiler import SourceModule
mod = SourceModule("""
	#include <stdio.h>
	#include <math.h>
	#include <stdlib.h>
	#include <stdint.h>
	#include <time.h>
	#include "curand.h"
	#include "curand_kernel.h"

extern "C"{
	__global__ void initializeCRS (float *matriz,
	                               float *vals,
	                               float *column_indexes,
	                               float *row_pointers,
	                               int n,
	                               int m,
	                               int *zeroCount) {


	    curandState_t curandState;
	    curand_init(100,0,0,&curandState);
	    *zeroCount = 0;
	    for(int  i = 0; i < n; i++){
	        for(int j = 0; j < m;j++){
	            matriz[i*n+j] = curand(&curandState)&10;
	            if(matriz[i*n+j] == 0){
	                (*zeroCount)=(*zeroCount)+1;
	            }
	        }
	    }
	    printf("despues de inicializar matriz con random");
	    if((*zeroCount) == 0){
	        (*zeroCount) = (*zeroCount)-2;
	        return;
	    }

	    int zeroIndex = 0;
	    //inicializacion de arreglos CRS
	    for(int i = 0; i < n;i++){
	        int firstNonZeroFound = true;
	        int rowAllZeros = true;
	        for(int j = 0; j < m;j++){
	            if(matriz[i*n+j] != 0){
	                vals[zeroIndex] = matriz[i*n+j];
	                column_indexes[zeroIndex] = j;
	                if(firstNonZeroFound){
	                    row_pointers[i] = zeroIndex;
	                    rowAllZeros = false;
	                    firstNonZeroFound = false;
	                }
	                zeroIndex++;
	            }
	        }
	        if(rowAllZeros){
	            row_pointers[i] = -1;
	        }
	    }
	    row_pointers[n] = n*m-(*zeroCount);
	    printf("Finished calculating things");
	}
	__global__ void GpuPrintCRS(
        int *matriz,
        int *vals,
        int *column_indexes,
        int *row_pointers,
        int n,
        int m,
        int zeroCount){
    printf("on print crs");
    for(int i = 0; i < n;i++){
        for(int j = 0; j < m;j++){
            printf("%d,",matriz[i*n+j]);
        }
        printf("");
    }
    printf("%d ceros,  %d no-ceros",zeroCount,n*m-zeroCount);
    printf("Vals: [");
    for(int i = 0; i < n*m-zeroCount;i++){
        printf("%d,",vals[i]);
    }
    printf("]");
    printf("col_ind: [");
    for(int i = 0; i < n*m-zeroCount;i++){
        printf("%d ",column_indexes[i]);
    }
    printf("]");
    printf("row_ptr: [");
    for(int i = 0; i < n+1;i++){
        printf("%d ",row_pointers[i]);
    }
    printf("]");
	}

	__global__ void ThreadedGpuMatrixCRSMultiplication(int *vals,int *column_indexes,
	                                                   int *row_pointers,int *B, int *C, int Bn){
	    int row = blockIdx.x* blockDim.x;
	    int column = threadIdx.x;
	    int valIndex = row_pointers[blockIdx.x];
	    int valueCount = row_pointers[blockIdx.x+1]-row_pointers[blockIdx.x];
	    C[row+column] = 0;
	    for(int k = 0; k < valueCount;k++){
	        C[row+column]= C[row+column] + vals[valIndex+k]*B[column_indexes[valIndex+k]*Bn+column];
	    }
	}
	}
""",no_extern_c=True)

ThreadedGpuMatrixCRSMultiplication = mod.get_function("ThreadedGpuMatrixCRSMultiplication")
GpuPrintCRS = mod.get_function("GpuPrintCRS")
initializeCRS= mod.get_function("initializeCRS")
N = int(sys.argv[1])
a = numpy.random.randint(10,size=(N,N)).astype(numpy.float32)
b = numpy.random.randint(10,size=(N,N)).astype(numpy.float32)

dest = numpy.zeros(shape=(N,N)).astype(numpy.float32)

valsA = numpy.random.randint(10,size=(N+1,1)).astype(numpy.float32)
column_indexesA = numpy.random.randint(10,size=(N*N,1)).astype(numpy.float32)

row_pointersA = numpy.random.randint(10,size=(N+1,1)).astype(numpy.float32)
zeroCountA = numpy.uint32(0)
n_gpu = numpy.uint32(N)
zeroCountA_gpu = drv.mem_alloc(zeroCountA.nbytes)
a_gpu = drv.mem_alloc(a.nbytes)
b_gpu = drv.mem_alloc(b.nbytes)
valsA_gpu = drv.mem_alloc(valsA.nbytes)
column_indexesA_gpu = drv.mem_alloc(column_indexesA.nbytes)
row_pointersA_gpu = drv.mem_alloc(row_pointersA.nbytes)

initializeCRS(
        a_gpu,valsA_gpu,column_indexesA_gpu,row_pointersA_gpu,n_gpu,n_gpu,zeroCountA_gpu,
        block=(1,1,1), grid=(1,1,1))


c = numpy.zeros_like(a).astype(numpy.float32)
drv.Context.synchronize()
GpuPrintCRS(
        a_gpu,valsA_gpu,column_indexesA_gpu,row_pointersA_gpu,n_gpu,n_gpu,zeroCountA,
        block=(1,1,1), grid=(1,1,1))
ThreadedGpuMatrixCRSMultiplication(valsA_gpu,column_indexesA_gpu,row_pointersA_gpu,b_gpu,drv.Out(c),
        block=(1,1,1), grid=(1,1,1))
print(zeroCountA)
print(c)
#print(a)
#print(b)
#print(numpy.matmul(a,b))
#print(dest)
#print(numpy.equal(dest,numpy.matmul(a,b)))