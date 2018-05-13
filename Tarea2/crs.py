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
    }
    printf("]");
	}

	__global__ void ThreadedGpuMatrixCRSMultiplication(float *vals,float *column_indexes,
	                                                   float *row_pointers,float *B, float *C, int Bn){
	    int row = blockIdx.x* blockDim.x;
	    int column = threadIdx.x;
	    int valIndex = row_pointers[blockIdx.x];
	    int valueCount = row_pointers[blockIdx.x+1]-row_pointers[blockIdx.x];
	    
	    C[row+column] = 0;
	    for(int k = 0; k < valueCount;k++){
	    	int columnValue = column_indexes[valIndex+k];
	        C[row+column]= C[row+column] + vals[valIndex+k]*B[columnValue*Bn+column];
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

valsA = numpy.random.randint(10,size=N*N).astype(numpy.float32)
column_indexesA = numpy.random.randint(10,size=N*N).astype(numpy.float32)

#Pass values to GPU
row_pointersA = numpy.random.randint(10,size=N+1).astype(numpy.float32)
zeroCountA = numpy.uint32(0)
n_gpu = numpy.uint32(N)
a_gpu = drv.mem_alloc(a.nbytes)
valsA_gpu = drv.mem_alloc(valsA.nbytes)
column_indexesA_gpu = drv.mem_alloc(column_indexesA.nbytes)
row_pointersA_gpu = drv.mem_alloc(row_pointersA.nbytes)

initializeCRS(
        a_gpu,valsA_gpu,column_indexesA_gpu,row_pointersA_gpu,n_gpu,n_gpu,drv.In(zeroCountA),
        block=(1,1,1), grid=(1,1,1))
#Return A variables to GPU
drv.Context.synchronize()
drv.memcpy_dtoh(a,a_gpu)
drv.memcpy_dtoh(valsA, valsA_gpu)
drv.memcpy_dtoh(column_indexesA, column_indexesA_gpu)
drv.memcpy_dtoh(row_pointersA,row_pointersA_gpu)
#Send to GPU again
drv.Context.synchronize()
#valsA_gpu = drv.mem_alloc(valsA.nbytes)
#column_indexesA_gpu = drv.mem_alloc(column_indexesA.nbytes)
#row_pointersA_gpu = drv.mem_alloc(row_pointersA.nbytes)

c = numpy.zeros_like(a).astype(numpy.float32)
drv.Context.synchronize()
print(a)
print(valsA)
print(column_indexesA)
print(row_pointersA)
ThreadedGpuMatrixCRSMultiplication(drv.In(valsA),drv.In(column_indexesA),drv.In(row_pointersA),
	drv.In(b),drv.Out(c),n_gpu,
        block=(N,1,1), grid=(N,1,1))
drv.Context.synchronize()
print(zeroCountA)
print(c)
print(numpy.equal(c,numpy.matmul(a,b)))
#print(a)
#print(b)
#print(numpy.matmul(a,b))
#print(dest)
#print(numpy.equal(dest,numpy.matmul(a,b)))