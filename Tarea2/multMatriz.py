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
__global__ void matrixMult (float *a, float *b, float *_n, float *_m, float *_p, float *dest) {

  int n = _n[0];
  int m = _m[0];
  int p = _p[0];
  int aIndex= blockIdx.x*m;
  int i = blockIdx.x*blockDim.x;
  int j = threadIdx.x;
  dest[i+j] = 0;
  for(int k = 0; k < m; k++){
    dest[i+j]+= a[aIndex+k]*b[k*p+j];
  }
}
""")

matrixMult = mod.get_function("matrixMult")

x1 = int(sys.argv[1])
x2 = int(sys.argv[2])
y1 = int(sys.argv[3])
y2 = int(sys.argv[4])

if x2 == y1:
	a = numpy.random.randint(10,size=(x1,x2)).astype(numpy.float32)
	b = numpy.random.randint(10,size=(y1,y2)).astype(numpy.float32)

	dest = numpy.zeros(shape=(x1,y2)).astype(numpy.float32)


	m = numpy.random.randn(1).astype(numpy.float32)
	m[0] = y1
	n = numpy.random.randn(1).astype(numpy.float32)
	n[0] = x1
	p = numpy.random.randn(1).astype(numpy.float32)
	p[0] = y2
	matrixMult(
	        drv.In(a), drv.In(b),drv.In(n),drv.In(m),drv.In(p),drv.Out(dest),
	        block=(y2,1,1), grid=(x1,1,1))
	drv.Context.synchronize()

	print(a)
	print(b)
	print(numpy.matmul(a,b))
	print(dest)
	print(numpy.equal(dest,numpy.matmul(a,b)))
else:
	print("Input no valido")
	print ("Para matrices (x1,x2) , (y1,y2) , 'x2' y 'y1' tienen que ser iguales")

