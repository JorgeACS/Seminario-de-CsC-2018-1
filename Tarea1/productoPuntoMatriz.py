import pycuda.autoinit
import pycuda.driver as drv
import numpy
import math
from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void vectorMultiplication(float *c, float *a, float *b)
{
 	const int i = threadIdx.x;
 	c[i]+= a[i] * b[i];
}

__global__ void sumReduce( float *d,float *c,float *p)
{
	
	const int i = threadIdx.x;
	
	const int power = p[0];
	const int length = blockDim.x*power;
	if(i*power >= length)return;
	
	d[i*power] = c[i*power];
	if(i*power+(power/2) < length){
		d[i*power]+= c[i*power+(power/2)];
	}
}


""")

vectorMultiplication = mod.get_function("vectorMultiplication")
sumReduce = mod.get_function("sumReduce")

length = 500
a = numpy.random.randn(length).astype(numpy.float32)
b = numpy.random.randn(length).astype(numpy.float32)

c = numpy.zeros_like(a)

#multiplicamos a*b para obtener c. Este es el primer paso para obtener el producto punto.
vectorMultiplication(
        drv.Out(c), drv.In(a), drv.In(b),
        block=(length,1,1), grid=(1,1))
d = numpy.copy(c)
power = numpy.random.randn(1).astype(numpy.float32)
power[0] = 2
done = False
#Hacemos reduce. Consiste en ir sumando los elementos del arreglo, 
#de tal manera que al final los queda la suma en el indice 0 de 'c'
while True:
	blockSize = math.ceil(length/int(power[0]))
	if blockSize == 1:
		done = True
	sumReduce(drv.Out(d),drv.In(c),drv.In(power),block=(blockSize,1,1),grid=(1,1))
	c = numpy.copy(d)
	power[0]*=2
	if done:
		break
print(numpy.dot(a,b))
print(c[0])