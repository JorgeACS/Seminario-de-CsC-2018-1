import pycuda.autoinit
import pycuda.driver as drv
import numpy
import math
from pycuda.compiler import SourceModule
mod = SourceModule("""

#include "curand.h"
#include "curand_kernel.h"
#include <limits.h>

extern "C"{
   __global__ void montecarlo (int *dataArray) {
   
  //int i = threadIdx.x;
  curandState_t curandState;
  int i = blockIdx.x*blockDim.x;
  int j = threadIdx.x;
  curand_init(blockIdx.x,threadIdx.x,0,&curandState);
  double x = curand_uniform_double(&curandState);
  double y = curand_uniform_double(&curandState);
  if(x*x + y*y <= 1) {
    dataArray[i+j]=1;
  }else{
    dataArray[i+j]=0;
  }
  return;


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

}
""", no_extern_c=True)

montecarlo = mod.get_function("montecarlo")
sumReduce = mod.get_function("sumReduce")
length = 1000
samples = numpy.zeros((length,),dtype=int)

#multiplicamos a*b para obtener c. Este es el primer paso para obtener el producto punto.
montecarlo(
        drv.Out(samples),block=(length,1,1), grid=(1,1))
d = numpy.copy(samples)
#Hacemos reduce. Consiste en ir sumando los elementos del arreglo, 
#de tal manera que al final los queda la suma en el indice 0 de 'c'
power = numpy.random.randn(1).astype(numpy.float32)
power[0] = 2
done = False
while True:
	blockSize = math.ceil(length/int(power[0]))
	if blockSize == 1:
		done = True
	sumReduce(drv.Out(d),drv.In(samples),drv.In(power),block=(blockSize,1,1),grid=(1,1))
	samples = numpy.copy(d)
	power[0]*=2
	if done:
		break
print((samples[0]/length)*4)
print(math.pi)