import PIL
from PIL import Image
import time

import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
mod = SourceModule("""
	#include <stdio.h>
__global__ void upscale (float *input, float *output) {
  int i = blockIdx.x*blockDim.x;
  int j = threadIdx.x;
  int index = i+j;
  int iOutput = (blockDim.x*2)*blockIdx.x;
  int outputIndex = iOutput + threadIdx.x;
  output[outputIndex*6] = input[index*3];
  output[outputIndex*6+1] = input[index*3+1];
  output[outputIndex*6+2] = input[index*3+2];

  output[outputIndex*6+3] = input[index*3];
  output[outputIndex*6+4] = input[index*3+1];
  output[outputIndex*6+5] = input[index*3+2];

  output[(outputIndex + (blockDim.x))*6] = input[index*3];
  output[(outputIndex + (blockDim.x))*6+1] = input[index*3+1];
  output[(outputIndex + (blockDim.x))*6+2] = input[index*3+2];

  output[(outputIndex + (blockDim.x))*6+3] = input[index*3];
  output[(outputIndex + (blockDim.x))*6+4] = input[index*3+1];
  output[(outputIndex + (blockDim.x))*6+5] = input[index*3+2];
}
""")

intensity = numpy.uint32(1)
upscale = mod.get_function("upscale")
inputImage = Image.open("./input.jpg")
print(inputImage.mode)
inputAsArray = numpy.array(inputImage)
print(inputAsArray.shape)
inputAsArray = inputAsArray.astype(numpy.float32)
size = inputImage.size[0]*inputImage.size[1]*4*3
outputAsArray = numpy.zeros((inputImage.size[0]*2,inputImage.size[1]*2,3)).astype(numpy.float32)
print(outputAsArray.shape)
# Black and White
print(f'{inputImage.size[0]}x{inputImage.size[1]}')
upscale(
        drv.In(inputAsArray),drv.Out(outputAsArray),
        block=(inputImage.size[0],1,1), grid=(inputImage.size[1],1,1))
drv.Context.synchronize()
outputAsArray = (numpy.uint8(outputAsArray))
outputImage = Image.fromarray(outputAsArray,mode ="RGB")
outputImage.save("./output-upscaled.jpg")