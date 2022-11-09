import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

a = np.random.randn(4,4).astype(np.float32)
b = np.random.randn(4,4).astype(np.float32)

a_gpu = gpuarray.to_gpu(a)
a_doubled = (2*a_gpu).get()

print(a_doubled)
print(a_gpu)


b_gpu = gpuarray.to_gpu(b)

b_matmul = (a_gpu*b_gpu).get()

print(b_matmul)
print(np.matmul(a,b))
