#http://scikit-cuda.readthedocs.io/en/latest/generated/skcuda.linalg.mdot.html#skcuda.linalg.mdot


import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np
import skcuda.linalg as linalg
linalg.init()
a = np.asarray(np.random.rand(4, 2), np.float32)
b = np.asarray(np.random.rand(2, 2), np.float32)
c = np.asarray(np.random.rand(2, 2), np.float32)
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
d_gpu = linalg.mdot(a_gpu, b_gpu)
print(np.allclose(np.dot(a, b), d_gpu.get()))

