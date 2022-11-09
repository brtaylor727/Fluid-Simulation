
import numpy as np
import matplotlib.pyplot as plt

import sys


a = np.loadtxt(sys.argv[1])[:,2]
x = np.loadtxt(sys.argv[1])[:,0]
y = np.loadtxt(sys.argv[1])[:,1]

a = a.reshape(int(np.max(x))+1,int(np.max(y))+1)

plt.imshow(a.T)
plt.title(sys.argv[1])
plt.colorbar()
plt.savefig('movie/'+sys.argv[1].split('/')[-1].split('.')[0]+'.png')
#plt.show()


