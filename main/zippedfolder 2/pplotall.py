
"""Script to plot the data given a string to match"""

import glob


import numpy as np
import matplotlib.pyplot as plt

import sys


files = glob.glob(sys.argv[1]+"*.dat") 
print(files)

myinfo = []

for myfile in files:
    infodict = {}



    print(myfile)

    a = np.loadtxt(myfile)[:,2]
    x = np.loadtxt(myfile)[:,0]
    y = np.loadtxt(myfile)[:,1]

    a = a.reshape(int(np.max(x))+1,int(np.max(y))+1)


    infodict['myfile'] = myfile
    infodict['a'] = np.loadtxt(myfile)[:,2]
    infodict['x'] = np.loadtxt(myfile)[:,0]
    infodict['y'] = np.loadtxt(myfile)[:,1]
    infodict['a'] = a

    myinfo.append(infodict)


cmax = np.max([np.max(a['a']) for a in myinfo])
cmin = np.min([np.min(a['a']) for a in myinfo])

for a in range(len(myinfo)):
    myinfo[a].update({'cmax': cmax})
    myinfo[a].update({'cmin': cmin})



for infodict in myinfo:
    a = infodict['a']
    myfile = infodict['myfile']

    print(infodict['cmax'])

    fig = plt.figure()
    plt.imshow(a.T,vmax=infodict['cmax'],vmin=infodict['cmin'])
    plt.title(myfile)
    plt.colorbar()
    savefile = 'movie/'+myfile.split('/')[-1].split('.')[0]+'.png'
    print("saving in file "+savefile)
    plt.savefig(savefile)
    #plt.show()
    plt.close(fig)


