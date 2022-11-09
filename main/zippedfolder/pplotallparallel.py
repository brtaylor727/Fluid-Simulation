
"""Script to plot the data given a string to match"""

"""argument 1 should be string to match the data"""
"""argument 2 should be the number of threads to use"""

import glob


import numpy as np
import matplotlib.pyplot as plt

import sys

from multiprocessing import Pool
import matplotlib


import time


def setupfiles(myfile):

    infodict = {}



    #print(myfile)

    a = np.loadtxt(myfile)[:,2]
    x = np.loadtxt(myfile)[:,0]
    y = np.loadtxt(myfile)[:,1]

    a = a.reshape(int(np.max(x))+1,int(np.max(y))+1)


    infodict['myfile'] = myfile
    infodict['a'] = np.loadtxt(myfile)[:,2]
    infodict['x'] = np.loadtxt(myfile)[:,0]
    infodict['y'] = np.loadtxt(myfile)[:,1]
    infodict['a'] = a

    return infodict

#for infodict in myinfo:

def makeplot(infodict):
    a = infodict['a']
    myfile = infodict['myfile']

    #print(infodict['cmax'])

    fig = plt.figure()

    plt.title(myfile)
    cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.1, 0.5, 0.5),
                 (0.2, 0.0, 0.0),
                 (0.4, 0.2, 0.2),
                 (0.6, 0.0, 0.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),
        'green':((0.0, 0.0, 0.0),
                 (0.1, 0.0, 0.0),
                 (0.2, 0.0, 0.0),
                 (0.4, 1.0, 1.0),
                 (0.6, 1.0, 1.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.0, 0.0),
                 (0.1, 0.5, 0.5),
                 (0.2, 1.0, 1.0),
                 (0.4, 1.0, 1.0),
                 (0.6, 0.0, 0.0),
                 (0.8, 0.0, 0.0),
                 (1.0, 0.0, 0.0))}

    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    #plt.pcolor(np.random.rand(10,10),cmap=my_cmap)

    plt.imshow(a,vmax=infodict['cmax'],vmin=infodict['cmin'],cmap=my_cmap)
    plt.colorbar()
    savefile = 'movie/'+myfile.split('/')[-1].split('.')[0]+'.png'
    #print("saving in file "+savefile)
    plt.savefig(savefile)
    #plt.show()
    plt.close(fig)




if __name__ == '__main__':



    start = time.time()
    p = Pool(int(sys.argv[2]))

    #print("Using ",sys.argv[2]," Threads")
    print(sys.argv[2])

    files = glob.glob(sys.argv[1]+"*.dat") 

    #print(files)


    myinfo =  p.map(setupfiles,files)

    cmax = np.max([np.max(a['a']) for a in myinfo])
    cmin = np.min([np.min(a['a']) for a in myinfo])

    for a in range(len(myinfo)):
        myinfo[a].update({'cmax': cmax})
        myinfo[a].update({'cmin': cmin})



    #start = time.time()

    p.map(makeplot,myinfo)


    end = time.time()

    #print("Time Elapsed: ",end - start)
    print(end - start)




















