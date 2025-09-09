import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm

plt.rcParams['text.usetex'] = True

Nstate = 2
N  = 40
FS = 15

std = 0.08

angle = 8*np.pi/180
A = 0.99*np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])

NMC = 50000

Time = [k for k in range(N)]
End = 15

spastAll = []
spredAll = []

Box = np.zeros([2,2])

spast = np.array([1,1]).reshape(2,1)
for now in range(0,End):
    spred = [spast[:,-1].reshape(2,1)]*NMC

    for time in Time:
        for n in range(0,NMC):
            sps = np.matmul(A,spred[n][:,-1]).reshape(2,1) + np.random.uniform(-0.1,0.1,2).reshape(2,1)#np.random.normal(0,std,2).reshape(2,1)#
            spred[n] = np.concatenate( (spred[n],sps.reshape(2,1)), axis=1)
           


    snew = np.matmul(A,spast[:,-1].reshape(2,1)) + np.random.uniform(-0.1,0.1,2).reshape(2,1)#np.random.normal(0,std,2).reshape(2,1)#
    
    spastAll.append(spast)
    spredAll.append(spred)
    
    
    
    spast = np.concatenate( (spast,snew.reshape(2,1)), axis=1)
        
for now in range(0,End):
    for n in range(0,NMC):
        Box[0,0] = np.min([ Box[0,0], np.min(spredAll[now][n][0])  ])
        Box[0,1] = np.max([ Box[0,1], np.max(spredAll[now][n][0])  ])
        Box[1,0] = np.min([ Box[1,0], np.min(spredAll[now][n][1])  ])
        Box[1,1] = np.max([ Box[1,1], np.max(spredAll[now][n][1])  ])

for now in range(0,End):
    
    plt.close('all')

    figID = plt.figure(1,figsize=(12,6))
    
    ax1 = figID.add_subplot(111)
    ax1.plot(spastAll[now][0,:],spastAll[now][1,:],color=[0,0,0.5],linewidth=2,marker='o')

    #for n in range(0,NMC):
    #    ax1.plot(spredAll[now][n][0,:],spredAll[now][n][1,:],color='b',linewidth=1)

    x = []
    y = []
    for n in range(0,NMC):
        x += list(spredAll[now][n][0,:])
        y += list(spredAll[now][n][1,:])
    x = np.array(x)
    y = np.array(y)
    
    H, xedges, yedges = np.histogram2d(x, y,bins = 20) #, bins=(xedges, yedges)
    H = H.T
    H = H/np.max(H)
    #ax1 = fig.add_subplot(133, title='NonUniformImage: interpolated',
            #aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
    im = NonUniformImage(ax1, interpolation='bilinear',cmap=cm.Blues)
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    im.set_data(xcenters, ycenters, H)
    ax1.add_image(im)

    #ax1.pcolormesh(xcenters, ycenters, H, vmin=0., vmax=0.5, cmap=cm.Blues,interpolation='bilinear')

    LW = 3
    CO = 'r'
    ax1.plot([Box[0,0],Box[0,1]],[Box[1,0],Box[1,0]],color=CO,linewidth=LW)
    ax1.plot([Box[0,1],Box[0,1]],[Box[1,0],Box[1,1]],color=CO,linewidth=LW)
    ax1.plot([Box[0,1],Box[0,0]],[Box[1,1],Box[1,1]],color=CO,linewidth=LW)
    ax1.plot([Box[0,0],Box[0,0]],[Box[1,1],Box[1,0]],color=CO,linewidth=LW)
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('equal', 'box')
    ax1.set_axis_off()
    plt.show(block=False)

    #sys.exit()


    
    #figID.savefig('Markov'+str(now)+'.eps', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'
    
    plt.pause(.1)




