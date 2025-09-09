import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm


plt.rcParams['text.usetex'] = True

Nstate = 2
N  = 40
FS = 15

std = 0.05

angle = 8*np.pi/180
A = 0.99*np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])

NMC     = 2000
NMCplot = 10

Time = [k for k in range(N)]
End = 5

def MakeBox(spredAll, CO, Time = [], LW = 3):
    
    Box = np.array([[np.inf,-np.inf],[np.inf,-np.inf]])
    if Time:
        now  = Time[0]
        time = Time[1]
        for n in range(0,NMC):
            Box[0,0] = np.min([ Box[0,0], np.min(spredAll[now][n][0][time])  ])
            Box[0,1] = np.max([ Box[0,1], np.max(spredAll[now][n][0][time])  ])
            Box[1,0] = np.min([ Box[1,0], np.min(spredAll[now][n][1][time])  ])
            Box[1,1] = np.max([ Box[1,1], np.max(spredAll[now][n][1][time])  ])
    else:
        for now in range(0,End):
            for n in range(0,NMC):
                Box[0,0] = np.min([ Box[0,0], np.min(spredAll[now][n][0])  ])
                Box[0,1] = np.max([ Box[0,1], np.max(spredAll[now][n][0])  ])
                Box[1,0] = np.min([ Box[1,0], np.min(spredAll[now][n][1])  ])
                Box[1,1] = np.max([ Box[1,1], np.max(spredAll[now][n][1])  ])
        
    #if Time:
    #    ax1.fill([Box[0,0], Box[0,1], Box[0,1], Box[0,0]],[Box[1,0],Box[1,0],Box[1,1], Box[1,1]  ],color=CO,alpha=0.1)

    ax1.plot([Box[0,0],Box[0,1]],[Box[1,0],Box[1,0]],color=CO,linewidth=LW)
    ax1.plot([Box[0,1],Box[0,1]],[Box[1,0],Box[1,1]],color=CO,linewidth=LW)
    ax1.plot([Box[0,1],Box[0,0]],[Box[1,1],Box[1,1]],color=CO,linewidth=LW)
    ax1.plot([Box[0,0],Box[0,0]],[Box[1,1],Box[1,0]],color=CO,linewidth=LW)

    
    
    
    return Box



spastAll = []
spredAll = []



spast = np.array([1,1]).reshape(2,1)
for now in range(0,End):
    spred = [spast[:,-1].reshape(2,1)]*NMC

    for time in Time:
        for n in range(0,NMC):
            sps = np.matmul(A,spred[n][:,-1]).reshape(2,1) + np.random.uniform(-std,std,2).reshape(2,1)#np.random.normal(0,std,2).reshape(2,1)#
            spred[n] = np.concatenate( (spred[n],sps.reshape(2,1)), axis=1)
           


    snew = np.matmul(A,spast[:,-1].reshape(2,1)) + np.random.uniform(-std,std,2).reshape(2,1)#np.random.normal(0,std,2).reshape(2,1)#
    
    spastAll.append(spast)
    spredAll.append(spred)
    
    
    
    spast = np.concatenate( (spast,snew.reshape(2,1)), axis=1)
        

for now in range(0,End):

    for boxit in [False,True]:
        plt.close('all')

        figID = plt.figure(1,figsize=(12,6))
        
        ax1 = figID.add_subplot(111)
        
        if boxit:
            for time in range(now,N+1):
                Box = MakeBox(spredAll,'c',Time = [now,time],LW = 1)

        ax1.plot(spastAll[now][0,:],spastAll[now][1,:],color=[0,0,0.5],linewidth=2,marker='o')

        for n in range(0,NMCplot):
            ax1.plot(spredAll[now][n][0,:],spredAll[now][n][1,:],color='b',linewidth=1,marker='.')

        MakeBox(spredAll,'r')

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_axis_off()
        ax1.set_aspect('equal', 'box')
     
        if boxit:
            figID.savefig('PessimisticModelBOXED'+str(now)+'.eps', transparent=True, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'
        else:
            figID.savefig('PessimisticModel'+str(now)+'.eps', transparent=True, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'





        plt.show(block=False)




    
    





