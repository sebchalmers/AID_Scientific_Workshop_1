import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm


plt.rcParams['text.usetex'] = True

Nstate = 2
N  = 40
FS = 15

std = 0.01

Start  = [0,0]
Finish = [1,1]
gain   = 0.1
inertia = 0.5
NMC     = 2000
NMCplot = 30

Time = [k for k in range(N)]
End = 1

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
    """
    if Time:
        ax1.fill([Box[0,0], Box[0,1], Box[0,1], Box[0,0]],[Box[1,0],Box[1,0],Box[1,1], Box[1,1]  ],color=CO),alpha=0.1)
    """
    ax1.plot([Box[0,0],Box[0,1]],[Box[1,0],Box[1,0]],color=CO,linewidth=LW)
    ax1.plot([Box[0,1],Box[0,1]],[Box[1,0],Box[1,1]],color=CO,linewidth=LW)
    ax1.plot([Box[0,1],Box[0,0]],[Box[1,1],Box[1,1]],color=CO,linewidth=LW)
    ax1.plot([Box[0,0],Box[0,0]],[Box[1,1],Box[1,0]],color=CO,linewidth=LW)

    
    
    
    return Box



spastAll = []
spredAll = []

Finish = np.array(Finish).reshape(2,1)

C      = np.array([.5,.75]).reshape(2,1)
D      = 0.2

Angle  = np.linspace(0,2*np.pi,1000)


tau    = 0.1

Vec    = np.zeros(2).reshape(2,1)
Vecsys = np.zeros(2).reshape(2,1)

spast = np.array(Start).reshape(2,1)
for now in range(0,End):
    spred = [spast[:,-1].reshape(2,1)]*NMC

    for time in Time:
        for n in range(0,NMC):
            Vec  = inertia*Vec + (1-inertia)*(Finish - spred[n][:,-1].reshape(2,1))
            
            Vec2C = spred[n][:,-1].reshape(2,1) - C
            
            Dist = np.linalg.norm(Vec2C)
            if Dist - D > 0:
                F    = .01 / (  Dist - D  )
            else:
                F    = -(  Dist - D  )**2
            Vec += F*Vec2C/Dist
            
            sps = spred[n][:,-1].reshape(2,1) + gain*Vec.reshape(2,1) + np.random.normal(0,std,2).reshape(2,1)#
            spred[n] = np.concatenate( (spred[n],sps), axis=1)
           

    Vecsys = inertia*Vecsys + (1-inertia)*(Finish - spast[:,-1].reshape(2,1))
    snew = spast[:,-1].reshape(2,1) + gain*Vecsys.reshape(2,1) + np.random.normal(0,std,2).reshape(2,1)#
    
    spastAll.append(spast)
    spredAll.append(spred)
    
    spast = np.concatenate( (spast,snew.reshape(2,1)), axis=1)
        
for now in range(0,End):

        plt.close('all')

        figID = plt.figure(1,figsize=(12,6))
        
        ax1 = figID.add_subplot(111)
        #ax1.plot(D*np.cos(Angle)+C[0],D*np.sin(Angle)+C[1],color='r')
        ax1.fill(D*np.cos(Angle)+C[0],D*np.sin(Angle)+C[1],color='r')
        
        
                
        plt.show(block=False)
        #sys.exit()
        
        ax1.plot(spastAll[now][0,:],spastAll[now][1,:],color=[0,0,0.5],linewidth=2,marker='o')
        
        
        EndMass = {'x': [], 'y' : []}
        for n in range(0,NMCplot):
            ax1.plot(spredAll[now][n][0,:],spredAll[now][n][1,:],color='b',linewidth=1,marker='.')
            EndMass['x'].append(spredAll[now][n][0,-1])
            EndMass['y'].append(spredAll[now][n][1,-1])
            
        MakeBox(spredAll,'r')

        #ax1.plot(1,1,color='c',linewidth=2,marker='o')
        
        
        EndD = np.max([np.std(EndMass['x']),np.std(EndMass['y'])])
        ax1.plot(2*EndD*np.cos(Angle)+np.mean(EndMass['x']),2*EndD*np.sin(Angle)+np.mean(EndMass['y']),color='c',linewidth=2)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_axis_off()
        ax1.set_aspect('equal', 'box')
     

        
        for time in range(now,N+1):
            Box = MakeBox(spredAll,[.7,0,0],Time = [now,time],LW = 1)
            plt.show(block=False)
            plt.pause(0.1)
            figID.savefig('PessimisticModel'+str(time)+'.eps', transparent=True, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'
            









    
    





