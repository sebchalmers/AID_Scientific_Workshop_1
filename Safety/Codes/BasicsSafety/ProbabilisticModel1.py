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
NMC     = 1000


Time = [k for k in range(N)]


def MakeBox(spred, CO, Time = [], LW = 3):
    
    Box = np.array([[np.inf,-np.inf],[np.inf,-np.inf]])
        
    for n in range(0,NMC):
        Box[0,0] = np.min([ Box[0,0], np.min(spred[n][0])  ])
        Box[0,1] = np.max([ Box[0,1], np.max(spred[n][0])  ])
        Box[1,0] = np.min([ Box[1,0], np.min(spred[n][1])  ])
        Box[1,1] = np.max([ Box[1,1], np.max(spred[n][1])  ])
    """
    if Time:
        ax1.fill([Box[0,0], Box[0,1], Box[0,1], Box[0,0]],[Box[1,0],Box[1,0],Box[1,1], Box[1,1]  ],color=CO),alpha=0.1)
    """
    ax1.plot([Box[0,0],Box[0,1]],[Box[1,0],Box[1,0]],color=CO,linewidth=LW)
    ax1.plot([Box[0,1],Box[0,1]],[Box[1,0],Box[1,1]],color=CO,linewidth=LW)
    ax1.plot([Box[0,1],Box[0,0]],[Box[1,1],Box[1,1]],color=CO,linewidth=LW)
    ax1.plot([Box[0,0],Box[0,0]],[Box[1,1],Box[1,0]],color=CO,linewidth=LW)

    
    
    
    return Box


Finish = np.array(Finish).reshape(2,1)

C      = np.array([.5,.75]).reshape(2,1)
D      = 0.2

Angle  = np.linspace(0,2*np.pi,1000)

tau    = 20

Fgain  = 0.5

"""
#########
Dall = np.linspace(0,1,100)
Dall = list(Dall)
F    = []
for Dist in Dall:
    
   
    if Dist - D > 0:
        F.append( np.exp( -tau*(Dist - D) ) )
    else:
        F.append( 1  - tau * (  Dist - D  ) )
        
plt.figure(10)
plt.plot(Dall,F)
plt.show(block=False)
#sys.exit()
#########
"""

Vec    = np.zeros(2).reshape(2,1)
Vecsys = np.zeros(2).reshape(2,1)


spred = [np.array([0,0]).reshape(2,1)]*NMC

for time in Time:
    for n in range(0,NMC):
        Vec  = inertia*Vec + (1-inertia)*(Finish - spred[n][:,-1].reshape(2,1))
        
        Vec2C = spred[n][:,-1].reshape(2,1) - C
        
        Dist = np.linalg.norm(Vec2C)

        if Dist - D > 0:
            F = np.exp( -tau*(Dist - D) )
        else:
            F =  1  - tau * (  Dist - D  )

        Vec += Fgain*F*Vec2C/Dist
        
        sps = spred[n][:,-1].reshape(2,1) + gain*Vec.reshape(2,1) + np.random.normal(0,std,2).reshape(2,1)#
        spred[n] = np.concatenate( (spred[n],sps), axis=1)
               

#Flag Trajectories
Safe      = []
Violation = []
for k, traj in enumerate(spred):
    Err = traj-C
    DistTraj = np.min(np.sqrt(Err[0,:]**2 + Err[1,:]**2)-D)
    if DistTraj < 0:
        Violation.append(traj)
    else:
        Safe.append(traj)

plt.close('all')

figID = plt.figure(1,figsize=(12,6))

ax1 = figID.add_subplot(111)
ax1.fill(D*np.cos(Angle)+C[0],D*np.sin(Angle)+C[1],color='r')


EndMass = {'x': [], 'y' : []}
for traj in Safe[0:51]:
    ax1.plot(traj[0,:],traj[1,:],color='b',linewidth=1,marker='.')
    EndMass['x'].append(traj[0,-1])
    EndMass['y'].append(traj[1,-1])
for traj in Violation[0:1]:
    ax1.plot(traj[0,:],traj[1,:],color=[.8,0,0],linewidth=1,marker='.')
    EndMass['x'].append(traj[0,-1])
    EndMass['y'].append(traj[1,-1])
    
MakeBox(spred,'r')



EndD = np.max([np.std(EndMass['x']),np.std(EndMass['y'])])
ax1.plot(2*EndD*np.cos(Angle)+np.mean(EndMass['x']),2*EndD*np.sin(Angle)+np.mean(EndMass['y']),color='c',linewidth=2)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_axis_off()
ax1.set_aspect('equal', 'box')

plt.show(block=False)

figID.savefig('ProbabilisticModel.eps', transparent=True, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'


















