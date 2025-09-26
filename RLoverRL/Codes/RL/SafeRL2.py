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

C      = np.array([.5,.75]).reshape(2,1)
D      = 0.3

Time = [k for k in range(N)]


tau    = 20

FgainList = list(np.linspace(0,0.9,10))

Finish = np.array(Finish).reshape(2,1)



Angle  = np.linspace(0,2*np.pi,1000)




for kGain, Fgain in enumerate(FgainList):

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
    for traj in Safe:#[0:51]:
        ax1.plot(traj[0,:],traj[1,:],color='b',linewidth=1,marker='.')
        EndMass['x'].append(traj[0,-1])
        EndMass['y'].append(traj[1,-1])
    for traj in Violation:#[0:1]:
        ax1.plot(traj[0,:],traj[1,:],color=[.8,0,0],linewidth=1,marker='.')
        EndMass['x'].append(traj[0,-1])
        EndMass['y'].append(traj[1,-1])
        
    #MakeBox(spred,'r')



    EndD = np.max([np.std(EndMass['x']),np.std(EndMass['y'])])
    ax1.plot(2*EndD*np.cos(Angle)+np.mean(EndMass['x']),2*EndD*np.sin(Angle)+np.mean(EndMass['y']),color='c',linewidth=2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_axis_off()
    ax1.set_aspect('equal', 'box')

    plt.show(block=False)
    plt.pause(0.1)
    figID.savefig('RLSafety'+str(kGain)+'.pdf', transparent=True, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'


















