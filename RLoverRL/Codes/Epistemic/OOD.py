import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
FS = 40

Mu  = [np.array([-.5,-.5]),np.array([0.5,1])]
Std = [3*np.eye(2),1*np.eye(2)]
Wei = [.6,.4]

Gain = 0.1
angle = 10*np.pi/180
A    = 0.9*np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])

N = 50

x = np.linspace(-1,1,100)
y = np.linspace(-1,1,99)
xv, yv = np.meshgrid(x, y, indexing='ij')

# Distribution
x = np.linspace(-1,1,1000)
Model = 0

for k in range(0,len(Wei)):
    Expon = 0.5*Std[k][0,0]*(xv - Mu[k][0])**2 + 0.5*Std[k][1,1]*(yv - Mu[k][1])**2 + Std[k][0,1]*(yv - Mu[k][1])*(xv - Mu[k][0])
    Model += Wei[k]*np.exp( - Expon )

# Guarded trajectory


SA = np.array([[-0.4, 0.4]]).reshape(2,1)

for k in range(0,N):

    Gradient = 0
    for k in range(0,len(Wei)):
        sa = SA[:,-1]
        Expon = 0.5*Std[k][0,0]*(sa[0] - Mu[k][0])**2 + 0.5*Std[k][1,1]*(sa[1] - Mu[k][1])**2 + Std[k][0,1]*(sa[1] - Mu[k][1])*(sa[0] - Mu[k][0])
        
        #Expon2 = np.matmul(Std[k],np.array([sa[0] - Mu[k][0],sa[1] - Mu[k][1]]))
        #Expon2 = 0.5*np.matmul(  np.array([sa[0] - Mu[k][0],sa[1] - Mu[k][1]]), Expon2 )
        
        #assert(Expon == Expon2)
        #Model += Wei[k]*np.exp( - Expon )
        Gradient += -Wei[k]*np.exp( - Expon )*np.matmul(Std[k],np.array([sa[0] - Mu[k][0],sa[1] - Mu[k][1]]))

    
        
    plt.close('all')
    figID = plt.figure(1,figsize=(12,6))
    ax1 = figID.add_subplot(111)

    ax1.pcolormesh(xv, yv, Model, cmap='Blues')
    ax1.set_aspect('equal', 'box')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('State',fontsize=FS)
    ax1.set_ylabel('Action',fontsize=FS)


    ax1.plot(SA[0,:],SA[1,:],marker='o',color='r')
    ax1.plot([SA[0,-1],SA[0,-1]+Gain*Gradient[0]],[SA[1,-1],SA[1,-1]+Gain*Gradient[1]],color='k')
    
    
    plt.show(block=False)
    #sys.exit()

    plt.pause(1)

    saplus = np.matmul(A,SA[:,-1]) + np.random.normal(0,0.0,2) + Gain*Gradient
    
    SA = np.concatenate((SA,saplus.reshape(2,1)),axis=1)

    #sys.exit()
    #figID.savefig('OOD_SA.jpeg', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'

