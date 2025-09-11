import sys
import numpy as np
import matplotlib.pyplot as plt
#import scipy as sc

plt.rcParams['text.usetex'] = True
FS = 40

Mu  = [np.array([-.5,-.5])]#,np.array([0.75,.75])]
Std = [3*np.eye(2)]#,1*np.eye(2)]
Wei = [1]#.6,.4]

weight = 1

A = 1
B = 0.25

sref = .25
aref = .25

Tol  = 1e-4

N = 200

x = np.linspace(-1,1,100)
y = np.linspace(-1,1,99)
xv, yv = np.meshgrid(x, y, indexing='ij')

# Distribution
x = np.linspace(-1,1,1000)
ModelPlot = 0
for k in range(0,len(Wei)):
    Expon = 0.5*Std[k][0,0]*(xv - Mu[k][0])**2 + 0.5*Std[k][1,1]*(yv - Mu[k][1])**2 + Std[k][0,1]*(yv - Mu[k][1])*(xv - Mu[k][0])
    ModelPlot += Wei[k]*np.exp( - Expon )

# DP on guarded policy
ns = 300
na = 301

sg = np.linspace(-1,1,ns)
ag = np.linspace(-1,1,na)
sv, av = np.meshgrid(sg, ag, indexing='ij')

ModelDP = 0
for k in range(0,len(Wei)):
    Expon = 0.5*Std[k][0,0]*(sv - Mu[k][0])**2 + 0.5*Std[k][1,1]*(av - Mu[k][1])**2 + Std[k][0,1]*(av - Mu[k][1])*(sv - Mu[k][0])
    ModelDP += Wei[k]*np.exp( - Expon )
    


def DP(weight):
    V     = np.zeros(ns)
    gamma = 0.99

    iter = True
    while iter:
        L       = 0.5*(sv-sref)**2 + 0.5*(av-aref)**2 - weight*ModelDP
        #L      += 2*(sv < -.9)*(sv + 0.9)
        #L      += 2*(sv > +.9)*(sv - 0.9)
        
        splus   = A*sv + B*av
        Vplus   = np.interp(splus,sg,V)
        Q       = L + gamma*Vplus
        
        Vprev   = V
        V       = np.min(Q,axis=1)
        
        Residual = np.linalg.norm(V-Vprev)
        print(np.log10(Residual))
        if Residual < Tol:
            iter = False
            
    Indices = np.argmin(Q,axis=1)
    Pi      = ag[Indices]

    return Pi

Pi = {'Star' : DP(0), 'OOD' : DP(weight)}

def Policy(s,Pi):
    a = np.interp(s,sg,Pi)
    return a

spolplot     = np.linspace(-1,1,100)
apolplot     = {}
for key in Pi:
    apolplot[key] = Policy(spolplot,Pi[key])

Lplot = {'Star' : 0.5*(sv-sref)**2 + 0.5*(av-aref)**2,
         'OOD'  : 0.5*(sv-sref)**2 + 0.5*(av-aref)**2 - weight*ModelDP}

for case in ['Map','Star','OOD']:

    print(case)
    
    if case == 'Map':
        plt.close('all')
        figID = plt.figure(1,figsize=(12,6))
        ax1 = figID.add_subplot(111)
        
        ax1.pcolormesh(xv, yv, ModelPlot, cmap='Blues')
        ax1.set_aspect('equal', 'box')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlabel('State',fontsize=FS)
        ax1.set_ylabel('Action',fontsize=FS)
        ax1.contour(sv,av,Lplot['Star'],colors='k',linestyles='solid')
        ax1.plot(sref,aref,marker='o',color='b')
        
        figID.savefig('OOD_Map.jpeg', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'

    else:
        x0 = .75

        SA = np.array([[x0, Policy(x0,Pi[case])]]).reshape(2,1)

        for k in range(0,N):
            splus   = A*SA[0,-1] + B*SA[1,-1] + np.random.normal(0,0.05)
            aplus   = Policy(splus,Pi[case])
            
            SA = np.concatenate((SA,np.array([splus,aplus]).reshape(2,1)),axis=1)


        plt.close('all')
        figID = plt.figure(1,figsize=(12,6))
        ax1 = figID.add_subplot(111)

        ax1.pcolormesh(xv, yv, ModelPlot, cmap='Blues')
        ax1.set_aspect('equal', 'box')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlabel('State',fontsize=FS)
        ax1.set_ylabel('Action',fontsize=FS)

        ax1.plot(sref,aref,marker='o',color='b')
        ax1.contour(sv,av,Lplot[case],colors='k',linestyles='solid')
                
        
        ax1.plot(spolplot,apolplot['Star'],color='g',linewidth=3)
        
       
        
        if case == 'OOD':
            ax1.plot(spolplot,apolplot['OOD'],color='m',linewidth=3)
            ax1.text(.5,Policy(.5,Pi['OOD']),r'$\bf\pi_{OOD}(s)$',fontsize=30,verticalalignment='bottom',horizontalalignment='left',color='m')
       
        ax1.plot(SA[0,:],SA[1,:],marker='o',color='r')
        
        ax1.text(.5,Policy(.5,Pi['Star']),r'$\bf\pi(s)$',fontsize=30,verticalalignment='bottom',horizontalalignment='left',color='g')

        plt.show(block=False)
        
        
        

        figID.savefig('OOD_'+case+'.jpeg', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'

