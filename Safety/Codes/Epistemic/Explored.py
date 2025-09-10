import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
FS = 40




x = np.linspace(-1,1,100)
y = np.linspace(-1,1,99)
xv, yv = np.meshgrid(x, y, indexing='ij')


x = np.linspace(-1,1,1000)
Model = 0

Mu  = [np.array([-.5,-.5]),np.array([0.5,1])]
Std = [3*np.eye(2),1*np.eye(2)]
Wei = [.6,.4]
for k in range(0,2):
    #Mu.append(np.random.uniform(-.3,0,2))
    #std = np.random.uniform(.1,.1,4).reshape(2,2)
    #Std.append(np.matmul(std,std.T))
    #Wei.append(np.random.uniform(.3,.7))

    #Model += Wei[k]*np.exp( - (x - Mu[k])**2/Std[k]**2/2 ) / np.sqrt(2*np.pi*Std[k]**2)
    Expon = 0.5*Std[k][0,0]*(xv - Mu[k][0])**2 + 0.5*Std[k][1,1]*(yv - Mu[k][1])**2 + Std[k][0,1]*(yv - Mu[k][1])*(xv - Mu[k][0])
    Model += Wei[k]*np.exp( - Expon )


    
plt.close('all')
figID = plt.figure(1,figsize=(12,6))
ax1 = figID.add_subplot(111)

ax1.pcolormesh(xv, yv, Model, cmap='Blues')
ax1.set_aspect('equal', 'box')
ax1.set_xticks([])
ax1.set_yticks([])

#ax1.set_xlabel(r'$\bf{s}_{+}$',fontsize=FS)
#ax1.set_ylabel(r'$\bf{P}(s_+|s,a)$',fontsize=FS,color='r')
#ax1.text(x[700],Model[700],r'$\bf{ P}_{\theta}(s_+|s,a)$',fontsize=FS,color='r',verticalalignment='bottom',horizontalalignment='left')

plt.show(block=False)
plt.pause(1)
#sys.exit()

figID.savefig('Explored_SA.jpeg', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'

