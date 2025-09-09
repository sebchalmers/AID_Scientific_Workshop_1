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

def MakeBox(s, CO, LW = 3):
    
    Box = np.array([[np.inf,-np.inf],[np.inf,-np.inf]])
    for n in range(0,NMC):
        Box[0,0] = np.min([ Box[0,0], np.min(s[n][0])  ])
        Box[0,1] = np.max([ Box[0,1], np.max(s[n][0])  ])
        Box[1,0] = np.min([ Box[1,0], np.min(s[n][1])  ])
        Box[1,1] = np.max([ Box[1,1], np.max(s[n][1])  ])

    ax1.plot([Box[0,0],Box[0,1]],[Box[1,0],Box[1,0]],color=CO,linewidth=LW)
    ax1.plot([Box[0,1],Box[0,1]],[Box[1,0],Box[1,1]],color=CO,linewidth=LW)
    ax1.plot([Box[0,1],Box[0,0]],[Box[1,1],Box[1,1]],color=CO,linewidth=LW)
    ax1.plot([Box[0,0],Box[0,0]],[Box[1,1],Box[1,0]],color=CO,linewidth=LW)

    
    return Box


s  = np.array([.75,.75])
sp = []
for n in range(0,NMC):
    sp.append( np.matmul(A,s) + np.random.uniform(-std,std,2))


plt.close('all')

figID = plt.figure(1,figsize=(12,6))

ax1 = figID.add_subplot(111)
Box = MakeBox(sp,'r',LW = 2)


ax1.text(s[0],s[1],r'$\bf s, a$',verticalalignment='bottom',horizontalalignment='left',fontsize=20,color='c')
for n in range(0,NMCplot):
    ax1.plot([s[0],sp[n][0]],[s[1],sp[n][1]],color='b',linewidth=1,marker='.')
    ax1.text(sp[n][0],sp[n][1],r'$\bf s_+$',verticalalignment='bottom',horizontalalignment='left',fontsize=15,color='b')
ax1.plot(s[0],s[1],color='c',linewidth=2,marker='o',markerfacecolor='c')




ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_axis_off()
ax1.set_aspect('equal', 'box')

plt.show(block=False)

figID.savefig('PessimisticOneStepr.eps', transparent=True, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'










    
    





