import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm

plt.rcParams['text.usetex'] = True


N = 10

a = []
for k in range(0,N):
    a.append( 3*(k+1-N/2)*np.array([1.2,1])/N )

LW = 2
for k in range(0,N):
    plt.close('all')

    figID = plt.figure(1,figsize=(8,8))

    ax1 = figID.add_subplot(111)

    ax1.text(-1,1.01,r'$\bf S(s)$',fontsize=30,verticalalignment='bottom',horizontalalignment='left',color='r')
    ax1.plot([-1,1],[1,1],color='r',linewidth=LW)
    ax1.plot([1,1],[1,-1],color='r',linewidth=LW)
    ax1.plot([-1,1],[-1,-1],color='r',linewidth=LW)
    ax1.plot([-1,-1],[1,-1],color='r',linewidth=LW)

    ax1.plot([-1.1,1.1],[0,0],color='k',linewidth=1)
    ax1.plot([0,0],[-1.1,1.1],color='k',linewidth=1)
    ax1.text(1.01,.01,r'$\bf a_1$',fontsize=30,verticalalignment='bottom',horizontalalignment='left',color='k')
    ax1.text(.01,1.01,r'$\bf a_2$',fontsize=30,verticalalignment='bottom',horizontalalignment='left',color='k')

    ax1.set_axis_off()


    aproj = [np.max([-1,np.min([a[k][0],1])]),
             np.max([-1,np.min([a[k][1],1])])]
    ax1.plot([a[k][0],aproj[0]],[a[k][1],aproj[1]],linestyle='-',marker='none',color='k',linewidth=1)
    ax1.plot(a[k][0],a[k][1],linestyle='none',marker='o',color='r',markersize=10)
    ax1.plot(aproj[0],aproj[1],linestyle='none',marker='o',color='b',markersize=10)

    ax1.set_xlim([-1.5,1.85])
    ax1.set_ylim([-1.25,1.55])
    
    plt.show(block=False)
    plt.pause(0.1)
    #sys.exit()
    figID.savefig('Projection'+str(k)+'.eps', transparent=True, dpi='figure', format=None,metadata=None, bbox_inches='tight', facecolor='auto',edgecolor='auto', backend=None) #



    
    





