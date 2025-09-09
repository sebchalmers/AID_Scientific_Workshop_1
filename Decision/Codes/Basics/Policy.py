import sys
import numpy as np
import matplotlib.pyplot as plt

N  = 30
FS = 15
a  = 0.9
NMC = 10

Time = [k for k in range(N)]
End = 15

def Barrier(x):
    Thresh = 0.01
    Tau    = 0.15
    if x > Thresh:
        barrier = - Tau * np.log(x)
    else:
        barrier = - Tau * x / Thresh
    return barrier

apast = []
spast = [1]
tpast = []
for now in range(0,End):
    apred = []
    spred = [[spast[-1]]*NMC]
    for time in Time:
        if time == 0:
            ap = [-1*(spast[-1]) + Barrier(spast[-1]) ]*NMC
        else:
            ap = []
            for n in range(0,NMC):
                ap.append(   -1*spred[-1][n] + Barrier(spred[-1][n]) )
            
        sp = []
        for n in range(0,NMC):
            sps = a*spred[-1][n] + 0.5*ap[n] + np.random.uniform(-0.2,0.2)
            sp.append(sps)

        spred.append(sp)
        apred.append(ap)


    snew = a*spast[-1] + 0.5*apred[0][0] + np.random.uniform(-0.2,0.2)
    
    plt.close('all')

    figID = plt.figure(1,figsize=(6,10))
    ax1 = figID.add_subplot(211)
    ax2 = figID.add_subplot(212)

    ax1.step(tpast+[now],apast+[apred[0]],where='post',color=[0.6,0,0],linewidth=2)
    ax1.step(now+np.array(Time),apred,where='post',color=[1,0.7,0.7],linewidth=1)
    ax1.plot([now,now+1],[apred[0],apred[0]],color=[1,0,0],linewidth=2)
        
    
    ax1.plot([now,now],[-1,1],linestyle=':',color=[0,.7,.1])
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('Time',fontsize=FS)
    ax1.set_ylabel('Actions',fontsize=FS)
    ax1.text(now,1,'Now',rotation=90,horizontalalignment='left',verticalalignment='top',fontsize=FS,color=[0,.7,.1])

    ax1.set_xlim(0,N-1)

    if tpast:
        ax2.plot(tpast+[now],spast[:-1]+[spred[0][0]],color=[0,0,0.5],linewidth=2)
    ax2.plot(now+np.array(Time),spred[:-1],color=[.75,.75,1],linewidth=1)
    #ax2.plot([now,now+1],[spred[0][-1],snew],color='c',linewidth=2)
    #ax2.plot([now+1],[snew],color='c',linewidth=2,marker='o')
    
    ax2.plot([now],[spred[0]],color='g',linewidth=2,marker='o')
    ax2.plot([now,now],[-0.5,1.5],linestyle=':',color=[0,.7,.1])
    ax2.plot([0,N],[0,0],color='r',linewidth=2)

    ax2.fill_between([0,N], [0,0], [-.5,-.5],color=[1,0.7,.7])

    ax2.set_ylim(-.5,1.5)
    
    ax2.set_xticks([])
    #ax2.set_yticks([])
    ax2.set_xlabel('Time',fontsize=FS)
    ax2.set_ylabel('Info',fontsize=FS)
    ax2.text(now,1.5,'Now',rotation=90,horizontalalignment='left',verticalalignment='top',fontsize=FS,color=[0,.7,.1])
    ax2.set_xlim(0,N-1)
    
    plt.show(block=False)

    tpast.append(now)
    apast.append(apred[0])
    spast.append(snew)
    plt.pause(.1)
        
    #sys.exit()
    figID.savefig('Policy'+str(now)+'.eps', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'
    
    




