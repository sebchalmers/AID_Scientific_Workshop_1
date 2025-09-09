import sys
import numpy as np
import matplotlib.pyplot as plt

N  = 15
FS = 15
a  = 1.2

Time = [k for k in range(N)]
End = 15

apast = []
spast = [1]
tpast = []
    
for now in range(0,End):
    apred = []
    spred = [spast[-1]]
    for time in Time:
        apred.append(-1*spred[-1])
        
        sp = a*spred[-1] + 0.5*apred[-1]
        spred.append(sp)


    aplot = apast + apred
    splot = spast + spred[1:]
    tplot = tpast + Time


    snew = spred[1] + np.random.uniform(-0.1,0.1)
    
    plt.close('all')

    figID = plt.figure(1,figsize=(6,10))
    ax1 = figID.add_subplot(211)
    ax2 = figID.add_subplot(212)

    ax1.step(tpast+[now],apast+[apred[0]],where='post',color=[0.5,0,0],linewidth=2)
    ax1.step(now+np.array(Time),apred,where='post',color=[0.8,0,0],linewidth=1)
    ax1.plot([now,now+1],[apred[0],apred[0]],color=[1,0,0],linewidth=2)
        
    
    ax1.plot([now,now],[-1,1],linestyle=':',color=[0,.7,.1])
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('Time',fontsize=FS)
    ax1.set_ylabel('Action plan',fontsize=FS)
    ax1.text(now,1,'Now',rotation=90,horizontalalignment='right',verticalalignment='top',fontsize=FS,color=[0,.7,.1])
    ax1.plot([now+N-1,now+N-1],[-1,1],linestyle=':',color='k')
    ax1.text(now+N-1,1,'Horizon',rotation=90,horizontalalignment='left',verticalalignment='top',fontsize=FS,color='k')

    ax1.set_xlim(0,N+End)

    #ax2.plot(tplot,splot[:-1],marker='.',linewidth=2)
    if tpast:
        ax2.plot(tpast+[now],spast[:-1]+[spred[0]],color=[0,0,0.5],linewidth=2)
    ax2.plot(now+np.array(Time),spred[:-1],color='b',linewidth=1)
    ax2.plot([now,now+1],[spred[0],snew],color='c',linewidth=2)
    
    ax2.plot([now+1],[snew],color='c',linewidth=2,marker='o')
    ax2.plot([now],[spred[0]],color='g',linewidth=2,marker='o')
    ax2.plot([now,now],[-0.5,1.5],linestyle=':',color=[0,.7,.1])
    ax2.plot([now+N-1,now+N-1],[-0.5,1.5],linestyle=':',color='k')
    ax2.text(now+N-1,1.5,'Horizon',rotation=90,horizontalalignment='left',verticalalignment='top',fontsize=FS,color='k')


    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel('Time',fontsize=FS)
    ax2.set_ylabel('Predictions',fontsize=FS)
    ax2.text(now,1.5,'Now',rotation=90,horizontalalignment='right',verticalalignment='top',fontsize=FS,color=[0,.7,.1])
    ax2.set_xlim(0,N+End)
    
    plt.show(block=False)

    tpast.append(now)
    apast.append(apred[0])
    spast.append(snew)
    
    figID.savefig('MPC'+str(now)+'.eps', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'
    
    plt.pause(1)




