import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

N  = 20
FS = 15
a  = 0.95
NMC = 500

Time = [k for k in range(N)]
End = 15

spast = [np.random.normal(1,0.05,NMC)]
tpast = []
    
PolSet = np.linspace(.2,.8,10)#np.logspace(np.log10(.2),np.log10(.8),10)
Perf   = []
for poln, pol in enumerate(PolSet):
    apred = []
    spred = [spast[-1]]
    for time in Time:
        
        ap = []
        sp = []
        for n in range(0,NMC):
            aps = -pol*spred[-1][n]
            sps = a*spred[-1][n] + 1.5*aps + np.random.normal(0,0.1)
            sp.append(sps)
            ap.append(aps)

        apred.append(ap)
        spred.append(sp)


    splot = [[spast[-1]]*NMC]+spred[1:]
    tplot = tpast + Time


    snew = spred[1][-1] + np.random.normal(0,0.1)
    
    plt.close('all')

    figID = plt.figure(1,figsize=(12,6))
    ax1 = figID.add_subplot(211)
    ax2 = figID.add_subplot(212)


    ax1.step(np.array(Time),apred,where='post',color=[0.8,0,0],linewidth=1)
        
    
    
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('Time',fontsize=FS)
    ax1.set_ylabel('Actions',fontsize=FS)
    #ax1.text(now,1,'Now',rotation=90,horizontalalignment='right',verticalalignment='top',fontsize=FS,color=[0,.7,.1])
    #ax1.plot([now+N-1,now+N-1],[-1,1],linestyle=':',color='k')
    #ax1.text(now+N-1,1,'Horizon',rotation=90,horizontalalignment='left',verticalalignment='top',fontsize=FS,color='k')

    ax1.set_xlim(0,N+End)



            
    ax2.plot(np.array(Time),spred[:-1],color='b',linewidth=1)

    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel('Time',fontsize=FS)
    ax2.set_ylabel('State',fontsize=FS)
    #ax2.text(now,0,'Now',rotation=90,horizontalalignment='right',verticalalignment='top',fontsize=FS,color=[0,.7,.1])
    ax1.set_xlim(0,N-1)
    ax2.set_xlim(0,N-1)
    ax2.set_ylim(-.5,1.5)
    
    
    Perf.append( (-np.sum(np.array(apred)**2) - np.sum(np.array(spred)**2)) / NMC )
    
    figID2 = plt.figure(2,figsize=(6,6))
    ax3 = figID2.add_subplot(111)

    ax3.plot(PolSet[:poln+1],Perf,marker='.')
    ax3.set_xlim(PolSet[0],PolSet[-1])
    ax3.set_ylim(-3.5,-1.5)
    plt.show(block=False)
    plt.pause(.1)
    
    figID2.savefig('PolicyPerf'+str(poln)+'.eps', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'
    figID.savefig('MDPPerf'+str(poln)+'.eps', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'

    #




