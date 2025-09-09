import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

N  = 20
FS = 15
a  = 1
NMC = 50

Time = [k for k in range(N)]
End = 9

apast = []
spast = [1]
tpast = []
    
for now in range(0,End):
    apred = []
    spred = [[spast[-1]]*NMC]
    for time in Time:

        sp = []
        for n in range(0,NMC):
            apred.append(-spred[-1][n]) #+ 2*np.tanh(10*spred[-1][n])/10)
            sps = a*spred[-1][n] + 0.25*apred[-1] + np.random.normal(0,0.1)
            sp.append(sps)

        spred.append(sp)


    splot = [[spast[-1]]*NMC]+spred[1:]
    tplot = tpast + Time


    snew = spred[1][-1] + np.random.normal(0,0.1)
    
    plt.close('all')

    figID = plt.figure(1,figsize=(12,6))
    ax1 = figID.add_subplot(211)
    ax2 = figID.add_subplot(212)


    ax1.step(tpast+[now],apast+[apred[0]],where='post',color=[0.5,0,0],linewidth=2)
    #ax1.step(now+np.array(Time),apred,where='post',color=[0.8,0,0],linewidth=1)
    ax1.plot([now,now+1],[apred[0],apred[0]],color=[1,0,0],linewidth=2)
        
    
    ax1.plot([now,now],[-1,1],linestyle=':',color=[0,.7,.1])
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('Time',fontsize=FS)
    ax1.set_ylabel('Actions',fontsize=FS)
    #ax1.text(now,1,'Now',rotation=90,horizontalalignment='right',verticalalignment='top',fontsize=FS,color=[0,.7,.1])
    #ax1.plot([now+N-1,now+N-1],[-1,1],linestyle=':',color='k')
    #ax1.text(now+N-1,1,'Horizon',rotation=90,horizontalalignment='left',verticalalignment='top',fontsize=FS,color='k')

    ax1.set_xlim(0,N+End)


    #ax2.plot(tplot,splot[:-1],marker='.',linewidth=2)
    if tpast:
        #ax2.plot(tpast+[now],spast[:-1]+[spred[0]],color=[0,0,0.5],linewidth=2)
        ax2.plot(tpast+[now],spast,color=[0,0,0.5],linewidth=2,marker='o')
        for t in tpast:
            ax2.text(t+0.05,spast[t]+0.05,r'$\bf{s}_{'+str(t)+'}$',horizontalalignment='left',verticalalignment='bottom',fontsize=20,color=[0,0,0.5])

            
    ax2.plot(now+np.array(Time),spred[:-1],color='b',linewidth=1)
    ax2.plot([now],[spred[0]],color=[.8,0,0],linewidth=2,marker='o')


    ax2.text(now+0.05,spred[0][0]+0.05,r'$\bf{s}_{'+str(now)+'}$',horizontalalignment='left',verticalalignment='bottom',fontsize=30,color=[.8,0,0])
    ax1.text(now+0.05,apred[0]+0.05,r'$\bf{a}_{'+str(now)+'} = \pi({s}_'+str(now)+')$',horizontalalignment='left',verticalalignment='bottom',fontsize=25,color=[.8,0,0])

    #= \bf{\pi}(\bf{s}_{'+str(now)+'})
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel('Time',fontsize=FS)
    ax2.set_ylabel('State',fontsize=FS)
    #ax2.text(now,0,'Now',rotation=90,horizontalalignment='right',verticalalignment='top',fontsize=FS,color=[0,.7,.1])
    ax1.set_xlim(0,19)
    ax2.set_xlim(0,19)
    ax2.set_ylim(-.5,1.2)
    
    plt.show(block=False)



    tpast.append(now)
    apast.append(apred[0])
    spast.append(snew)
    
    #sys.exit()
    figID.savefig('MDP'+str(now)+'.eps', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'
    
    plt.pause(1)




