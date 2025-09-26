import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

C = np.logspace(-3,-1,3)
for c in(list(C)):
    Ns    = 1e3
    Ndraw = 1e7

    Wei = [1-c,c]


    meanS = []
    varS  = []
    for k in range(0,int(Ndraw)):
        S   = np.random.choice([0,1],int(Ns),p=Wei)
        meanS.append( np.mean(S) )
        varS.append( np.var(S)  )

    [H,s] = np.histogram(meanS,bins=20,density=True)
    s = 0.5*s[:-1] + 0.5*s[1:]

    Ipos = np.where(H>0)
    H = H[Ipos]
    s = s[Ipos]


    plt.close('all')
    figID = plt.figure(1,figsize=(12,6))
    ax1 = figID.add_subplot(111)
    ax1.set_xscale('log')

    ax1.fill_between(s,H,np.zeros(len(s)),color='b',step='mid')
    ax1.plot([c,c],[0,np.max(H)],color='r',linewidth=3)
    ax1.text(.5*1e1*c,0.8*np.max(H),r'Draw $10^{'+str(int(np.log10(Ns)))+'}$ trajectories',fontsize=20,horizontalalignment='right')
    ax1.set_yticks([])
    ax1.set_xlabel('Estimation of prob. of failure ',fontsize=30)
    ax1.tick_params(axis='both', which='major', labelsize=30)
    ax1.set_xlim([0.1*c,10*c])
    #ax1.set_axis_off()
    
    plt.show(block=False)
    plt.pause(0.1)
    #sys.exit()

    figID.savefig('SafetyEstimation'+str(int(np.log10(Ns)))+'_'+str(int(-np.log10(c)))+'.pdf', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'

