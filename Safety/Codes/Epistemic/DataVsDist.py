import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
FS = 40

Nsamples = np.logspace(np.log10(40),6,5)
Nsamples = [ int(Nsamples[k]) for k in range(0,Nsamples.shape[0]) ]
#Nsamples = [40]
#Nsamples = [int(1e6)]
for i, Ns in enumerate(Nsamples):

    Nbins = 1000#int(Ns/10)

    S  = []
    Mu  = [-.3,.2]
    Std = [.1,.2]
    Wei = [.2,.8]

    s = np.linspace(-1,1,Nbins)

    for k in range(0,Ns):
        bin = np.random.choice(range(0,len(Mu)),p=Wei)
        S.append(np.random.normal(Mu[bin],Std[bin]))

    [H,s] = np.histogram(S,bins=s,density=False)
    s = 0.5*s[:-1] + 0.5*s[1:]



    x = np.linspace(-1,1,1000)
    Model = 0
    for k in range(0,len(Wei)):
        Model += Wei[k]*np.exp( - (x - Mu[k])**2/Std[k]**2/2 ) / np.sqrt(2*np.pi*Std[k]**2)

    #Model *= Ns*2/Nbins
    H = H*Nbins/2/Ns
    
    plt.close('all')
    figID = plt.figure(1,figsize=(12,6))
    ax1 = figID.add_subplot(111)
    

    ax1.fill_between(s,H,np.zeros(len(s)),color='b',step='mid')
    ax1.plot(x,Model,color='r',linewidth=3)
    #ax1.plot(S,np.zeros(Ns),marker='.',markersize=3,color='k',linestyle='none')

    ax1.set_ylim([0,2.1])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel(r'$\bf{s}_{+}$',fontsize=FS)
    ax1.set_ylabel(r'$\bf{P}(s_+|s,a)$',fontsize=FS,color='b')
    #ax1.text(s[700],H[700],r'$\bf{\hat P}_{\theta}(s_+|s,a)$',fontsize=FS,color='r',verticalalignment='bottom',horizontalalignment='left')

    plt.show(block=False)
    plt.pause(1)
    #sys.exit()
    
    figID.savefig('DataVsDist'+str(i)+'.eps', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'

