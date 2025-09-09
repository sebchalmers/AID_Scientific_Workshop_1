import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
FS = 40

Nbins = 1000
Ns = 1000000
S  = []
Mu  = [-.3,.2]
Var = [.1,.2]
Wei = [.2,.8]

for k in range(0,Ns):
    bin = np.random.choice(range(0,len(Mu)),p=Wei)
    S.append(np.random.normal(Mu[bin],Var[bin]))

[H,x] = np.histogram(S,bins=1000,density=True)
MuTot  = np.mean(S)
VarTot = np.var(S)

x      = np.linspace(np.min(S),np.max(S),Nbins)
Model  = np.exp( - (x - MuTot)**2/VarTot/2 ) / np.sqrt(2*np.pi*VarTot)

plt.close('all')
figID = plt.figure(1,figsize=(12,6))
ax1 = figID.add_subplot(111)
#ax1.hist(S,bins=Nbins,density=True)
ax1.fill_between(x,H,np.zeros(Nbins),color='b')
ax1.plot(x,Model,color='r',linewidth=3)


ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel(r'$\bf{s}_{+}$',fontsize=FS)
ax1.set_ylabel(r'$\bf{P}(s_+|s,a)$',fontsize=FS,color='b')
ax1.text(x[700],H[700],r'$\bf{\hat P}_{\theta}(s_+|s,a)$',fontsize=FS,color='r',verticalalignment='bottom',horizontalalignment='left')

plt.show(block=False)

figID.savefig('Model.eps', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'

