import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
FS = 40

Mu  = [-.06,.04]
Std = [.02,.04]
Wei = [.2,.8]

    
Ns    = int(1e7)
Nbins = 1000

s = np.linspace(-1,1,Nbins)

x = np.linspace(-1,1,1000)
Model = 0
for k in range(0,len(Wei)):
    Model += Wei[k]*np.exp( - (x - Mu[k])**2/Std[k]**2/2 ) / np.sqrt(2*np.pi*Std[k]**2)

S = []
for k in range(0,Ns):
    draw = True
    if draw:
        bin = np.random.choice(range(0,len(Mu)),p=Wei)
        sample = np.random.normal(Mu[bin],Std[bin])
        if sample < 1 and sample > -1:
            draw = False
    
    S.append(sample)

[H,s] = np.histogram(S,bins=s,density=False)
s = 0.5*s[:-1] + 0.5*s[1:]
H = H*Nbins/2/Ns

Ipos = np.where(H>0)
H = H[Ipos]
s = s[Ipos]

top = np.max(H)

def Support(b,top,type, CO):
    
    if type == 'low':
        sign = +1
    else:
        sign = -1
        
    ax1.plot([b,b],[-top/10,top/10],color=CO,linewidth=3)
    ax1.plot([b,b+sign*0.05],[-top/10,-top/10],color=CO,linewidth=3)
    ax1.plot([b,b+sign*0.05],[+top/10,+top/10],color=CO,linewidth=3)
    
    return

plt.close('all')
figID = plt.figure(1,figsize=(12,6))
ax1 = figID.add_subplot(111)




ax1.fill_between(s,H,np.zeros(len(s)),color='b',step='mid')


ax1.set_xlim([-1.2,1.2])
ax1.set_xticks([])
ax1.set_yticks([])

Support(np.min(S),top,'low',  'b')
Support(np.max(S),top,'high', 'b')

ax1.set_xlabel(r'$\bf s_{+}$',fontsize=25)
ax1.text(0.5,0.7*top,r'$10^{'+str(int(np.log10(Ns)))+'}$ samples',fontsize=25)

plt.show(block=False)



figID.savefig('Support0.pdf', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'

ax1.plot(x,Model,color='r',linewidth=3)

Support(-1,top,'low',  'r')
Support(+1,top,'high', 'r')


plt.show(block=False)

figID.savefig('Support1.pdf', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'



