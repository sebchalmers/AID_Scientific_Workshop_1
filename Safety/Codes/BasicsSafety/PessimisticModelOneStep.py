import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm


plt.rcParams['text.usetex'] = True



Nstate = 2
N  = 40
FS = 15

std = 0.01

angle = 8*np.pi/180
A = 0.99*np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
B1 = np.random.normal(0,1,[2,2])
B2 = np.random.normal(0,1,[2,2])

NMC     = 5000
NMCplot = 10

Time = [k for k in range(N)]
End = 5

def MakeBox(s, CO, LW = 3):
    
    Box = np.array([[np.inf,-np.inf],[np.inf,-np.inf]])
    for n in range(0,NMC):
        Box[0,0] = np.min([ Box[0,0], np.min(s[0,n])  ])
        Box[0,1] = np.max([ Box[0,1], np.max(s[0,n])  ])
        Box[1,0] = np.min([ Box[1,0], np.min(s[1,n])  ])
        Box[1,1] = np.max([ Box[1,1], np.max(s[1,n])  ])

    ax1.plot([Box[0,0],Box[0,1]],[Box[1,0],Box[1,0]],color=CO,linewidth=LW)
    ax1.plot([Box[0,1],Box[0,1]],[Box[1,0],Box[1,1]],color=CO,linewidth=LW)
    ax1.plot([Box[0,1],Box[0,0]],[Box[1,1],Box[1,1]],color=CO,linewidth=LW)
    ax1.plot([Box[0,0],Box[0,0]],[Box[1,1],Box[1,0]],color=CO,linewidth=LW)

    
    return Box

bounds = 0.1

s  = np.array([1,1])
splus = np.array([[],[]])
for n in range(0,NMC):
    dist = np.random.choice([0,1],p=[.5,.5])
    draw = True
    while draw:
        Noise  = dist*(np.matmul(B1,np.random.normal([0,0],std,2)) + np.array([0.025,-0.025]))
        Noise += (1-dist)*np.matmul(B2,np.random.normal(0,std,2))
        if Noise[0] <= bounds and Noise[0] >= -bounds and Noise[1] <= bounds and Noise[1] >= -bounds:
            draw = False
            
    sp   = np.matmul(A,s) + Noise
    splus = np.concatenate( (splus,sp.reshape(2,1)), axis=1)


plt.close('all')

figID = plt.figure(1,figsize=(12,6))

ax1 = figID.add_subplot(111)

ax1.plot(splus[0,:],splus[1,:],color=[.7,.7,.7],linestyle='none',marker='.',markersize=2)
    
    
"""
h, x, y = np.histogram2d(splus[0,:],splus[1,:], density=False, bins = 60)
xc = (x[:-1] + x[1:]) / 2
yc = (y[:-1] + y[1:]) / 2

h = np.ma.masked_where(h < 1e-1, h)
cm = plt.colormaps.get_cmap('Blues')
cm.set_bad('white')
#im = NonUniformImage(ax1, interpolation='bilinear', cmap=cm)
#im.set_data(xcenters, ycenters, h)
#ax1.add_image(im)
#ax1.imshow(xcenters, ycenters, h)
ax1.pcolormesh(xc, yc, h, cmap=cm,shading='gouraud')
"""

ax1.text(s[0],s[1],r'$\bf s, a$',verticalalignment='bottom',horizontalalignment='left',fontsize=20,color='c')
for n in range(0,NMCplot):
    ax1.plot([s[0],splus[0,n]],[s[1],splus[1,n]],color='b',linewidth=1,marker='.')
    ax1.text(splus[0,n],splus[1,n],r'$\bf s_+$',verticalalignment='bottom',horizontalalignment='left',fontsize=15,color='b')
ax1.plot(s[0],s[1],color='c',linewidth=2,marker='o',markerfacecolor='c')





ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_axis_off()
ax1.set_aspect('equal', 'box')



plt.show(block=False)

figID.savefig('PessimisticOneStep0.pdf', bbox_inches='tight')#,dpi='figure', format=None,metadata=None, pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #,


Box = MakeBox(splus,'r',LW = 2)


plt.show(block=False)

figID.savefig('PessimisticOneStep1.pdf', bbox_inches='tight')#,dpi='figure', format=None,metadata=None, pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None, bbox_inches='tight') #,
