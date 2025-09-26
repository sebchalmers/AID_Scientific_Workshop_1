import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#import scipy as sc

#plt.rcParams['text.usetex'] = True
FS = 40

STD = np.linspace(0.6,3.5,5)

for kstd, stdfactor in enumerate(list(STD)):

    Std = [stdfactor*np.eye(2),(2*stdfactor/3)*np.eye(2)]

    Mu  = [np.array([-.55,-.55]),np.array([0.55,.55])]

    Wei = [.525,.475]

    Bounds = 1.5

    x = np.linspace(-Bounds,Bounds,200)
    y = np.linspace(-Bounds,Bounds,199)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    # Distribution
    ModelPlot = 0
    for k in range(0,len(Wei)):
        Expon = 0.5*Std[k][0,0]*(xv - Mu[k][0])**2 + 0.5*Std[k][1,1]*(yv - Mu[k][1])**2 + Std[k][0,1]*(yv - Mu[k][1])*(xv - Mu[k][0])
        ModelPlot += Wei[k]*np.exp( - Expon )

    ModelPlot *= 1/np.max(ModelPlot)

    MLLevel = 0.7

    plt.close('all')
    figID = plt.figure(1,figsize=(12,6))
    ax1 = figID.add_subplot(111)

    ax1.pcolormesh(xv, yv, ModelPlot, cmap='Blues')
    cs = ax1.contour(xv, yv, ModelPlot,linestyles='solid',levels = [MLLevel],colors='k')

    TextCoo = np.mean(np.array(Mu),axis=0)
    ax1.text(TextCoo[0],TextCoo[1],'AI Model',fontsize=20,verticalalignment='center',horizontalalignment='center',rotation=45,color=[1,1,1])
    ax1.text(-0*Bounds,0.95*Bounds,'Knowledge-driven Model',fontsize=20,verticalalignment='top',horizontalalignment='center',rotation=0,color=[0,.6,0])



    #ax1.clabel(cs, cs.levels, fmt='0.7', fontsize=20)
    #cs.collections[0].get_paths()[0]

    ax1.set_aspect('equal', 'box')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('State',fontsize=20)
    ax1.set_ylabel('Action',fontsize=20)

    plt.show(block=False)
    plt.pause(0.1)

    figID.savefig('Hybrid'+str(kstd)+'.jpeg', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None)

