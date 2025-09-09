
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
#import pickle
#from   datetime import timezone
#from   datetime import date, datetime, timedelta
#import pytz

plt.close('all')
    
########## DP ############################

Iterations = 200

N        = 50
gamma    = 0.99

#S        = list(np.linspace(   -1,   0, N+1))
#S       += [-S[0]+s for s in S[1:]]

S        = list(np.linspace(    0,  1, N+1))
A        = list(np.linspace(  -.25,  0, int(N/2)  ))
A       += [-A[0]+s for s in A[1:]]


alpha    = 1#0.9
beta     = 1

wmin     = -0.25
wmax     = +0.25
wspread  =  0.001

V        = np.zeros([len(S),1])
Pi       = np.zeros([len(S),1])

wgrid = np.linspace(wmin,wmax,1000)
Prob  = np.exp( - 0.5 * wgrid**2 / wspread**2  )
plt.figure(101)
plt.step(wgrid,Prob)
plt.show(block=False)
plt.pause(0.1)

#0 < alpha * s + beta * a + w < 1
#0 < alpha * smin + beta * amax + wmin

Smin  = (0-(beta * np.max(A) + wmin))/alpha

#alpha * smax + beta * amin + wmax < 1

Smax  = (1 - (beta * np.min(A) + wmax))/alpha

ResPi = []
ResV  = []
  
for iter in range(Iterations):
    print(iter)
    
    EV = np.zeros([len(S),len(A)])
    L  = np.zeros([len(S),len(A)])
    for i, s in enumerate(S):
        for j, a in enumerate(A):
            
            L[i,j]  = abs(s-1/2)#((s < 1/2)*(-s) + (s > 1/2)*s)
            L[i,j] += abs(a)#(a < 0)*a + 2*(a > 0)*a

            # Feasibility check
            smin = alpha * s + beta * a + wmin
            smax = alpha * s + beta * a + wmax
            
            # Build E[V(s+)]
            if smin < 0:
                EV[i,j] = 1e2 - 1e3*smin
            if smax > 1:
                EV[i,j] = 1e2 + 1e3*smax

            
            ProbSum = 0
            ev      = 0
            for k, splus in enumerate(S):
            
                # splus = alpha * s + beta * a + w
                w = splus - (alpha * s + beta * a)
                
                if w >= wmin and w <= wmax:
                    Prob = np.exp( - 0.5 * w**2 / wspread**2  )
                else:
                    Prob = 0
                    
                ProbSum += Prob
                ev      += Prob * V[k]
                
            if ProbSum > 0:
                EV[i,j] += ev / ProbSum
                    
    Q    = L + gamma * EV



    Vnew  = []
    Pinew = []
    for k in range(len(S)):
        Vnew.append(  np.min( Q[k,:] ))
        index = np.argmin( Q[k,:] )
        Pinew.append( A[index] )

    ResPi.append(np.linalg.norm(Pi-np.array(Pinew)))
    ResV.append( np.linalg.norm( np.array(V)-np.array(Vnew)))
    
    #sys.exit()
    V  = Vnew
    Pi = np.array(Pinew)

    """
    #####################################
    
    plt.close('all')

    X, Y = np.meshgrid(A,S)

    fig = plt.figure(11,figsize=[12,4])

    ax = fig.add_subplot(131, projection='3d')
    ax.plot_surface(X,Y,Q)
    ax.set_xlabel('a')
    ax.set_ylabel('s')
    ax.set_zlabel('Q')
    
    ax = fig.add_subplot(132)
    
    plt.plot(S,V,marker='.')
    ax.set_xlabel('s')
    ax.set_ylabel('V')
    
    ax = fig.add_subplot(133)
  
    plt.plot(S,Pi,marker='.')
    ax.set_xlabel('s')
    ax.set_ylabel('a')
    
    plt.show(block=False)
    plt.pause(0.1)
    
    #sys.exit()
    
    #####################################
    """



# Some plotting on DP
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(S,V,marker='.')
plt.subplot(1,2,2)
plt.plot(S,Pi,marker='.')
plt.show(block=False)
plt.pause(0.1)

"""
X, Y = np.meshgrid(A,S)

fig = plt.figure(11,figsize=[12,4])

ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Q)
ax.set_xlabel('a')
ax.set_ylabel('s')
ax.set_zlabel('Q')
ax.set_zlim([0,5])
plt.show(block=False)
plt.pause(0.1)
"""

# Some plotting on DP
plt.figure(2)
#plt.subplot(1,2,1)
plt.semilogy(ResV)
#plt.subplot(1,2,2)
#plt.semilogy(ResPi)
plt.show(block=False)
plt.pause(0.1)

print('Save MPC')
Results = {'V' : V, 'Pi' : Pi, 'Q' : Q, 'S' : S, 'A' : A, 'Smin' : Smin, 'Smax' : Smax}
f = open('MPC.pkl',"wb")
pickle.dump(Results,f, protocol=2)
f.close()

#sys.exit()
### Simulate
wspread  =  0.05
wgrid = np.linspace(wmin,wmax,1000)
Prob  = np.exp( - 0.5 * wgrid**2 / wspread**2  )

# Make delta grid
Nsim      = 100
SimLength = 500

Sim = {'S' : [], 'A' : [], 'W' : []}
SOCA = np.array(S)
for sim in range(Nsim):

    for key in ['A', 'W']:
        Sim[key].append([])
    Sim['S'].append([0.5])
    
    for hour in range(SimLength):
        indexgrid = np.argmin(np.abs(SOCA - Sim['S'][-1][-1]))

        a = Pi[indexgrid]

        w = random.choices( wgrid, Prob )[0]
        
        splus = alpha * Sim['S'][-1][-1] + beta * a + w
        
        Sim['W'][-1].append(w)
        
        Sim['A'][-1].append( a  )
        Sim['S'][-1].append( splus  )


TimeDisp = [k / 24 for k in range(SimLength)]


fig = plt.figure(21, figsize=[18,6])
plt.subplot(1,3,1)
plt.step(wgrid,Prob,color='k',linewidth=3)
#plt.title('Delta dist.',fontsize=12)
plt.xlabel('W',fontsize=15)
plt.ylabel('Probability',fontsize=15)
#plt.yticks([])

plt.subplot(1,3,2)
plt.plot(S,len(S)*[0],color=[.8,.8,.8],linewidth=3)
plt.plot(S,V,color='k',linewidth=3)

#plt.title('V',fontsize=12)
#plt.ylim([-1,1])
plt.xlabel('S',fontsize=15)
plt.ylabel('Value',fontsize=15)
plt.yticks([])
plt.subplot(1,3,3)
plt.step(S,Pi,color='k',linewidth=3)
#plt.title('Pi',fontsize=12)
plt.xlabel('S',fontsize=15)
plt.ylabel('A',fontsize=15)

plt.show(block=False)

#fig.savefig('DP1_EndResults.eps', bbox_inches='tight')


fig = plt.figure(22, figsize=[12,5])
plt.subplot(1,2,1)
plt.title('MDP')

ax  = plt.gca()
#ax2 = plt.twinx()

ax.step([TimeDisp[0],TimeDisp[-1]],[+1,+1],linestyle='-',color='k')
ax.step([TimeDisp[0],TimeDisp[-1]],[-1,-1],linestyle='-',color='k')

for sim in range(10):
    ax.step(TimeDisp,Sim['S'][sim][:-1],linewidth=1,color='k',linestyle='-')

#X, Y = np.meshgrid(Bins['SOC'],TimeDisp)
#ax.pcolor(Y,X,Stats['Hist']['SOC'])



plt.ylabel('S',fontsize=15)
plt.xlabel('time',fontsize=15)


plt.subplot(1,2,2)

ax  = plt.gca()
for sim in range(10):
    plt.step(TimeDisp,np.array(Sim['A'][sim]),'k',linestyle='-')
    
#X, Y = np.meshgrid(Bins['Pgrid'],TimeDisp)
#ax.pcolor([Y,X],Stats['Hist']['Pgrid'])
    
#mean = np.mean(np.array(Sim['Delta']),axis=0)
#plt.step(TimeDisp,mean,color='g',label='Delta',linewidth=3)
#ax.step([TimeDisp[0],TimeDisp[-1]],[+Pgridmax * Batsize,+Pgridmax * Batsize],linestyle='-',color='k')
#ax.step([TimeDisp[0],TimeDisp[-1]],[-Pgridmax * Batsize,-Pgridmax * Batsize],linestyle='-',color='k')


#mean = np.mean(np.array(Sim['Pbat']),axis=0)
#plt.step(TimeDisp,mean,color='b',label='Pbat',linewidth=3)

plt.ylabel('A',fontsize=15)
plt.xlabel('time',fontsize=15)

plt.show(block=False)

#fig.savefig('DP1_SimResults.eps', bbox_inches='tight')


