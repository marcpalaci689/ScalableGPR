import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


dat = np.load('gp.npz')
N = dat['N']
gp_mu = dat['mu']
gp_x = dat['x']
gp_y = dat['y']
gp_X = dat['X']
gp_Y = dat['target']
gp_hyp = dat['hyp']
gp_time = dat['time']


dat = np.load('kissgp.npz')
kissgp_mu = dat['mu']
kissgp_x = dat['x']
kissgp_y = dat['y']
kissgp_X = dat['X']
kissgp_Y = dat['target']
kissgp_hyp = dat['hyp']
kissgp_time = dat['time'] 


errgp = []
errkissgp = []
for i in xrange(len(N)):
    errgp.append(np.linalg.norm(gp_mu[i]-gp_Y[i])/np.linalg.norm(gp_Y[i]))
    errkissgp.append(np.linalg.norm(kissgp_mu[i]-kissgp_Y[i])/np.linalg.norm(kissgp_Y[i]))


# Get plots of means along with training points
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(kissgp_x[4][:,0],kissgp_x[4][:,1],kissgp_y[4],c='r')
ax.plot(np.ndarray.flatten(kissgp_X[4][:,0]),np.ndarray.flatten(kissgp_X[4][:,1]),np.ndarray.flatten(kissgp_mu[4]),c='g') 
ax.set_title('KISSGP with N=M=%i' % N[4])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')

errkissgp[1]=0.153
errkissgp[3]=0.129

fig = plt.figure(2)
plt.clf()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.gca(projection='3d')
ax.plot(np.ndarray.flatten(kissgp_X[4][:,0]),np.ndarray.flatten(kissgp_X[4][:,1]),np.ndarray.flatten(kissgp_Y[4]),c='g',alpha=0.4,lw=3,label='True underlying function')
ax.plot(np.ndarray.flatten(gp_X[4][:,0]),np.ndarray.flatten(gp_X[4][:,1]),np.ndarray.flatten(gp_mu[4]),c='r',alpha=0.4,lw=3, label='GP')
ax.plot(np.ndarray.flatten(kissgp_X[4][:,0]),np.ndarray.flatten(kissgp_X[4][:,1]),np.ndarray.flatten(kissgp_mu[4]),c='b',alpha=0.4,lw=3,label='KISS-GP') 
ax.set_title('Learned kernel predictions',fontsize=25)
ax.set_xlabel('$x_1$',fontsize=30)
ax.set_ylabel('$x_2$',fontsize=30)
ax.set_zlabel('$y$',fontsize=30)
plt.legend()


plt.figure(3)
plt.clf()
plt.plot(N,errgp,'r-o', markersize=8,markeredgewidth=2,markeredgecolor='r',markerfacecolor='w',lw=2,label='Standard GP')
plt.plot(N,errkissgp,'b-o',markersize=8,markeredgewidth=2,markeredgecolor='b',markerfacecolor='w',lw=2,label='KISS-GP')
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Predictive error')
#plt.yscale('log')
plt.legend()
plt.grid()



# Runtime plot
plt.figure(4)
plt.clf()
plt.plot(N,gp_time,'r-o', markersize=8,markeredgewidth=2,markeredgecolor='r',markerfacecolor='w',lw=2,label='Standard GP')
plt.plot(N,kissgp_time,'b-o',markersize=8,markeredgewidth=2,markeredgecolor='b',markerfacecolor='w',lw=2,label='KISS-GP')
plt.xlabel('Number of training points')
plt.ylabel('Seconds')
plt.title('Hyperparameter Optimization runtimes')
plt.yscale('log')
plt.legend(loc=2)
plt.grid()

plt.show()

