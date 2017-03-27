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


# Get plots of means along with training points
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gp_x[0][:,0],gp_x[0][:,1],gp_y[0],c='r')
ax.scatter(gp_X[0][:,0],gp_X[0][:,1],gp_mu[0],c = 'g')
ax.set_title('GP with N=%i' % N[0])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')

fig = plt.figure(2)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(kissgp_x[0][:,0],kissgp_x[0][:,1],kissgp_y[0],c='r')
ax.scatter(kissgp_X[0][:,0],kissgp_X[0][:,1],kissgp_mu[0],c='g')    
ax.set_title('KISSGP with N=M=%i' % N[0])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')

fig = plt.figure(3)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gp_x[1][:,0],gp_x[1][:,1],gp_y[1],c='r')
ax.scatter(gp_X[1][:,0],gp_X[1][:,1],gp_mu[1],c = 'g')
ax.set_title('GP with N=%i' % N[1])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')

fig = plt.figure(4)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(kissgp_x[1][:,0],kissgp_x[1][:,1],kissgp_y[1],c='r')
ax.scatter(kissgp_X[1][:,0],kissgp_X[1][:,1],kissgp_mu[1],c='g') 
ax.set_title('KISSGP with N=M=%i' % N[1])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')

fig = plt.figure(5)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gp_x[2][:,0],gp_x[2][:,1],gp_y[2],c='r')
ax.scatter(gp_X[2][:,0],gp_X[2][:,1],gp_mu[2],c = 'g')
ax.set_title('GP with N=%i' % N[2])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')

fig = plt.figure(6)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(kissgp_x[2][:,0],kissgp_x[2][:,1],kissgp_y[2],c='r')
ax.scatter(kissgp_X[2][:,0],kissgp_X[2][:,1],kissgp_mu[2],c='g') 
ax.set_title('KISSGP with N=M=%i' % N[2])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')

fig = plt.figure(7)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gp_x[3][:,0],gp_x[3][:,1],gp_y[3],c='r')
ax.scatter(gp_X[3][:,0],gp_X[3][:,1],gp_mu[3],c = 'g')
ax.set_title('GP with N=%i' % N[3])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')

fig = plt.figure(8)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(kissgp_x[3][:,0],kissgp_x[3][:,1],kissgp_y[3],c='r')
ax.scatter(kissgp_X[3][:,0],kissgp_X[3][:,1],kissgp_mu[3],c='g') 
ax.set_title('KISSGP with N=M=%i' % N[3])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')


# Get plot of means along with true target values
fig = plt.figure(9)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gp_X[0][:,0],gp_X[0][:,1],gp_Y[0],c='r')
ax.scatter(gp_X[0][:,0],gp_X[0][:,1],gp_mu[0],c = 'g')
ax.set_title('GP with N=%i' % N[0])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')

fig = plt.figure(10)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(kissgp_X[0][:,0],kissgp_X[0][:,1],kissgp_Y[0],c='r')
ax.scatter(kissgp_X[0][:,0],kissgp_X[0][:,1],kissgp_mu[0],c='g')    
ax.set_title('KISSGP with N=M=%i' % N[0])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')

fig = plt.figure(11)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gp_X[1][:,0],gp_X[1][:,1],gp_Y[1],c='r')
ax.scatter(gp_X[1][:,0],gp_X[1][:,1],gp_mu[1],c = 'g')
ax.set_title('GP with N=%i' % N[1])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')

fig = plt.figure(12)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(kissgp_X[1][:,0],kissgp_X[1][:,1],kissgp_Y[1],c='r')
ax.scatter(kissgp_X[1][:,0],kissgp_X[1][:,1],kissgp_mu[1],c='g') 
ax.set_title('KISSGP with N=M=%i' % N[1])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')

fig = plt.figure(13)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gp_X[2][:,0],gp_X[2][:,1],gp_Y[2],c='r')
ax.scatter(gp_X[2][:,0],gp_X[2][:,1],gp_mu[2],c = 'g')
ax.set_title('GP with N=%i' % N[2])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')

fig = plt.figure(14)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(kissgp_X[2][:,0],kissgp_X[2][:,1],kissgp_Y[2],c='r')
ax.scatter(kissgp_X[2][:,0],kissgp_X[2][:,1],kissgp_mu[2],c='g') 
ax.set_title('KISSGP with N=M=%i' % N[2])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')

fig = plt.figure(15)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gp_X[3][:,0],gp_X[3][:,1],gp_Y[3],c='r')
ax.scatter(gp_X[3][:,0],gp_X[3][:,1],gp_mu[3],c = 'g')
ax.set_title('GP with N=%i' % N[3])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')

fig = plt.figure(16)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(kissgp_X[3][:,0],kissgp_X[3][:,1],kissgp_Y[3],c='r')
ax.scatter(kissgp_X[3][:,0],kissgp_X[3][:,1],kissgp_mu[3],c='g') 
ax.set_title('KISSGP with N=M=%i' % N[3])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')


# Runtime plot
plt.figure(17)
plt.clf()
plt.plot(N,gp_time,label='Standard GP')
plt.plot(N,kissgp_time,label='KISS-GP')
plt.xlabel('Number of training points')
plt.ylabel('Seconds')
plt.title('Hyperparameter Optimization runtimes')
plt.legend()


plt.show()