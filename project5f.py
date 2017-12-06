import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
import scipy.sparse as sps
import time
import random
from multiprocessing import Pool,cpu_count

import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


Nt = 10000
tfinal = 1.
Nx = 20
frequency = 100
n = 2
t = n*frequency/Nt 	# time 


def explicitsolver(Nt,tfinal,Nx,frequency,n):
	
	U = np.zeros((Nx+1,Nx+1))

	alpha = Nx**2*tfinal/Nt
	axis = np.linspace(0,1,num = Nx+1)
	
# initial conditions: u(x,y,0)=sin(pi*x)sin(pi*y) boundaries: u(0,y,t)=u(1,y,t)=u(x,0,t)=u(x,1,t)
	x0 = np.sin(axis *np.pi)
	for i in range(Nx):
		for j in range(Nx):
			U[i,j]=x0[i]*x0[j]
	
# initial conditions: u(x,y,0)=0 for 0<x<1 and 0<y<1 boundaries: u(1,y,t)=u(x,1,t)=1
	#U[Nx,:] = np.ones(Nx+1)
	#U[:,Nx] = np.ones(Nx+1)	
	
	Unew = U.copy()

	for l in range(Nt):

		for i in range(1,Nx):
			for j in range(1,Nx):

				Unew[i,j] = alpha*(U[i+1,j]+U[i-1,j]+U[i,j+1]+U[i,j-1]-4*U[i,j])+U[i,j]
		U = Unew.copy()

		if l%frequency == 0: # plots every frequency solution
			plt.plot(axis, U[:,10])

		if l == n*frequency:
			Ureturn = U.copy()
			
	plt.xlabel('x')
	plt.ylabel('u(x,0.5,t)')
	plt.title('A cross section of the plane')
	plt.show()
	return(Ureturn,axis)


U,axis = explicitsolver(Nt,tfinal,Nx,frequency,n)


X, Y = np.meshgrid(axis, axis)
plt.figure()
CS = plt.contour(X, Y, U)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Contour plot of u(x,y,%.3f)' %t)
plt.xlabel('x')
plt.ylabel('y')
#plt.savefig("diffusion", dpi=225)
plt.show()


def implicitsolver(Nt,tfinal,Nx,frequency):
		#initial conditions
	U = np.zeros((Nx+1,Nx+1))
	U[Nx,:] = np.ones(Nx+1)
	U[:,Nx] = np.ones(Nx+1)
	Unew = np.zeros((Nx+1,Nx+1))
	alpha = Nx**2*tfinal/Nt
	beta = 1/(1+4*alpha)
	counter = 0
	for l in range(Nt):

		Unew[Nx,:] = np.ones(Nx+1)
		Unew[:,Nx] = np.ones(Nx+1)

		for i in range(1,Nx):
			for j in range(1,Nx):

				Unew[i,j] = beta*(alpha*(U[i+1,j]+U[i-1,j]+U[i,j+1]+U[i,j-1])+U[i,j])
				U = Unew

		Unew[0,:] = np.zeros(Nx+1)
		Unew[:,0] = np.zeros(Nx+1)

		if l%frequency == 0: # to store every 1/frequency solution
			x = axis
			y = axis
			X, Y = np.meshgrid(x, y)
			plt.figure()
			CS = plt.contour(X, Y, U)
			plt.clabel(CS, inline=1, fontsize=10)
			plt.title('time = %d' %l)
			plt.savefig("diffusion%d" %counter, dpi=225)
			#plt.show()

			counter += 1


#implicitsolver(40,tfinal,Nx,5)