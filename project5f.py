import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
import scipy.sparse as sps
import time
import random
from multiprocessing import Pool,cpu_count
import sys

import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# The explicitsolver2D solves the 2D diffusion equation. 
# initial conditions and boundaries are set manually inside the function
# you choose the number of time step, and number og grid points
# the function returns solution number n*frequency and a vector representing the x- and y axis

def explicitsolver2D(Nt,tfinal,Nx,frequency,n):
	
	U = np.zeros((Nx+1,Nx+1))
	alpha = Nx**2*tfinal/Nt
	axis = np.linspace(0,1,num = Nx+1)
	
	
# initial conditions: u(x,y,0)=sin(pi*x)sin(pi*y) boundaries: u(0,y,t)=u(1,y,t)=u(x,0,t)=u(x,1,t)
	x0 = np.sin(axis *np.pi)
	np.outer(x0, x0, U)
	
# initial conditions: u(x,y,0)=0 for 0<x<1 and 0<y<1 boundaries: u(1,y,t)=u(x,1,t)=1
	#U[Nx,:] = np.ones(Nx+1)
	#U[:,Nx] = np.ones(Nx+1)	

	for l in range(Nt):

		U[1:Nx,1:Nx] = alpha*(U[2:Nx+1,1:Nx] + U[0:Nx-1,1:Nx] + U[1:Nx,2:Nx+1] + U[1:Nx,0:Nx-1] - 4*U[1:Nx,1:Nx]) + U[1:Nx,1:Nx]

		if l == n*frequency:                                                  # the function returns only one solution, solution n*frequency
			Ureturn = U.copy()

 #to make the cross section plot:
		if l%frequency == 0:                                                  # plots every frequency solution
			ax = plt.plot(axis, U[:,10])
			plt.plot(0.5,np.exp(-2*np.pi**2*l/Nt),'bo')

	plt.xlabel('x')
	plt.ylabel('u(x,0.5,t)')
	plt.title('A cross section of the plane')
	plt.savefig('crosssection' , dpi = 225)
	plt.show()

	return(Ureturn,axis)

# set the variables here ------------------------------------------------------
Nt = 100000
tfinal = 1.
Nx = 20
frequency = 1000
n = 4
t = n*frequency/Nt 	# time 

# make the solution -----------------------------------------------------------
U,axis = explicitsolver2D(Nt,tfinal,Nx,frequency,n)

# to calculate the error and to check the benchmarks: -------------------------
exact = np.sin(axis *np.pi)*np.exp(-2*np.pi**2*t)                             # u_exact(x,0.5,t)
print('t',t,'Nx',Nx)
print('exact',exact)
print('numerical',U[:,Nx//2])
error = np.linalg.norm(exact -U[:,Nx//2] )/np.linalg.norm(exact)              # relative error
print('error',error)


# to make contour plots of the solution ---------------------------------------
X, Y = np.meshgrid(axis, axis)
plt.figure()
CS = plt.contour(X, Y, U,np.linspace(0.1,0.9,9))
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Contour plot of u(x,y,%.3f)' %t)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("contour40", dpi=225)
plt.show()


# this is only a sketch of the implicit solver. it does not work
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
		for i in range(10):
			U[1:Nx,1:Nx] = beta*(U[2:Nx+1,1:Nx] + U[0:Nx-1,1:Nx] + U[1:Nx,2:Nx+1] + U[1:Nx,0:Nx-1] ) + U[1:Nx,1:Nx]

		if l%frequency == 0: # to store every 1/frequency solution
			x = axis
			y = axis
			X, Y = np.meshgrid(x, y)
			plt.figure()
			CS = plt.contour(X, Y, U)
			plt.clabel(CS, inline=1, fontsize=10)
			plt.title('time = %d' %l)
			plt.savefig("diffusion%d" %counter, dpi=225)
			plt.show()

			counter += 1


#implicitsolver(40,tfinal,Nx,5)