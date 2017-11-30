import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
import scipy.sparse as sps
import time
import random
from multiprocessing import Pool,cpu_count


def makeA(Nx,alpha): # to make the tridiagonal matrix
	A = np.identity(Nx)*(1+2*alpha)
	for i in range(Nx-1):
		A[i,i+1] = -alpha
		A[i+1,i] = -alpha

	return(A)

def rowreduction(A,v):
	for i in range(A.shape[0]-1): # Gaussian elimination
		coeff=A[i+1,i]/A[i,i]
		A[i+1,i] = 0
		A[i+1,i+1]=A[i+1,i+1]-A[i,i+1]*coeff
		v[i+1]=v[i+1]-v[i]*coeff
	# return(A,v)

def backsubstitution(A,v,solution,alpha):
	Nx = A.shape[0]
	solution[Nx] = 1 # boundary condition
	for i in range(1,Nx+1): #backward substitution
		k = Nx-i
		solution[k] = (v[k]+alpha*solution[k+1])/A[k,k]

	return(solution)

def forwardstep(u,unew,alpha):
	unew[0] = (1-2*alpha)*u[0] + alpha*(u[1]) # the boundary u_0 = 1 is not included in the vecror u
	unew[1:Nx] = (1-2*alpha)*u[1:Nx] + alpha*(u[2:Nx+1]+u[0:Nx-1])

	#return(unew)
def explisitsolver(Nt,tfinal,Nx,frequency):
	alpha = Nx**2*tfinal/Nt # alpha = dt/dx**2 =(tfinal/Nt)/(1/Nx)**2
	u = np.zeros(Nx+1)
	u[Nx] = 1
	unew = np.zeros(Nx+1)
	unew[Nx] = 1
	U = np.zeros((Nx+2,Nt//frequency))
	counter = 0
	for j in range(Nt):
		forwardstep(u,unew,alpha)
	
		u = unew.copy()
		u[Nx] = 1
		if j%frequency == 0:
			U[1:,counter] = u

			counter += 1
	return(U)

def implistsolver(Nt,tfinal,Nx,frequency):
	alpha = Nx**2*tfinal/Nt # alpha = dt/dx**2 =(tfinal/Nt)/(1/Nx)**2
	B = makeA(Nx,alpha)
	U = np.zeros((Nx+2,Nt//frequency)) # matrix to store solutions in
	counter = 0
	v = np.zeros(Nx+1)
	solution = np.zeros(Nx+1)

	for t in range(Nt):
		v[Nx] = 1 # boundary condition
		A = B.copy()

		rowreduction(A,v)

		v = backsubstitution(A,v,solution,alpha)

		if t%frequency == 0: # to store every 1/frequency solution
			U[1:,counter] = v
			counter += 1
	return(U)

def cranknicolson(Nt,tfinal,Nx,frequency):
	beta = Nx**2*tfinal/(2*Nt)
	u = np.zeros(Nx+1)
	solution = np.zeros(Nx+1)
	u[Nx] = 1
	unew = np.zeros(Nx+1)
	unew[Nx] = 1
	U = np.zeros((Nx+2,Nt//frequency))
	B = makeA(Nx,beta)
	counter = 0
	for t in range(Nt):
		forwardstep(u,unew,beta)
		A = B.copy()
		#u[Nx] = 1
		rowreduction(A,unew)

		u = backsubstitution(A,unew,solution,beta)

		if t%frequency == 0: # to store every 1/frequency solution
			U[1:,counter] = u
			counter += 1
	return(U)

Nt = 1000
tfinal = 1.
Nx = 10
frequency = 10
axis = np.linspace(0,1,num = Nx+2)

U = explisitsolver(Nt,tfinal,Nx,frequency)
#U = implistsolver(Nt,tfinal,Nx,frequency)
#U = cranknicolson(Nt,tfinal,Nx,frequency)

plt.plot(axis,U[:,2],axis,U[:,10], axis,U[:,20], axis,U[:,50])
plt.show()
