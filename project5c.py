import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
import scipy.sparse as sps
import time
import random
from multiprocessing import Pool,cpu_count
import sys

# this is a program for solving the 1D diffuion equation.
# there are three different schemes: explicit, implicit and crank nicolson


# to make the tridiagonal matrix of the implicit solver and the crank nicolson solver
def makeA(Nx,alpha): 
	A = np.identity(Nx)*(1+2*alpha)
	for i in range(Nx-1):
		A[i,i+1] = -alpha
		A[i+1,i] = -alpha

	return(A)

# to do row reduction on the matrix equation of the implicit solver and the crank nicolson
# in place modification of A and u
def rowreduction(A,u): 
	for i in range(A.shape[0]-1):                                             # Gaussian elimination
		coeff = A[i+1,i]/A[i,i]
		A[i+1,i+1] = A[i+1,i+1]-A[i,i+1]*coeff
		u[i+1] = u[i+1]-u[i]*coeff
	

# to find the solution of the matrix equation of the implicit solver and the crank nicolson
def backsubstitution(A, u, alpha): 
	Nx = A.shape[0]
	solution = u.copy()
	solution[Nx] = 1                                      					  # boundary condition
	for i in range(1,Nx+1):                              					  # backward substitution
		k = Nx-i
		solution[k] = (u[k]+alpha*solution[k+1])/A[k,k]

	return(solution)

# to make one euler forward step. used in the explicit solver and the crank nicolson
def forwardstep(u, alpha): 
	unew = u.copy()
	unew[0] = (1-2*alpha)*u[0] + alpha*(u[1])                                 # the boundary u_0 = 0 is not included. the vector u starts at u_1. This is to make the solutions of same length for all solvers
	unew[1:Nx] = (1-2*alpha)*u[1:Nx] + alpha*(u[2:Nx+1]+u[0:Nx-1])
	
	return(unew)

# to initialize the variables which are common for all the solvers
def initialize(Nt,tfinal,Nx,frequency): 
	alpha = Nx**2*tfinal/Nt           	                                      # alpha = dt/dx**2 =(tfinal/Nt)/(1/Nx)**2
	u = np.zeros(Nx+1)
	u[Nx] = 1
	U = np.zeros((Nx+2,Nt//frequency))
	counter = 0
	
	return(alpha,u,U,counter)

# solves the 1D diffusion equation
def explicitsolver(Nt,tfinal,Nx,frequency): 
	alpha,u,U,counter = initialize(Nt,tfinal,Nx,frequency)

	for j in range(Nt):
		u = forwardstep(u, alpha)

		if j%frequency == 0:                                                  # to store every Nt/frequency solution
			U[1:,counter] = u
			counter += 1

	return(U)

# solves the 1D diffusion equation
def implictsolver(Nt,tfinal,Nx,frequency): 
	
	alpha,u,U,counter = initialize(Nt,tfinal,Nx,frequency)
	B = makeA(Nx,alpha)

	for t in range(Nt):

		A = B.copy()
		rowreduction(A, u)

		u = backsubstitution(A, u, alpha)

		if t%frequency == 0: 				                                  # to store every frequency solution
			U[1:,counter] = u
			counter += 1
	return(U)

# solves the 1D diffusion equation
def cranknicolson(Nt,tfinal,Nx,frequency): 

	alpha,u,U,counter = initialize(Nt,tfinal,Nx,frequency)
	beta = alpha/2                                                            # for crank nicolson, alpha is half of alpha in the other solvers
	B = makeA(Nx,beta)

	for t in range(Nt):
		u = forwardstep(u, beta)

		A = B.copy()
		rowreduction(A, u)

		u = backsubstitution(A, u, beta)

		if t%frequency == 0:                                                  # to store every frequency solution
			U[1:,counter] = u
			counter += 1
	return(U)

# makes the analytical solution, which is a sum of fourier coefficients
def exactsolution(x,t): 

	u    = np.zeros(len(x))	
	k = 1
	norm = 1
		
	while norm> 1e-15:                                                        # here you can set the precision of the analytical solution
		
		kpi  = np.pi *k
		uold = u.copy()

		if k%2 == 0:
			u += 2*np.exp(-(kpi)**2 *t)*np.sin(kpi*x)/(kpi)
		else:
			u += -2*np.exp(-(kpi)**2 *t)*np.sin(kpi*x)/(kpi)
		k +=1
		norm = np.linalg.norm(u-uold)
		
	u += x
	return(u)

# to plot the solutions. Inputs are solution vectors, and the time at which the solution is calculated
def plotsolutions(ue,ui,uc, t): 

	Nx = len(ue)
	x = np.linspace(0,1,num = 1000)
	exact = exactsolution(x,t)
	axis = np.linspace(0,1,num = Nx)

	plt.plot( x, exact, label = 'exact')
	plt.plot(axis,ue, label = 'explicit')
	plt.plot(axis,ui, label = 'implict')
	plt.plot(axis,uc, label = 'cranknicolson')
	plt.legend()
	plt.xlabel('x')
	plt.ylabel('u')
	plt.title('u(x,t = %.3f) '%t )
	#plt.savefig('Nt150Nx10t02', dpi = 225)
	plt.show()

def calculateerror(v,u): # to calculate the relative error. v is the exact solution, u is the numerical solution
	error = np.linalg.norm(v-u)/np.linalg.norm(v)
	return(error)

# set the variables here ------------------------------------------------------
Nt = 2000
tfinal = 1.
frequency = 10 	                                                              # Nt/frequency  is the number of solutions
n = 4				                                                          # solution number n
Nx = 10
t = n*frequency/Nt 	                                                          # time 
axis = np.linspace(0,1,num = Nx+2)

# make the solutions-----------------------------------------------------------
exact = exactsolution(axis,t)
Ue = explicitsolver(Nt,tfinal,Nx,frequency)
Ui = implictsolver(Nt,tfinal,Nx,frequency)
Uc = cranknicolson(Nt,tfinal,Nx,frequency)

#to calculate the error and to check the benchmarks:---------------------------
print('errorE',calculateerror(exact,Ue[:,n]))
print('errorI',calculateerror(exact,Ui[:,n]))
print('errorC',calculateerror(exact,Uc[:,n]))

print('Ue', Ue[:,n] )
print('Ui', Ui[:,n] )
print('Uc', Uc[:,n] )

plotsolutions(Ue[:,n], Ui[:,n], Uc[:,n],t)



