import numpy as np
import numba
import matplotlib.pyplot as plt
from thewalrus import charpoly
from thewalrus.charpoly import powertrace
"""
def powertrace_eigs(A,n):
    A += A.T
    powertrace = []  
    A = np.linalg.matrix_power(A,n)
    eigs = np.linalg.eigvals(A)
    powertrace.append(np.sum(eigs))
        
    return powertrace
  
A = np.random.rand(6,6)
A +=A.T

print(powertrace_eigs(A,2))
print(powertrace(A,3))

%timeit powertrace_eigs(A,2)
print()
%timeit powertrace(A,3)

eigs = []
powertrace = []
nmax = list(range(2,10,1))

for a in nmax:
    B = np.random.rand(5,5)
    B += B.T
    
    
    time_eigs = %timeit -o powertrace_eigs(B,a)
    time_powertrace = %timeit -o powertrace(B,a)
    eigs.append(time_eigs.average)
    powertrace.append(time_powertrace.average)
    

plt.semilogy(nmax,eigs,nmax,powertrace)
plt.legend(['Powertrace with eigenvalues ','Powertrace from thewalrus'])
plt.xlabel('Matrix size')
plt.ylabel('Time of computation')
"""

# Varying the power
A = np.random.rand(6,6)
A += A.T
def powertrace_eigs(A,n):
    """
        Get the power trace up to n-1 power of A
    """
    powertrace = []  
    for i in range(n):
        powered_matrix = np.linalg.matrix_power(A,i)
        eigs = np.linalg.eigvals(powered_matrix)
        powertrace.append(np.sum(eigs))        
    return powertrace

n = 20
eig_powertrace = []
charpoly_powertrace = []
for i in range(n):
    time_eigs = %timeit -o  powertrace_eigs(A,i)
    time_powertrace = %timeit -o charpoly.powertrace(A, i)
    eig_powertrace.append(time_eigs.average)
    charpoly_powertrace.append(time_powertrace.average)
temps = np.linspace(1,20,20, dtype = int)
plt.xlabel('number of powers')
plt.ylabel('time taken on average (t)')
plt.title("Time to find powertrace of matrix 6 by 6 for charpoly and eigenvalue algorithms")
plt.plot(temps, eig_powertrace, label = 'eigenvalues')
plt.plot(temps, charpoly_powertrace, label = 'charpoly')
plt.legend(loc = 'upper right')
