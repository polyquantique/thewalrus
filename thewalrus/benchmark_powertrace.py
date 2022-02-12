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
    powertrace_list = []  
    for i in range(n):
        powered_matrix = np.linalg.matrix_power(A,i)
        eigs = np.linalg.eigvals(powered_matrix)
        powertrace_list.append(np.sum(eigs))        
    return powertrace_list

n = 20
eig_powertrace = []
charpoly_powertrace = []
for i in range(n):
    time_eigs = %timeit -o  powertrace_eigs(A,i)
    time_powertrace = %timeit -o powertrace(A, i)
    eig_powertrace.append(time_eigs.average)
    charpoly_powertrace.append(time_powertrace.average)
temps = np.linspace(1,20,20, dtype = int)
plt.xlabel('number of powers')
plt.ylabel('time taken on average (s)')
plt.title("Time to find powertrace of matrix 6 by 6 for charpoly and eigenvalue algorithms")
plt.plot(temps, eig_powertrace, label = 'eigenvalues')
plt.plot(temps, charpoly_powertrace, label = 'charpoly')
plt.legend(loc = 'upper right')
# Varying the matrix size
max_size = 16
eig_powertrace = []
charpoly_powertrace = []
for i in range(6,max_size):
    A = np.random.rand(i,i)
    A += A.T
    time_eigs = %timeit -o  powertrace_eigs(A,5)
    time_powertrace = %timeit -o powertrace(A, 5)
    eig_powertrace.append(time_eigs.average)
    charpoly_powertrace.append(time_powertrace.average)
temps = np.linspace(6,16,10, dtype = int)
plt.xlabel('size of matrix')
plt.ylabel('time taken on average (s)')
plt.title("Time to find powertrace for charpoly and eigenvalue algorithms for power to the 6")
plt.plot(temps, eig_powertrace, label = 'eigenvalues')
plt.plot(temps, charpoly_powertrace, label = 'charpoly')
plt.legend(loc = 'upper right')
