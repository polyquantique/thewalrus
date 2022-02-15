import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from thewalrus import charpoly

"""
@jit(nopython = True)
def powertrace_eigs(A,n):
    """
"""
        Get the power trace up to n-1 power of A
    """
"""
    eigs = np.linalg.eigvals(A)
    powertrace = np.zeros(n).astype(complex)
    for i in range(n):
        powertrace[i] = np.sum(eigs ** i)
    return powertrace
def get_varying_power_time(A, max_power):
    """
"""
        Vary the value of power
    """
"""
    eig_powertrace = []
    charpoly_powertrace = []
    for i in range(max_power):
        time_eigs = %timeit -o powertrace_eigs(A,i)
        time_powertrace = %timeit -o charpoly.powertrace(A, i)
        eig_powertrace.append(time_eigs.average)
        charpoly_powertrace.append(time_powertrace.average)
    return eig_powertrace, charpoly_powertrace
def get_varying_size_time(max_dim, power):
    """
"""
        Vary the value of power
    """
"""
    eig_powertrace = []
    charpoly_powertrace = []
    for i in range(6,max_dim):
        A = np.random.rand(i,i).astype(complex)
        time_eigs = %timeit -o powertrace_eigs(A, power)
        time_powertrace = %timeit -o charpoly.powertrace(A, power)
        eig_powertrace.append(time_eigs.average)
        charpoly_powertrace.append(time_powertrace.average)
    return eig_powertrace, charpoly_powertrace
A = np.random.rand(6,6).astype(complex)
A += A.T
eigs, charpol = get_varying_power_time(A, max_power = 20)
eigs_size, charpol_size = get_varying_size_time(max_dim = 16, power = 4)
temps1 = np.linspace(1,20,20, dtype = int)
plot1 = plt.figure(1)
plt.xlabel('number of powers')
plt.ylabel('time taken on average (s)')
plt.title("Time to find powertrace of matrix 6 by 6 for charpoly and eigenvalue algorithms")
plt.plot(temps1, eigs, label = 'eigenvalues')
plt.plot(temps1, charpol, label = 'charpoly')
plt.legend(loc = 'upper right')
temps2 = np.linspace(1,10,10, dtype = int)
plot2 = plt.figure(2)
plt.xlabel('Size of matrix')
plt.ylabel('time taken on average (s)')
plt.title("Time to find powertrace of matrix 6 by 6 for charpoly and eigenvalue algorithms")
plt.plot(temps2, eigs_size, label = 'eigenvalues')
plt.plot(temps2, charpol_size, label = 'charpoly')
plt.legend(loc = 'upper right')
plt.show()
"""
@jit(nopython = True)
def powertrace_eigs(A,n):
    """
        Get the power trace up to n-1 power of A
    """
    powers = np.zeros(n)
    eigs = np.linalg.eigvals(A)
    for i in range(n):
        powers[i] = np.real(np.sum(eigs ** i ))
    return powers
# Vary the value of power
def get_varying_power_time(A, max_power):
    eig_powertrace = []
    charpoly_powertrace = []
    for i in range(max_power):
        time_eigs = %timeit -o powertrace_eigs(A,i)
        time_powertrace = %timeit -o charpoly.powertrace(A, i)
        eig_powertrace.append(time_eigs.average)
        charpoly_powertrace.append(time_powertrace.average)
    return eig_powertrace, charpoly_powertrace
# Vary the dimension of the matrix
def get_varying_size_time(max_dim, power):
    eig_powertrace = []
    charpoly_powertrace = []
    for i in range(6,max_dim):
        A = np.random.rand(i,i).astype(complex)
        time_eigs = %timeit -o powertrace_eigs(A, power)
        time_powertrace = %timeit -o charpoly.powertrace(A, power)
        eig_powertrace.append(time_eigs.average)
        charpoly_powertrace.append(time_powertrace.average)
    return eig_powertrace, charpoly_powertrace
A = np.random.rand(6,6)
A += A.T
eigs, charpol = get_varying_power_time(A, max_power = 30)
eigs_size, charpol_size = get_varying_size_time(max_dim = 26, power = 6)
temps = np.linspace(6,26,20, dtype = int)
plt.xlabel('size of matrix')
plt.ylabel('time taken on average (s)')
plt.title("Time to find powertrace for charpoly and eigenvalue algorithms for power to the 6")
plt.plot(temps, eigs_size, label = 'eigenvalues')
plt.plot(temps, charpol_size, label = 'charpoly')
plt.legend(loc = 'upper right')
temps = np.linspace(1,30,30, dtype = int)
plt.xlabel('number of powers')
plt.ylabel('time taken on average (s)')
plt.title("Time to find powertrace of matrix 6 by 6 for charpoly and eigenvalue algorithms")
plt.plot(temps, eigs, label = 'eigenvalues')
plt.plot(temps, charpol, label = 'charpoly')
plt.legend(loc = 'upper right')
