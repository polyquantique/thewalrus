import numpy as np
from number import jit
import matplotlib.pyplot as plt
from thewalrus import charpoly

@jit(nopython = True)
def powertrace_eigs(A,n):
    """
        Get the power trace of n-1 power of A
    """
    eigs = np.linalg.eigvals(A)
    eigs = eigs ** n
    powertrace = np.sum(eigs)
    return powertrace
def powertrace_eigs_list(A, n):
    """
        Get the power trace up to n-1 power of A
    """
    powertrace = []
    for i in range(n):
        power = powertrace_eigs(A, i)
        powertrace.append(power)
    return powertrace
def get_varying_power_time(A, max_power):
    """
        Vary the value of power
    """
    eig_powertrace = []
    charpoly_powertrace = []
    for i in range(max_power):
        time_eigs = %timeit -o powertrace_eigs_list(A,i)
        time_powertrace = %timeit -o charpoly.powertrace(A, i)
        eig_powertrace.append(time_eigs.average)
        charpoly_powertrace.append(time_powertrace.average)
    return eig_powertrace, charpoly_powertrace
def get_varying_size_time(max_dim, power):
    """
        Vary the value of power
    """
    eig_powertrace = []
    charpoly_powertrace = []
    for i in range(6,max_dim):
        A = np.random.rand(i,i)
        time_eigs = %timeit -o powertrace_eigs_list(A, power)
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
