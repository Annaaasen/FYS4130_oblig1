import numpy as np 
import matplotlib.pyplot as plt 
from scipy import optimize
import random

#For various Ns we want to find the corresponding Nx, Ny, and Nz that minimize F

def helmholtz(n, T=1, V=1, alpha=1, gamma=10):
    """Calculate the Helmholtz free energy given a certain consentration of particles 

    Args:
        n (_type_): array containing nx, ny, nz
    """
    nx,ny,nz = n
    F = T * V * (nx * np.log(alpha*nx) + ny * np.log(alpha*ny) + nz * np.log(alpha*nz)\
                  + gamma * (nx*ny + ny*nz + nz*nx))
    return F

def init_ns(n_tot):
    # Taking in a scalar where n_tot = nx + ny + nz 
    # Initializing the n's, assuring nx + ny + nz = n
    # Does this have to be initialized every time?
    nx = random.uniform(0,n_tot)
    ny = random.uniform(0,n_tot)
    nz = n_tot - (nx + ny)
    
    n = np.array([nx,ny,nz])

    print(n_tot, n)
    return n

def init_ns_arr(n_tot):
    #in this case n_tot is an array
    nx = np.random.uniform(low=0, high=n_tot/2, size=n_tot.size)
    ny = np.random.uniform(low=0, high=n_tot/2, size=n_tot.size)
    nz = n_tot - (nx + ny)

    n = np.array([nx,ny,nz])
    return n

def plot_slice():
    nx = np.linspace(0, 0.9, 1000)
    ny = 0.01
    nz = 0.01
    n  = np.array([nx, ny, nz])

    F = helmholtz(n)

    plt.plot(nx, F)
    plt.show()

def contour_plot(n_tot):
    nz = n_tot/4 
    nx = ny = np.linspace(0.001, n_tot/3, 100)

    NX, NY = np.meshgrid(nx, ny)

    F = helmholtz(np.array([NX,NY,nz]))

    plt.contourf(nx, ny, F)
    plt.colorbar()
    plt.title(f"n = {n_tot:.3f}, n/3 = {nz:.3f}")
    plt.show()

def constraint_1(n):
    #This constraint is of the type ineq
    nx, ny, nz = n 
    return 1 - (nx + ny + nz)

def constraint_2(n, n_tot):
    #This constraint is of the type eq
    nx, ny, nz = n 
    return n_tot - (nx + ny + nz)

def opt(n_tot):
    from scipy.optimize import Bounds, LinearConstraint 

    bounds = Bounds(0.001, 1)

    cons1 = {"type": "ineq", "fun": constraint_1}
    cons2 = {"type": "eq", "fun": constraint_2, "args": (n_tot,)}
    cons = [cons1, cons2]

    n0 = init_ns_arr(n_tot)
    
    print(helmholtz(n0))
    print(n0.ndim)
    #Seems like sending in a 2d array as n0 is problematic!

    res = optimize.minimize(helmholtz, n0, method="SLSQP", bounds=bounds, constraints=cons)

    print(res.x)
    return res.x

def plot_loop():
    n_tot = np.linspace(0.01, 0.9, 100)
    n = np.zeros((len(n_tot), 3))
    
    for i in range(len(n_tot)):
        n[i] = opt(n_tot[i])

    plt.plot(n_tot, n)
    plt.show()




if __name__ == "__main__":
    n_tot = np.linspace(0.01, 0.99, 10)
    opt(n_tot) 

    

    """
    Want to create a way of finding the minimum of F given a certain N within which Nx, Ny, and Nz can vary. 
    N or n = N/V should be 
    """

# The constraint of the ni's is nx+ny+nz < 1
# n = nx + ny + nz