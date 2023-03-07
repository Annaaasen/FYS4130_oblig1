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
    nx = random.uniform(0,n_tot/2)
    ny = random.uniform(0,n_tot/2)
    nz = n_tot - (nx + ny)

    assert (np.abs(n_tot - (nx+ny+nz)) < 1e-6).all
    
    n = np.array([nx,ny,nz])
    return n


def constraint_1(n):
    #This constraint is of the type ineq
    nx, ny, nz = n 
    return 1 - (nx + ny + nz)

def constraint_2(n, n_tot):
    #This constraint is of the type eq
    nx, ny, nz = n 
    return n_tot - (nx + ny + nz)

def opt(n_tot):
    bounds = ([0.001, 1], [0.001, 1], [0.001, 1])

    cons1 = {"type": "ineq", "fun": constraint_1}
    cons2 = {"type": "eq", "fun": constraint_2, "args": (n_tot,)}
    cons = [cons1, cons2]

    n0 = init_ns(n_tot)

    res = optimize.minimize(helmholtz, n0, method="SLSQP", bounds=bounds, constraints=cons)

    return res.x

def opts_and_F(n_tot):
    """A function which takes in an array of n_tots and calculates optimal nx,ny,nz as well as corresponding Helmholtz to each n_tot

    Args:
        n_tot (array): 
    """
    n = np.zeros((len(n_tot), 3))
    F = np.zeros((len(n_tot)))
    
    for i in range(len(n_tot)):
        n[i] = opt(n_tot[i])
        F[i] = helmholtz(n[i])

    return n, F

### PLOTTING ###

def plot_nxnynz_vs_n(n_tot, n, save=False):
    lineObjects = plt.plot(n_tot, n)
    plt.xlabel(r"$n = \frac{N}{V}$")
    plt.legend(iter(lineObjects), (r"$n_x$", r"$n_y$", r"$n_z$"))
    if save:
        plt.savefig("../tex/figs/n_vs_xyz.pdf")
    plt.show()

def plot_F(n_tot, F, save=False):
    plt.scatter(n_tot, F)
    plt.scatter(n_tot[np.argmin(F)], min(F), label="min")
    plt.xlabel(r"$n = \frac{N}{V}$")
    plt.ylabel(r"$F$[E]")
    plt.legend()
    if save:
        plt.savefig("../tex/figs/n_vs_F.pdf")
    plt.show()

def plot_contour(n_tot, save=False):
    nx = ny = np.linspace(0.001, n_tot/2 - 0.001, 100)
    NX, NY = np.meshgrid(nx, ny)
    NZ = n_tot - (NX + NY)
    assert (np.abs(n_tot - (NX+NY+NZ)) < 1e-6).all

    F = helmholtz(np.array([NX,NY,NZ]))

    min_idx = np.unravel_index(F.argmin(), F.shape)

    plt.contourf(nx, ny, F)
    plt.plot(NX[min_idx], NY[min_idx], '*r', label=f"({NX[min_idx]:.3f}, {NY[min_idx]:.3f}, {NZ[min_idx]:.3f})")
    plt.colorbar()
    plt.title(f"n = {n_tot:.3f}")
    plt.legend()
    if save:
        plt.savefig(f"../tex/figs/contour_{n_tot}.pdf")
    plt.show()



if __name__ == "__main__":
    n_tot = 0.27
    plot_contour(n_tot)

    # n_tot = np.linspace(0.01, 0.9, 100)
    # n, F = opts_and_F(n_tot)
    # plot_nxnynz_vs_n(n_tot, n)
    # plot_F(n_tot, F)




    

    """
    Want to create a way of finding the minimum of F given a certain N within which Nx, Ny, and Nz can vary. 
    N or n = N/V should be 
    """

# The constraint of the ni's is nx+ny+nz < 1
# n = nx + ny + nz