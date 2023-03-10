import numpy as np 
import matplotlib.pyplot as plt 
from scipy import optimize
import random
import warnings
import plot_utils

warnings.filterwarnings('ignore')

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

def p(n, T=1, gamma=10):
    nx,ny,nz = n
    return T * (nx + ny + nz + gamma * (nx*ny + ny*nz + nz*nx))

def gibbs(n, V, T=1, alpha=1, gamma=10):
    nx,ny,nz = n
    
    F = helmholtz(n, V=V)
    P = p(n, T, gamma)
    G = F + P*V
    return G


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
    plt.scatter(n_tot, F, s=8)
    # plt.scatter(n_tot[np.argmin(F)], min(F), label="min")
    plt.xlabel(r"$n = \frac{N}{V}$")
    plt.ylabel(r"$F$")
    plt.legend()
    if save:
        plt.savefig("../tex/figs/n_vs_F_zoomed.pdf")
    plt.show()

def get_indices_of_k_smallest(arr, k):
    """NB! Taken from: 
    https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
    """
    idx = np.argpartition(arr.ravel(), k)
    return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])

def plot_contour(n_tot, save=False):
    #NB look into what defines a minimum! 

    nx = ny = np.linspace(0.001, n_tot, 100)
    NX, NY = np.meshgrid(nx, ny)
    NZ = n_tot - (NX + NY)
    assert (np.abs(n_tot - (NX+NY+NZ)) < 1e-6).all

    F = helmholtz(np.array([NX,NY,NZ]))

    if n_tot < 0.2772:
        min_idx = np.unravel_index(np.nanargmin(F), F.shape)
    elif 0.2772 < n_tot < 0.2775:
        min_idx = get_indices_of_k_smallest(F, 4)
    else:
        min_idx = get_indices_of_k_smallest(F, 3)



    plt.contourf(nx, ny, F, levels=43, cmap='viridis')
    plt.plot(NX[min_idx], NY[min_idx], '*') 
    plt.title(f"n = {n_tot:.3f}")
    plt.xlabel(r"$n_x$")
    plt.ylabel(r"$n_y$")
    plt.colorbar()
    # plt.legend()
    if save:
        plt.savefig(f"../tex/figs/contour_nan_{n_tot}.pdf")
    plt.show()

def plot_gibbs(n_tot, n, save=False):
    G = np.zeros_like(n_tot)
    P = np.zeros_like(n_tot)
    for i in range(len(n_tot)):
        G[i] = gibbs(n[i], V=1/n_tot[i])
        P[i] = p(n[i])
    plt.scatter(P, G, s=2)
    plt.xlabel("P")
    plt.ylabel("G")
    if save: 
        plt.savefig(f"../tex/figs/gibbs.pdf")
    plt.show()

def plot_n_vs_P(n_tot, n, save=False):
    P = np.zeros_like(n_tot)
    F = np.zeros_like(n_tot)
    for i in range(len(n_tot)):
        P[i] = p(n[i])
        F[i] = helmholtz(n[i], V=1/n_tot[i])
    plt.scatter(n_tot, P, s=8)
    plt.xlabel("n")
    plt.ylabel("P")
    if save:
        plt.savefig(f"../tex/figs/n_vs_P_zoom.pdf")
    plt.show()

if __name__ == "__main__":
    # n_tot = 0.4
    # plot_contour(n_tot)

    n_tot = np.linspace(0.24, 0.7, 2000)
    n, F = opts_and_F(n_tot)
    # plot_nxnynz_vs_n(n_tot, n)
    # plot_F(n_tot, F)
    plot_gibbs(n_tot, n)
    # plot_n_vs_P(n_tot, n)


    





    

    """
    Want to create a way of finding the minimum of F given a certain N within which Nx, Ny, and Nz can vary. 
    N or n = N/V should be 
    """

# The constraint of the ni's is nx+ny+nz < 1
# n = nx + ny + nz