import numpy as np
from multiprocessing import Pool
import os
from numba import jit, vectorize, float64

@vectorize([float64(float64,float64)], nopython=True)
def wrap_bounds(trajectory, bound):
    return ((trajectory+bound)%(2*bound)) - bound

@vectorize([float64(float64,float64,float64)], nopython=True)
def gaus2d(x,y,omega_xy):
    return np.exp((-4*(x**2 + y**2))/(omega_xy**2))

def gen_trajectory(npoints, D, bound, dt, trajectory_arr, rng):
    deltas = rng.normal(0,np.sqrt(2*D*dt),size=(2,npoints))
    xy0 = rng.uniform(-1*bound, bound, size=(2,1))
    np.cumsum(deltas, axis=1, out=trajectory_arr)
    trajectory_arr += xy0

@jit(nopython=True)
def traj_to_counts(trajectory_arr, eps, omega_xy):
    counts = eps*gaus2d(trajectory_arr[0],trajectory_arr[1],omega_xy)
    return counts

def gen_noiseless_counts(npoints, D, omega_xy, bound, dt, eps, ntraces, seed):
    rng = np.random.default_rng(seed)
    trajectory_arr = np.empty((2, npoints))
    temp_nc = np.zeros(npoints)
    for i in range(ntraces):
        gen_trajectory(npoints, D, bound, dt, trajectory_arr, rng)
        trajectory_arr = wrap_bounds(trajectory_arr, bound)
        temp_nc += traj_to_counts(trajectory_arr, eps, omega_xy)
    return temp_nc

def calc_taud(omega_xy, D):
    '''calculates td in ms from psf size and D for 2photon'''
    return 1000*((omega_xy**2)/(8*D))

def calc_N(n_processes,ntraces,bound,omega_xy):
    '''calculates N from PSF size and concentration of molecules simulated'''
    return (n_processes*ntraces/((2*bound)**2))*(.5*np.pi*(omega_xy**2))

def make_param_dict(npoints, D, omega_xy, bound, eps, ntraces, dt, n_processes):
    N = calc_N(n_processes, ntraces, bound, omega_xy)
    paramdict = dict({'npoints':npoints,
                      'D':D,
                      'omega_xy':omega_xy,
                      'bound':bound,
                      'eps':eps,
                      'ntraces':ntraces,
                      'dt':dt,
                      'n_processes':n_processes,
                      'N': N})
    return paramdict

def gen_single_species_trace(npoints, D, omega_xy, bound, dt, eps, ntraces,
                             seed=None, n_processes=None):
    """ Simulate FCS photon count record

    Parameters
    ----------
    npoints : int
        number of timesteps to simulate
    D : float
        Diffusion coefficient in μm^2/s
    omega_xy : float
        size of the 2D Gaussian PSF (2 photon)
    bound : float
        size of the region to simulate (half-length of the box edge)
    dt : float
        time step size, in seconds
    eps : float
        brightness parameter
    ntraces : int
        number of particle trajectories to simulate on each core
    seed : {None, int, SeedSequence}, optional
        Seed passed to np.random.default_rng(). Default is None
    n_processes : {None, int}, optional
        Number of cores to use. If not specified, uses value returned by
        os.cpu_count().

    Returns
    -------
    noisyres : np.ndarray
        array of photon counts with poisson noise applied
    """
    if not n_processes:
        n_processes = os.cpu_count()
    rng = np.random.default_rng(seed)
    ss = rng.bit_generator._seed_seq
    child_states = ss.spawn(n_processes)
    with Pool(n_processes) as pool:
        argtups = [(npoints, D, omega_xy, bound, dt, eps, ntraces, child_state)
                   for child_state in child_states]
        reslist = pool.starmap(gen_noiseless_counts, argtups)
        resarr = np.array(reslist).sum(axis=0)
        noisyres = rng.poisson(resarr)
    return noisyres

def gen_two_species_trace(npoints, D1, D2, omega_xy, bound, dt, eps, ntraces1,
                          ntraces2, seed=None, n_processes=None):
    """ Simulate 2 diffusing species """
    if not n_processes:
        n_processes = os.cpu_count()
    rng = np.random.default_rng(seed)
    ss = rng.bit_generator._seed_seq
    child_states = ss.spawn(n_processes)
    with Pool(n_processes) as pool:
        argtups = [(npoints, D1, omega_xy, bound, dt, eps, ntraces1, child_state)
                   for child_state in child_states]
        reslist = pool.starmap(gen_noiseless_counts, argtups)
        resarr = np.array(reslist).sum(axis=0)
        child_states = ss.spawn(n_processes)
        argtups2 = [(npoints, D2, omega_xy, bound, dt, eps, ntraces2, child_state)
                   for child_state in child_states]
        reslist2 = pool.starmap(gen_noiseless_counts, argtups2)
        resarr2 = np.array(reslist2).sum(axis=0)
        totalres = np.sum((resarr,resarr2), axis=0)
        noisytotalres = rng.poisson(totalres)
    return noisytotalres
