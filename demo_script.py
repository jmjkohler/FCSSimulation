import os
import numpy as np
from simulate import make_param_dict, gen_single_species_trace

outdir = 'simulated_traces'

if __name__ == '__main__':
    npoints = 6000000 # number of steps to simulate
    D = 2.53125 # diffusion coefficient, um**2/sec
    omega_xy = 0.45 # PSF waist (um)
    bound = 9 # edge of region (1/2 of full width, um)
    eps = .03 # brightness parameter
    ntraces = 2500  # number of trajectories to simulate on each core; total number of
                # particles is ntraces*n_processes
    dt = 1e-5 # time step (sec)
    n_processes = 8

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    curr_params = make_param_dict(npoints, D, omega_xy, bound, eps, ntraces, dt, n_processes)
    params_path = os.path.join(outdir,'params.txt')
    with open(params_path,'w') as file:
        for key, value in curr_params.items():
            file.write("{:<14}{:<14} \n".format(key,value))
    for i in range(2):
        temp = gen_single_species_trace(npoints, D, omega_xy, bound, dt, eps, ntraces, n_processes=n_processes)
        fname = os.path.join(outdir,'trace_'+str(i).zfill(2)+'.npy')
        np.save(fname,temp)
