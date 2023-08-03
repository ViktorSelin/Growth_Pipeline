import time
import numpy as np
import sys
import meep as mp
from GrowthSim import Growth_nonPeriodic as Growth
import matplotlib.pyplot as plt

#########
#Define Functions
#########

#Function that initializes the growth simulation
def Initialize_Growth(seed):
    #Setup parameters
    y_dim = 1500
    x_dim = 1000
    frac = 0.3
    mu = -2.5
    KbT = 0.2
    e_nn = 2
    e_nl = 1.5
    e_ll = 1
    nano_mob = 30
    nano_size = 3
    n_nano = int(frac*(x_dim*y_dim)/(nano_size*nano_size))

    args = {'x_dim':x_dim,
            'y_dim':y_dim,
            'n_nano':n_nano,
            'KbT':KbT,
            'mu':mu,
            'e_nn':e_nn,
            'e_nl':e_nl,
            'e_ll':e_ll,
            'nano_mob':nano_mob,
            'nano_size':nano_size}


    #Initialize growth
    growth = Growth(**args,seed=seed)

    #Empty boundaries
    growth.fluid[:,0] = 0
    growth.fluid[:,-1] = 0
    growth.fluid[0,:] = 0
    growth.fluid[-1,:] = 0

    #Add probes
    probe_width = 178
    mid_probe = int((x_dim-probe_width)/2)

    #Left probe
    growth.nano[mid_probe:mid_probe+probe_width,0:250] = 1
    growth.fluid[mid_probe:mid_probe+probe_width,0:250] = 0

    #Right upper probe
    upper_probe = int((x_dim//2-1.6*probe_width))
    growth.nano[upper_probe:upper_probe+probe_width,-250:] = 1
    growth.fluid[upper_probe:upper_probe+probe_width,-250:] = 0

    #Right lower probe
    lower_probe = int((x_dim//2+0.6*probe_width))
    growth.nano[lower_probe:lower_probe+probe_width,-250:] = 1
    growth.fluid[lower_probe:lower_probe+probe_width,-250:] = 0

    #Initialize nanoparticles
    growth.initialize_nano()
    return growth


#Function that uses PyMeep to simulate light through the chip
def Simulate_Light(chip):

    #PyMeep simulation parameters
    resolution = 60
    tmax = 100
    flux_width = 1.3

    #Calculate dimensions
    shape = chip.shape
    lattice_size = 2.8/1000
    xspan = shape[0]*lattice_size
    yspan = shape[1]*lattice_size
    w = 178*lattice_size

    #Turn chip into epsilon array, epsilon for nano and 1 for air
    n = 3.5
    epsilon_nano = n**2
    epsilon_air = 1
    chip = (chip)*(epsilon_nano-epsilon_air)+epsilon_air


    #Initialize PyMeep objects
    cell = mp.Vector3(xspan,yspan,0)
    pml_layers = [mp.PML(0.5)]

    wcen = 1.4
    fcen = 1/wcen
    df = 0.5

    sources = [mp.Source(mp.GaussianSource(fcen,fwidth=df),
                        component=mp.Ez,
                        center=mp.Vector3(0,-2.8,0),
                        size=mp.Vector3(w*1.5,0,0))]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        sources=sources,
                        default_material=chip,
                        resolution=resolution,
                        filename_prefix=str(seed))

    nfreq = 100

    flux_fr_up = mp.FluxRegion(center=mp.Vector3(0.55,2.5,0),size=mp.Vector3(w*flux_width,0,0))
    flux_fr_dw = mp.FluxRegion(center=mp.Vector3(-0.55,2.5,0),size=mp.Vector3(w*flux_width,0,0))
    flux_up = sim.add_flux(fcen,df,nfreq,flux_fr_up)
    flux_dw = sim.add_flux(fcen,df,nfreq,flux_fr_dw)

    #sim.use_output_directory('Results')
    sim.run(until=tmax)

    chip_flux_up = mp.get_fluxes(flux_up)
    chip_flux_dw = mp.get_fluxes(flux_dw)

    eps = sim.get_array(center=mp.Vector3(),size=cell,component=mp.Dielectric)
    #sim.reset_meep()
    return eps,chip_flux_up,chip_flux_dw



#########
#Main
#########

#Get seed from arg
seed = int(sys.argv[1])

t_start = time.time()
##########
#Phase 1 - Initialize Growth
##########


growth = Initialize_Growth(seed)

print('Seed: %g, placed %g nanoparticles out of %g'%(seed,growth.n_nano_placed,growth.n_nano),flush=True)

#if seed < 5:
#    np.savez('Results/InitialGrowth_seed_%g.npz'%(seed),nano=growth.nano,fluid=growth.fluid)


##########
#Phase 2 - Run Growth
##########


MC_steps = 2000

print('Starting growth with %g MC steps'%(MC_steps),flush=True)

t0 = time.time()

for n in range(MC_steps+1):
    growth.step_simulation()
    if n%(MC_steps//10) == 0:
        t1 = time.time()
        print('Step %g, progress: %.1f%%, time: %.1f min'%(n,n/MC_steps*100,(t1-t0)/60),flush=True)

t1 = time.time()

#np.savez('Results/FinalGrowth_seed_%g.npz'%(seed),nano=growth.nano,fluid=growth.fluid)

fig,ax = plt.subplots(figsize=(10,10))
ax.imshow(growth.nano,cmap='gnuplot',interpolation='nearest',vmin = 0,vmax=1,origin='lower')
plt.tight_layout()
plt.axis('off')
fig.savefig('Results/FinalGrowth_seed_%g.png'%(seed))
plt.close(fig)

print('Seed: %g, finished growth in %.1f min'%(seed,(t1-t0)/60),flush=True)


##########
#Phase 3 - PyMeep Simulation
##########


chip = np.pad(growth.nano,((250,250),(500,500)),mode='edge')
del growth


t0 = time.time()
print('Starting PyMeep simulation',flush=True)

eps,flux_up,flux_dw = Simulate_Light(chip)

np.savez('Results/PyMeepResult_seed_%g.npz'%(seed),eps=eps,flux_up=flux_up,flux_dw=flux_dw)

t1 = time.time()
print('PyMeep simulation finished for seed %g after %.1f min'%(seed,(t1-t0)/60),flush=True)

t_final = time.time()

print('Finished job in total time: %.1f min'%((t_final-t_start)/60))
