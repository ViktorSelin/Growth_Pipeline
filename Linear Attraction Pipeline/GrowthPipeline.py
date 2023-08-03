import numpy as np
import copy
import torch
import meep as mp
from GrowthSimCleaner import Growth_NonPeriodic as Growth

mp.verbosity(0)

#Function that generates a random seed
def Generate_Seed():
    return np.random.randint(0,np.iinfo(np.uint32).max,dtype=np.uint32)

#Function that generates random new seeds for each network
def Generate_New_Seeds(seeds):
    for n in range(len(seeds)):
        seeds[n][0] = Generate_Seed()
        
#Function that initializes networks
def Initialize_Net(net):
    net.output.bias[0] -= 1.0
    net.output.bias[1] -= 2.5
    net.output.bias[3] += 0.4
#Function that sorts networks based on score - return sorted indices
def Sort_Growths(seeds,scores):
    #Sort from high to low
    sorted_indices = np.argsort(scores)[::-1]
    scores = [scores[i] for i in sorted_indices]
    seeds = [seeds[i] for i in sorted_indices]

    #Reorder the indices
    for i in range(len(seeds)):
        seeds[i][1] = i
    return seeds, scores

#Function that copies kept networks into culled networks
#def Clone_Nets(listOfNets,kept):
#    for i in range(len(listOfNets)-kept):
#        listOfNets[kept+i][0] = copy.deepcopy(listOfNets[i%kept][0])

#Function that mutates the networks
#Does not mutate 
def Mutate_Net(net,sigma):
    for param in net.parameters():
        param += torch.normal(torch.zeros(param.shape),sigma)

#Function that initializes a growth simulation
def Initialize_Growth(seed,baseline=False):
    #Setup parameters
    y_dim = 750
    x_dim = 500
    frac = 0.3
    mu = -2.5
    KbT = 0.2
    e_nn = 2
    e_nl = 1.5
    e_ll = 1
    nano_mob = 30
    nano_size = 3
    n_nano = int(frac*(x_dim*y_dim)/(nano_size*nano_size))
    lattice_size = 2.8/x_dim

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
    #Probe width
    w = 0.5

    #Current dimension, 1000 = 2.8 micro, this gives us how many lattice cells for the probe width
    probe_width = int(w/lattice_size)
    mid_probe = int((x_dim-probe_width)/2)

    if baseline:
        #Width probe
        growth.nano[mid_probe:mid_probe+probe_width,:] = 1
        growth.fluid[mid_probe:mid_probe+probe_width,:] = 0

    else:
        #Left probe
        growth.nano[mid_probe:mid_probe+probe_width,0:(x_dim//4)] = 1
        growth.fluid[mid_probe:mid_probe+probe_width,0:(x_dim//4)] = 0

        #Right probe
        growth.nano[mid_probe:mid_probe+probe_width,-(x_dim//4):] = 1
        growth.fluid[mid_probe:mid_probe+probe_width,-(x_dim//4):] = 0

        #Initialize nanoparticles
        growth.initialize_nano()
    return growth

#Function that scores chip growth using PyMeep in 3D
#DOES NOT WORK -- CHIP TOO BIG, need to scale it down before?
'''
def Score_Growth_3D_OLD(chip,baseline=False):
    #Setup chip
    #Dielectric constants
    n_si = 3.49
    n_sio4 = 1.45
    n_air = 1.0
    epsilon_si = n_si**2
    epsilon_sio4 = n_sio4**2
    epsilon_air = n_air**2

    #Lattice size - hardcoded w.r.t growth simulation resolution
    lattice_size = 2.8/1000

    #Calculate width/height of input/output probe(s)
    w_lattice = np.sum(chip[:,0]) 
    print('w_lattice: %g'%(w_lattice),flush=True)
    w = w_lattice * lattice_size
    h = 0.22
    h_lattice = int(h/lattice_size)

    #Padding to 3D chip
    chip = np.pad(chip,((250,250),(500,500)),mode='edge')
    chip = (chip)*(epsilon_si-epsilon_air) + epsilon_air
    chip.shape += 1,

    #Pad chip height
    chip = np.pad(chip,((0,0),(0,0),(h_lattice//2,h_lattice//2)),mode='edge') 

    #SiO4 pad
    chip = np.pad(chip,((0,0),(0,0),(0,int(1/lattice_size))),mode='constant',constant_values=epsilon_sio4)

    #Air pad
    chip = np.pad(chip,((0,0),(0,0),(int(1/lattice_size),0)),mode='constant',constant_values=epsilon_air)
    
    #Setup for PyMeep simulation
    shape = chip.shape
    xspan = shape[0]*lattice_size
    yspan = shape[1]*lattice_size
    zspan = shape[2]*lattice_size
    
    #Cell & boundary
    cell = mp.Vector3(xspan,yspan,zspan)
    pml_layers = [mp.PML(0.5)]

    #Source parameters
    wcen = 1.4
    fcen = 1/wcen
    df = 0.5

    #Simulation parameters
    resolution = 25
    tmax = 100

    #Source
    sources = [mp.Source(mp.GaussianSource(fcen,fwidth=df),
                        component=mp.Ex,
                        center=mp.Vector3(0,-2.8,0),
                        size=mp.Vector3(w*1.2,0,h*1.2))]

    #Main simulation
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        sources=sources,
                        default_material=chip,
                        resolution=resolution)
    
    #Flux monitor parameters
    nfreq = 100
    flux_width = 1.1
    #Different for baseline (measuring input) vs chip (2 outputs)
    if baseline:
        flux_fr_in = mp.FluxRegion(center=mp.Vector3(0,2.5,0),size=mp.Vector3(w*flux_width,0,h*flux_width))
        flux_in = sim.add_flux(fcen,df,nfreq,flux_fr_in)
        print('Starting PyMeep baseline simulation',flush=True)
        sim.run(until=tmax)
        print('Finished PyMeep baseline simulation',flush=True)
        baseline_flux = mp.get_fluxes(flux_in)
        flux_freqs = mp.get_flux_freqs(flux_in)
        eps = sim.get_array(center=mp.Vector3(),size=cell,component=mp.Dielectric)
        sim.reset_meep()
        np.savez_compressed('Baseline_eps.npz',eps=eps)
        np.savez_compressed('Baseline.npz',flux=baseline_flux,freqs=flux_freqs)
        print('Baseline saved',flush=True)
        return eps
    else:
        flux_fr_up = mp.FluxRegion(center=mp.Vector3(0.55,2.5,0),size=mp.Vector3(w*flux_width,0,h*flux_width))
        flux_fr_dw = mp.FluxRegion(center=mp.Vector3(-0.55,2.5,0),size=mp.Vector3(w*flux_width,0,h*flux_width))
        flux_up = sim.add_flux(fcen,df,nfreq,flux_fr_up)
        flux_dw = sim.add_flux(fcen,df,nfreq,flux_fr_dw)
        print('Starting PyMeep simulation',flush=True)
        sim.run(until=tmax)
        print('Finished PyMeep simulation',flush=True)

        chip_flux_up = mp.get_fluxes(flux_up)
        chip_flux_dw = mp.get_fluxes(flux_dw)

        eps = sim.get_array(center=mp.Vector3(),size=cell,component=mp.Dielectric)
        sim.reset_meep()

        baseline = np.load('Baseline.npz')
        input_flux = baseline['flux']
        input_freqs = baseline['freqs']
        wvl = 1/input_freqs

        Ts_up = chip_flux_up/input_flux
        Ts_dw = chip_flux_dw/input_flux

        scorer = gaussian_scorer(wvl)
        score = abs(np.sum(scorer*Ts_up)+np.sum(-scorer*Ts_dw))

        return [score]
'''


#Function that generates an e_func based on lattice_size and chip
def generate_chip_e_func(chip,lattice_size):
    n_si = 3.49
    n_sio4 = 1.45
    n_air = 1.0
    epsilon_si = n_si**2
    epsilon_sio4 = n_sio4**2
    epsilon_air = n_air**2
     
    x_shape = chip.shape[0]
    y_shape = chip.shape[1]

    xspan = (chip.shape[0])*lattice_size
    yspan = (chip.shape[1])*lattice_size
 
    def chip_e_func(coord):
        x,y,z = coord
    
        if z > 0.11:
            return epsilon_sio4
        elif z < -0.11:
            return epsilon_air
        else:   
            x_lat = int((x+x_shape/2*lattice_size)/lattice_size)
            y_lat = int((y+y_shape/2*lattice_size)/lattice_size)
            if x >= xspan/2:
                x_lat = 0
            elif x <= -xspan/2:
                x_lat = -1
            if y >= yspan/2:
                y_lat = -1
            elif y <= -yspan/2:
                y_lat = 0
            return chip[x_lat,y_lat]
    return chip_e_func


#Function that runs PyMeep and scores growth in 3D
def Score_Growth_3D(chip,resolution = 25,flux_width=1.1,tmax=100,baseline=False,e_func=False):
    w_lattice = np.sum(chip[:,0])
    x_dim = chip.shape[0]
    y_dim = chip.shape[1]
    lattice_size = 2.8/x_dim
    w = w_lattice * lattice_size
    h = 0.22
    h_lattice = int(h/lattice_size)
    
    n_si = 3.49
    n_sio4 = 1.45
    n_air = 1.0
    epsilon_si = n_si**2
    epsilon_sio4 = n_sio4**2
    epsilon_air = n_air**2
    
    if not e_func:
        chip = np.pad(chip,((x_dim//4,x_dim//4),(x_dim//2,x_dim//2)),mode='edge')

        chip = (chip)*(epsilon_si-epsilon_air) + epsilon_air
        chip.shape += 1,
        chip = np.pad(chip,((0,0),(0,0),(h_lattice//2,h_lattice//2)),mode='edge')

        #SiO4 pad
        chip = np.pad(chip,((0,0),(0,0),(0,int(1/lattice_size))),mode='constant',constant_values=epsilon_sio4)

        #Air pad
        chip = np.pad(chip,((0,0),(0,0),(int(1/lattice_size),0)),mode='constant',constant_values=1.0)

        #Setup for PyMeep simulation
        shape = chip.shape
        xspan = shape[0]*lattice_size
        yspan = shape[1]*lattice_size
        zspan = shape[2]*lattice_size
        
    elif e_func:
        chip = (chip)*(epsilon_si-epsilon_air) + epsilon_air
        xspan = (chip.shape[0]+(x_dim//4)*2)*lattice_size
        yspan = (chip.shape[1]+(x_dim//2)*2)*lattice_size
        zspan = 2+h
        e_func = generate_chip_e_func(chip,lattice_size)
    cell = mp.Vector3(xspan,yspan,zspan)
    pml_layers = [mp.PML(0.5)]

    wcen = 1.3
    fcen = 1/wcen
    df = 0.5
    
    sources = [mp.Source(mp.GaussianSource(fcen,fwidth=df),
                        component=mp.Ex,
                        center=mp.Vector3(0,-2.8,0),
                        size=mp.Vector3(w*flux_width,0,h*flux_width))]

    if not e_func:
        sim = mp.Simulation(cell_size=cell,
                            boundary_layers=pml_layers,
                            sources=sources,
                            default_material=chip,
                            resolution=resolution)
    elif e_func:
         sim = mp.Simulation(cell_size=cell,
                            boundary_layers=pml_layers,
                            sources=sources,
                            epsilon_func=e_func,
                            resolution=resolution)   
    nfreq = 100

    if baseline:
        flux_fr_in = mp.FluxRegion(center=mp.Vector3(0,2.5,0),size=mp.Vector3(w*flux_width,0,h*flux_width))
        flux_in = sim.add_flux(fcen,df,nfreq,flux_fr_in)
        print('Starting PyMeep baseline simulation',flush=True)
        sim.run(until=tmax)
        print('Finished PyMeep baseline simulation',flush=True)
        baseline_flux = mp.get_fluxes(flux_in)
        flux_freqs = mp.get_flux_freqs(flux_in)
        eps = sim.get_array(center=mp.Vector3(),size=cell,component=mp.Dielectric)
        sim.reset_meep()
        np.savez_compressed('Baseline_eps.npz',eps=eps)
        np.savez_compressed('Baseline.npz',flux=baseline_flux,freqs=flux_freqs)
        print('Baseline saved',flush=True)
        return eps

    else:
        flux_fr_out = mp.FluxRegion(center=mp.Vector3(0,2.5,0),size=mp.Vector3(w*flux_width,0,h*flux_width))
        flux_out = sim.add_flux(fcen,df,nfreq,flux_fr_out)
        print('Starting PyMeep simulation',flush=True)
        sim.run(until=tmax)
        print('Finished PyMeep simulation',flush=True)

        chip_flux_out = mp.get_fluxes(flux_out)

        eps = sim.get_array(center=mp.Vector3(),size=cell,component=mp.Dielectric)
        sim.reset_meep()

        baseline = np.load('Baseline.npz')
        input_flux = baseline['flux']
        input_freqs = baseline['freqs']
        wvl = 1/input_freqs

        Ts_out = chip_flux_out/input_flux
        scorer = bandpass_scorer_target(wvl)
        score = -np.sum(np.square(scorer-Ts_out))/len(Ts_out)
        

        return score,chip_flux_out,Ts_out


#Function that scores growth based on PyMeep simulation - DEPRICATED  
'''
def Score_Growth_2D(chip,baseline=False):
    chip = np.pad(chip,((250,250),(500,500)),mode='edge')
    #PyMeep simulation parameters
    resolution = 40
    tmax = 100
    flux_width = 1.1

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
                        component=mp.Ex,
                        center=mp.Vector3(0,-2.8,0),
                        size=mp.Vector3(w*1.2,0,0))]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        sources=sources,
                        default_material=chip,
                        resolution=resolution,
                        )

    nfreq = 100
    if baseline:
        flux_fr_in = mp.FluxRegion(center=mp.Vector3(0,2.5,0),size=mp.Vector3(w*flux_width,0,0))
        flux_in = sim.add_flux(fcen,df,nfreq,flux_fr_in)
        print('Starting PyMeep baseline simulation',flush=True)
        sim.run(until=tmax)
        print('Finished PyMeep baseline simulation',flush=True)
        baseline_flux = mp.get_fluxes(flux_in)
        flux_freqs = mp.get_flux_freqs(flux_in)
        eps = sim.get_array(center=mp.Vector3(),size=cell,component=mp.Dielectric)
        sim.reset_meep()
        np.savez_compressed('Baseline_eps.npz',eps=eps)
        np.savez_compressed('Baseline.npz',flux=baseline_flux,freqs=flux_freqs)
        print('Baseline saved',flush=True)
        return eps

    else:
        flux_fr_up = mp.FluxRegion(center=mp.Vector3(0.55,2.5,0),size=mp.Vector3(w*flux_width,0,0))
        flux_fr_dw = mp.FluxRegion(center=mp.Vector3(-0.55,2.5,0),size=mp.Vector3(w*flux_width,0,0))
        flux_up = sim.add_flux(fcen,df,nfreq,flux_fr_up)
        flux_dw = sim.add_flux(fcen,df,nfreq,flux_fr_dw)
        print('Starting PyMeep simulation',flush=True)
        sim.run(until=tmax)
        print('Finished PyMeep simulation',flush=True)

        chip_flux_up = mp.get_fluxes(flux_up)
        chip_flux_dw = mp.get_fluxes(flux_dw)

        eps = sim.get_array(center=mp.Vector3(),size=cell,component=mp.Dielectric)
        sim.reset_meep()

        baseline = np.load('Baseline.npz')
        input_flux = baseline['flux']
        input_freqs = baseline['freqs']
        wvl = 1/input_freqs

        Ts_up = chip_flux_up/input_flux
        Ts_dw = chip_flux_dw/input_flux

        score = equal_trans(Ts_up,Ts_dw)

        return [score]
'''


def bandpass_scorer(wavelength):
    scorer = np.zeros(wavelength.shape)-2
    scorer += 2*np.exp(-np.square(wavelength-1.3)/(2*np.square(0.05)))
    return scorer

def bandpass_scorer_target(wavelength):
    scorer = np.zeros(wavelength.shape)
    scorer += np.exp(-np.square(wavelength-1.3)/(2*np.square(0.05)))
    return scorer



#Score transmissions as to equally split the light out both outputs
#Maximize total transmission and minimize difference in outputs (both in magnitude and what wvls)
def equal_trans(trans_up,trans_dw):
    total_trans = trans_up+trans_dw
    score = np.sum(total_trans) - np.sum(np.abs(trans_up-trans_dw))
    return score

#Function that returns an array to score an array of transmission
def gaussian_scorer(wavelength):
    scorer = np.zeros(wavelength.shape)
    scorer += np.exp(-np.square(wavelength-1.3)/(2*np.square(0.05)))
    scorer -= np.exp(-np.square(wavelength-1.55)/(2*np.square(0.05)))
    return scorer


#Save a checkpoint
def Save_Checkpoint(net,seeds,epoch,cur_score):
    seeds_l = []
    np_rng = np.random.get_state()
    torch_rng = np.array(torch.get_rng_state())
    torch.save(net.state_dict(),'Checkpoint/NetStateDict_epoch_%g.pt'%(epoch))
    for j in range(len(seeds)):
        seeds_l.append(seeds[j][0])
    np.savez_compressed('Checkpoint/Checkpoint_epoch_%g.npz'%(epoch),cur_score=np.array([cur_score]),seeds=np.array(seeds_l),np_rng=np_rng,torch_rng=torch_rng)
    print('Checkpoint Created',flush=True)

#Load a checkpoint
def Load_Checkpoint(net,seeds,epoch):
    checkpoint = np.load('Checkpoint/Checkpoint_epoch_%g.npz'%(epoch),allow_pickle=True)
    seeds_check = checkpoint['seeds']
    cur_score = checkpoint['cur_score'][0]
    np.random.set_state(tuple(checkpoint['np_rng']))
    torch.set_rng_state(torch.tensor(checkpoint['torch_rng']))
    for j in range(len(seeds)):
        seeds[j][0] = seeds_check[j]
    net.load_state_dict(torch.load('Checkpoint/NetStateDict_epoch_%g.pt'%(epoch)))
    print('Checkpoint Loaded',flush=True)
    return cur_score

#Save best performing networks
def Save_Network(net,epoch):
    torch.save(net.state_dict(),'Results/HighScore_NetStateDict_epoch_%g.pt'%(epoch))
