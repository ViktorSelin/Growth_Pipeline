# Growth_Pipeline
Files for growth pipeline project

## **Baseline.npz, Baseline_eps.npz** 
Flux and dielectric information used to calculate transmission.

These files are generated from the **Score_Growth_3D()** function in **GrowthPipeline.py** with the "baseline=**True**" argument. The python file **Run_Baseline.py** does this so running *python Run_Baseline.py* is sufficient.

## **GrowthPipeline.py**
Contains a bunch of supporting functions. The most important ones being **Score_Growth_3D()** and **Initialize_Growth()**.


## **GrowthSim.py** / **GrowthSimCleaner.py** / **GrowthSimOptim.py**
**GrowthSim.py** was the initial implementation which is found in the Normal Pipeline folder, the other two are optimized versions in the Linear Attraction Pipeline. 

Should always use either GrowthSimCleaner or GrowthSimOptim, GrowthSim is there for reference. 

The difference between GrowthSimCleaner and GrowthSim is simply some minor optimization.

GrowthSimOptim changes how the solvent update works, it's about 40% faster but raised some issues with relation to detailed balance - Stephen can explain a bit. From the small tests I did it seems fine though. This version was made by Kyle. (Kyle' on Slack)


## **LearnGrowth.py**
Main file to run the entire pipeline. The imports show which functons from GrowthPipeline are used.

The **Simulate()** function defined here is what is run on each child process for the multiprocessing pool.starmap call.

Checkpointing occurs before the MC learning runs.

## **GrowthNetworks.py**
Contains the network used for policy generation. The linear attraction folder has an altered one which uses 4 outputs as compared to the normal 2.

The networks are initialized in **LeanGrowth.py** using the **Initialize_Net()** function from **GrowthPipeline.py**


## **SubmitArrayJob.sh**
Slurm file to submit a run to the cluster

Submit to the cluster when in the correct anaconda enviroment using **sbatch SubmitArrayJob.sh**




# THINGS TO CHANGE WHEN RUNNING
## In **LearnGrowth.py**:

**IMPORTANT:** Changing the reproducibility seeds to make sure you aren't running the same trajectory every time you start running.

**n_growths:** Determines how many growths occur for the current epoch. I usually stick around 40-80 for these tests.

**n_pool:** Controls how many growths occur in parallel, limited by where it's run, on Chromia 40 is the maximum.

Otherwise the Evolutionary learning parameters are pretty straight forwards.

## In **GrowthPipeline.py**

**Score_Growth_3D()** is what I usually alter when changing the scoring/test to be run. 

Usually I only change the lines at the bottom:

scorer = bandpass_scorer_target(wvl)
score = -np.sum(np.square(scorer-Ts_out))/len(Ts_out)

This calculates the score, changing the **bandpass_scorer_target()** function to something else would alter the scoring method. The next line is simply MSE score.
 
**Initialize_Growth()** can be changed if you want to mess with the chip layout, I.E have more output channels or whatnot.

# Output files
## Contents
### Results/

**growth_epoch_%g_index_%g.npz** is created in **LearnGrowth.py** and contains:

> *"growth"*: lattice of nanoparticles

> *"score"*: score of this growth

> *"flux_out"*: output flux of this growth

> *"ts_out"*: transmission of this growth

The associated frequencies for the flux and transmission is located in **baseline.npz**, *"wvl"*

**scores_epoch_%g.npz** is created in **LearnGrowth.py** and contains:

> *"score"*: array of all scores for this epoch

> *"mean_score"*: mean score for this epoch

> *"cur_score"*: current best mean score

### Checkpoint/

**Checkpoint_epoch_%g.npz** is created in **GrowthPipeline.py** and contains:

> *"cur_score"*: current best mean score

> *"seeds"*: rng seeds for the individual growths

> *"np_rng"*: numpy rng state 

> *"torch_rng"*: pytorch rng state

**NetStateDict_epoch_%g.pt** is created in **GrowthPipeline.py** and contains:

> The state dict (weights & biases) of the network for the current epoch

### Slurm_Outputs/

**slurm_output_%jobID_%g.out** is created by Slurm and contains:

> Terminal output for that submitted job

## Notes on usage

Import files with np.load(f), and extract array as a dictionary, 

ex:

> f = np.load("growth_epoch_1_index_0.npz")
> 
> growth = f['growth']

### Plotting

- To plot the growths:

Loading and padding the growth to match PyMeep simulation
```
with np.load(basefolder+'growth_epoch_%g_index_%g.npz'%(epoch,j)) as f:
   chip = f['growth']
   x_dim = chip.shape[0]
   chip = np.pad(chip,((x_dim//4,x_dim//4),(x_dim//2,x_dim//2)),mode='edge')
```
Then plot using imshow:
> plt.imshow(chip,cmap='gnuplot',interpolation='nearest',vmin=0,vmax=1)

- To plot the scores:

Loading the scores
```
scores = []
mean_scores = []
cur_scores = []
for i in range(max_epoch):
    with np.load(basefolder+'scores_epoch_%g.npz'%(i)) as f:
        scores.append(f['score'])
        mean_scores.append(f['mean_score'][0])
        cur_scores.append(f['cur_score'][0])
scores = np.array(scores)
mean_scores = np.array(mean_scores)
cur_scores = np.array(cur_scores)
```

Plotting is then simple, ex:
> plt.plot(np.arange(0,max_epoch),mean_scores)

- To plot the policies
```
from FullPipeline_MC_Test2.GrowthNetworks import Net

for i in range(len(epochs)):
    epoch = epochs[i]
    net = Net()
    net.load_state_dict(torch.load(checkpoint_folder+'NetStateDict_epoch_%g.pt'%(epoch)))
    net_output = net(torch.linspace(0,1,100).reshape(-1,1))
    net_output = net_output.detach().numpy()
    Xs = np.linspace(0,1,100)

    #Get parameter policies
    Ts = net_output[:,0]
    Mus = net_output[:,1]
    Att_xs = net_output[:,2]
    Att_es = net_output[:,3]

    #Plot
    axes[0].plot(Xs,Ts)
    axes[1].plot(Xs,Mus)
    ...
```

