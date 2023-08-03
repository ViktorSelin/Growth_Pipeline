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
