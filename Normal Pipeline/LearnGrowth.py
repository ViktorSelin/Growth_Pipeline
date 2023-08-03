from GrowthPipeline import Generate_Seed, Generate_New_Seeds, Initialize_Net, Sort_Growths, Mutate_Net, Score_Growth_3D, Initialize_Growth,  Save_Checkpoint, Load_Checkpoint, Save_Network
from GrowthNetworks import Net

#Multiprocessing for Pool and cpu_count
import multiprocessing

import torch
import numpy as np
import time
import sys
import copy


####
#Function for Pool - single epoch
#####
def Simulate(seed,index):
    global net, epoch, n_pool
    #Initialize growth simulator - need random seed to be random for all processes
    if index%n_pool == 0:
        print('Index %g initializing growth'%(index),flush=True)
    growth = Initialize_Growth(seed)
    if index%n_pool == 0:
        print('Index %g growth initialized'%(index),flush=True)
    #Grow, querying net
    if index%n_pool == 0:
        print('Index %g starting growth'%(index),flush=True)
        growth_t0 = time.time()
    for n in range(MC_steps):
        if n%MC_query == 0:
            netOutput = net(torch.tensor([n/MC_steps]).reshape(-1,1).float()).detach().numpy()[0]
            newKbT = netOutput[0]
            newMu = netOutput[1]
            growth.KbT = newKbT
            growth.mu = newMu
            if index%n_pool == 0:
                growth_t1 = time.time()
                print('Index %g growth progress: %.1f%%, %g/%g in %.1f min'%(index,((n)/(MC_steps))*100,n,MC_steps,(growth_t1-growth_t0)/60),flush=True)
        growth.step_simulation()

    if index%n_pool == 0:
        growth_t1 = time.time()
        print('Index %g growth finished in %1f min'%(index,(growth_t1-growth_t0)/60),flush=True)

    #calculate score of net
    score,flux_out,ts_out = Score_Growth_3D(growth.nano,e_func=True)

    if index < 5:
        np.savez_compressed('Results/growth_epoch_%g_index_%g.npz'%(epoch,index),growth=growth.nano,score=np.array([score]),flux_out=np.array(flux_out),ts_out=np.array(ts_out))
    return [score]

#### OLD FUNCTION 
def Simulate_old(seed,index):
    global net, epoch
    #Initialize growth simulator - need random seed to be random for all processes
    if index == 0:
        print('Initializing growth',flush=True)
    growth = Initialize_Growth(seed)
    if index == 0:
        print('Growth initialized',flush=True)
    #Grow, querying net
    if index == 0:
        print('Starting growth',flush=True)
        growth_t0 = time.time()
    for n in range(MC_steps):
        if n%MC_query == 0:
            netOutput = net(torch.tensor([n/MC_steps]).reshape(-1,1).float()).detach().numpy()[0]
            newKbT = netOutput[0]
            newMu = netOutput[1]
            growth.KbT = newKbT
            growth.mu = newMu
            if index == 0:
                growth_t1 = time.time()
                print('Growth progress: %.1f%%, %g/%g in %.1f min'%(((n)/(MC_steps))*100,n,MC_steps,(growth_t1-growth_t0)/60),flush=True)
        growth.step_simulation()
    #calculate score of net
    score_output = Score_Growth(growth.nano)

    if index < 5:
        np.savez_compressed('Results/growth_epoch_%g_index_%g.npz'%(epoch,index),growth=growth.nano,score=np.array(score_output))
    if index == 0:
        growth_t1 = time.time()
        print('Growth finished in %1f min'%((growth_t1-growth_t0)/60),flush=True)
    return score_output


if __name__ == '__main__':
    #####
    #Setup parameters
    #####
    #Default on Linux - set it for consistency
    multiprocessing.set_start_method('fork')

    print('Number of CPU cores: %g'%(multiprocessing.cpu_count()))

    torch.set_grad_enabled(False)

    #Reproducibility
    np.random.seed(1237)
    torch.manual_seed(1237)

    #Evolutionary learning parameters
    epochs = 5
    MC_steps = 2000
    MC_query = 50
    sigma = 0.01

    #Network
    net = Net()

    #Setup
    n_growths = 80
    n_pool = 40
    seeds = []
    for i in range(n_growths):
        seeds.append([Generate_Seed(),i])

    #Determine current epoch and load checkpoint if not start
    base_epoch = int(int(sys.argv[1])*epochs)
    print('Starting epoch: %g'%(base_epoch))
    if base_epoch == 0:
        Initialize_Net(net)   
        cur_score = -np.inf
        print('Initialized Net for epoch 0 with initial score %.2f'%(cur_score),flush=True)
    else:
        cur_score = Load_Checkpoint(net,seeds,base_epoch-1)
        #Clone_Nets(listOfNets,kept)
        #Mutate_Net(net,sigma)
        #Generate_New_Seeds(seeds)
        print('Loaded checkpoint from epoch %g'%(base_epoch - 1),flush=True)

    #Starting time
    t0 = time.time()

    checkpointAlways = True
    #Run neuroevolution growth
    for epoch in range(base_epoch,base_epoch+epochs):
        tes = time.time()

        print('\n------------------------\nEpoch %g'%(epoch),flush=True)
        #Get previous state dict, mutate and generate new seeds
        net_params = copy.deepcopy(net.state_dict())
        Mutate_Net(net,sigma)
        Generate_New_Seeds(seeds)
        print('Starting parallel growths',flush=True)
        #Run growths in parallel
        with multiprocessing.Pool(n_pool) as pool:
            simulate_output = pool.starmap(Simulate, seeds)
        
        #Calculate mean score
        scores = [simulate_output[i][0] for i in range(len(simulate_output))]
        mean_score = np.mean(np.array(scores))

        #Accept if new score is higher, reject otherwise
        if mean_score >= cur_score:
            cur_score = mean_score
        else:
            net.load_state_dict(net_params)

        #Saving data
        #Sort networks and scores
        seeds, scores = Sort_Growths(seeds,scores)

        print('Scores: ',scores)
        np.savez_compressed('Results/scores_epoch_%g.npz'%(epoch),score=np.array(scores),mean_score=np.array([mean_score]),cur_score=np.array([cur_score]))

        #Checkpoint every N epochs and at final epoch
        if checkpointAlways or epoch%5 == 0 or epoch == base_epoch+epochs-1:
            Save_Checkpoint(net,seeds,epoch,cur_score)


        tee = time.time()
        print('\nEpoch %g Finished in %.2fmin\ntime remaining: %.2fhr\nmean score:%.2f\ncur score:%.2f\nbest score:%.2f\n'%(epoch,(tee-tes)/60,((tee-tes)/3600)*(base_epoch+epochs-epoch-1),mean_score,cur_score,max(scores)),flush=True)


    t1 = time.time()
    print('Finished job in %.1fhr'%((t1-t0)/3600),flush=True)
