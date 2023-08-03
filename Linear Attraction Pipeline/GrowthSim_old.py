#Imports
import numpy as np
import numba as nb
import math
from numba.experimental import jitclass
from numba import jit
from numba.types import UniTuple
from typing import List
from numba.typed import List as NumbaList


'''
Main Growth class 

Args:
dim: dimension of lattice (dim x dim)
n_nano: number of nanoparticles to place
KbT: temperature * Kb
mu: chemical potential
e_nn: nano-nano attraction
e_nl: nano-solvent attraction
e_ll: solvent-solvent attraction -- All other attractions and mu are given in relation to this, leave as 1
seed: random seed for rng generator
nano_mob: number of nanoparticle cycles to perform per solvent cycle
nano_size: size of nanoparticles (nano_size x nano_size)

Use:
growth = Growth(args)
growth.initialize_nano
for range(n_epochs):
  growth.step_simulation()
'''
@jitclass
class Growth_NonPeriodic:
    #Type annotation for Numba
    x_dim: int
    y_dim: int
    n_nano: int
    n_nano_placed: int
    nano_size: int
    fluid: nb.int64[:,:]
    nano: nb.int64[:,:]
    nano_list: List[UniTuple(nb.int64,2)]
    total_energy: float
    e_ll: float
    e_nl: float
    e_nn: float
    mu: float
    nano_mob: int
    solv_iter: int
    KbT: float
    seed: int
        
    def __init__(self, x_dim, y_dim, n_nano, KbT, mu, e_nn, e_nl, e_ll, nano_mob, nano_size, seed):
        #Keep parameters
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_nano = n_nano
        self.n_nano_placed = 0
        self.seed = seed
        self.nano_size = nano_size
        
        #Initialize RNG seed
        np.random.seed(self.seed)
        
        #Initialize lattices, padded with an extra entry in each direction
        self.fluid = np.ones((self.x_dim,self.y_dim),dtype=np.int64)
        self.nano = np.zeros((self.x_dim,self.y_dim),dtype=np.int64)
        #Initialize list of nanoparticles
        self.nano_list = NumbaList([(1,2) for x in range(0)])
        
        #Initialize prev energy
        self.total_energy = 0
        
        #Initialize constants
        #Liquid - liquid attraction
        self.e_ll = e_ll
        #Nano - nano attraction
        self.e_nn = e_nn*self.e_ll
        #Nano - liquid attraction
        self.e_nl = e_nl*self.e_ll
        #Chemical potential
        self.mu = mu*self.e_ll
        
        #Nano mobility and per step solvent iterations
        self.nano_mob = nano_mob
        self.solv_iter = x_dim*y_dim
        
        #Boltzmann*Temperature
        self.KbT = self.e_ll*KbT
        
    '''
    Old function to calculate energy
    Calculates total energy on entire lattice
    '''
    #Function that multiplies all elements with neighbours and sums
    def sum_neighbour_energy(self,A,B):
        return np.sum(A*(np.roll(B,1,0) + np.roll(B,-1,0) + np.roll(B,1,1) + np.roll(B,-1,1)))

    #Slow way of calcuating total energy
    def calculate_total_energy(self,nano,fluid):
        total_energy = 0

        #Liquid - liquid contribution
        total_energy -= self.e_ll * self.sum_neighbour_energy(fluid,fluid) / 2

        #Nano - nano contribution
        total_energy -= self.e_nn * self.sum_neighbour_energy(nano,nano) / 2

        #Nano - liquid contribution
        total_energy -= self.e_nl * self.sum_neighbour_energy(nano,fluid)

        #Liquid phase contribution
        total_energy -= self.mu * np.sum(fluid)

        return total_energy
        
        
        
        
    '''
    Function that performs a single solvent step
    Chooses a random lattice site and attempts to change phase
    '''
    def step_fluid(self):
        #Choose random lattice cell
        x_i = np.random.randint(1,self.x_dim-1)
        y_i = np.random.randint(1,self.y_dim-1)

        #Only proceed if no nano in cell
        delta_e = 0
        norm = 1/(1+1/math.sqrt(2))
        if self.nano[x_i,y_i] == 0:

            #Calculate change in energy
            delta_e = -(1-2*self.fluid[x_i,y_i])*((norm)*(self.e_ll*(self.fluid[(x_i-1),y_i]+
                                                             self.fluid[(x_i+1),y_i]+
                                                             self.fluid[x_i,(y_i-1)]+
                                                             self.fluid[x_i,(y_i+1)]+
                                                             (1/math.sqrt(2))*
                                                             (self.fluid[(x_i-1),(y_i-1)]+
                                                             self.fluid[(x_i+1),(y_i-1)]+
                                                             self.fluid[(x_i-1),(y_i+1)]+
                                                             self.fluid[(x_i+1),(y_i+1)]))
                                                 +self.e_nl*(self.nano[(x_i-1),y_i]+
                                                             self.nano[(x_i+1),y_i]+
                                                             self.nano[x_i,(y_i-1)]+
                                                             self.nano[x_i,(y_i+1)]+
                                                             (1/math.sqrt(2))*
                                                             (self.nano[(x_i-1),(y_i-1)]+
                                                             self.nano[(x_i+1),(y_i-1)]+
                                                             self.nano[(x_i-1),(y_i+1)]+
                                                             self.nano[(x_i+1),(y_i+1)])))
                                                 +self.mu)



            #Calculate probability of accepting move
            Pacc = min(1,np.exp(-delta_e/self.KbT))

            if np.random.random() <= Pacc:
                #Accept move - update total energy
                self.total_energy += delta_e
                
                #Change solvent phase
                self.fluid[x_i,y_i] = 1-self.fluid[x_i,y_i]

        return delta_e

    '''
    Function that calculates the change in bond energy for a nanoparticle move
    '''
    def neigh_de(self,cell,dis):
        x_i = cell[0]
        y_i = cell[1]
        x_i_neigh = (cell[0]+dis[0])
        y_i_neigh = (cell[1]+dis[1])
        
        #Calculate bond energy change
        de = (1-2*self.fluid[x_i,y_i])*\
                (self.e_ll*(self.fluid[x_i_neigh,y_i_neigh])+\
                 self.e_nl*(self.nano[x_i_neigh,y_i_neigh]))+\
             (1-2*self.nano[x_i,y_i])*\
                (self.e_nn*(self.nano[x_i_neigh,y_i_neigh])+\
                 self.e_nl*(self.fluid[x_i_neigh,y_i_neigh]))
        return -de
    
    '''
    Function that calculates total energy using neighbour bonds
    '''
    def calculate_total_energy_neigh(self):
        total_e = 0
        diss = [(1,0),(-1,0),(0,1),(0,-1)]
        nneigh = [(1,1),(-1,1),(1,-1),(-1,-1)]
        for i in range(1,self.x_dim-1):
            for j in range(1,self.y_dim-1):
                for k in range(4):
                    dis = diss[k]
                    neigh = nneigh[k]
                    x = (i+dis[0])
                    y = (j+dis[1])
                    x_n = (i+neigh[0])
                    y_n = (j+neigh[1])
                    total_e -= self.fluid[i,j]*\
                                ((1/(1+1/math.sqrt(2)))*(self.e_ll*(self.fluid[x,y]+(1/math.sqrt(2))*self.fluid[x_n,y_n])/2 +\
                                 self.e_nl*(self.nano[x,y]+(1/math.sqrt(2))*self.nano[x_n,y_n])) +\
                                 self.mu/4)+\
                               self.nano[i,j]*\
                                 ((1/(1+1/math.sqrt(2)))*self.e_nn*(self.nano[x,y]+(1/math.sqrt(2))*self.nano[x_n,y_n])/2)
        return total_e
                

    '''
    Function that performs a single nanoparticle step
    Chooses a random nanoparticle and attempts to move it in a random direction
    '''
    def step_nano(self):
        #Select nano particle to move
        i_nano = np.random.randint(0,len(self.nano_list))
        
        x_nano = self.nano_list[i_nano][0]
        y_nano = self.nano_list[i_nano][1]
        
        #Select displacement direction
        dir_nano = np.random.randint(0,4)
        hitBoundary = False
        match dir_nano:
            case 0: #+y
                dis = (0,1)
                offset = (0,self.nano_size)
                wake_offset = (0,0)
                hitBoundary = y_nano >= (self.y_dim-self.nano_size-1)		
            case 1: #-y
                dis = (0,-1)
                offset = (0,-1)
                wake_offset = (0,self.nano_size-1)
                hitBoundary = y_nano <= 1
            case 2: #+x
                dis = (1,0)
                offset = (self.nano_size,0)
                wake_offset = (0,0)
                hitBoundary = x_nano >= (self.x_dim-self.nano_size-1)
            case 3: #-x
                dis = (-1,0)
                offset = (-1,0)
                wake_offset = (self.nano_size-1,0)
                hitBoundary = x_nano <= 1

        fluid_sum = 0
        nano_sum = 0
        for i in range(self.nano_size):
            fluid_sum += self.fluid[(x_nano+offset[0]+i*abs(dis[1])),(y_nano+offset[1]+i*abs(dis[0]))]
            nano_sum += self.nano[(x_nano+offset[0]+i*abs(dis[1])),(y_nano+offset[1]+i*abs(dis[0]))]
        
        delta_e = 0
        #Move only if no nanoparticles blocking and all cells are occupied by fluid
        if not hitBoundary and fluid_sum == self.nano_size and nano_sum == 0:

            #Get cell and wake cell indices
            x = (x_nano + offset[0])
            y = (y_nano + offset[1]) 
            x_wake = (x_nano + wake_offset[0])
            y_wake = (y_nano + wake_offset[1])         

            for i in range(self.nano_size):
                #Get indices of cells
                x_i = (x+i*abs(dis[1]))
                y_i = (y+i*abs(dis[0]))
                x_i_wake = (x_wake+i*abs(dis[1]))
                y_i_wake = (y_wake+i*abs(dis[0]))
                
                #Add needed bond contributions - nearest neighbours
                delta_e += self.neigh_de((x_i,y_i),dis)
                delta_e += self.neigh_de((x_i_wake,y_i_wake),(-dis[0],-dis[1]))
                
                #Second nearest neighbours
                delta_e += (1/math.sqrt(2))*self.neigh_de((x_i,y_i),(dis[0]+dis[1],dis[1]+dis[0]))
                delta_e += (1/math.sqrt(2))*self.neigh_de((x_i,y_i),(dis[0]-dis[1],dis[1]-dis[0]))
                
                delta_e += (1/math.sqrt(2))*self.neigh_de((x_i_wake,y_i_wake),(-dis[0]+dis[1],-dis[1]+dis[0]))                
                delta_e += (1/math.sqrt(2))*self.neigh_de((x_i_wake,y_i_wake),(-dis[0]-dis[1],-dis[1]-dis[0]))
                
                if i == 0:
                    delta_e += self.neigh_de((x_i,y_i),(-abs(dis[1]),-abs(dis[0])))
                    delta_e += self.neigh_de((x_i_wake,y_i_wake),(-abs(dis[1]),-abs(dis[0])))
                    
                    delta_e += (1/math.sqrt(2))*self.neigh_de((x_i,y_i),(-dis[0]-abs(dis[1]),-dis[1]-abs(dis[0])))
                    delta_e += (1/math.sqrt(2))*self.neigh_de((x_i_wake,y_i_wake),(dis[0]-abs(dis[1]),dis[1]-abs(dis[0])))                    
                elif i == (self.nano_size-1):
                    delta_e += self.neigh_de((x_i,y_i),(abs(dis[1]),abs(dis[0])))
                    delta_e += self.neigh_de((x_i_wake,y_i_wake),(abs(dis[1]),abs(dis[0])))
                    
                    delta_e += (1/math.sqrt(2))*self.neigh_de((x_i,y_i),(-dis[0]+abs(dis[1]),-dis[1]+abs(dis[0])))
                    delta_e += (1/math.sqrt(2))*self.neigh_de((x_i_wake,y_i_wake),(dis[0]+abs(dis[1]),dis[1]+abs(dis[0])))
            delta_e *= (1/(1+1/math.sqrt(2)))
            #Calculate probability of accepting move
            Pacc = min(1,np.exp(-delta_e/self.KbT))
            if np.random.random() <= Pacc:
                #Accept 
                self.total_energy += delta_e
                #Move nanoparticle
                for i in range(self.nano_size):
                    x_i = (x_nano + offset[0] + i*abs(dis[1]))
                    y_i = (y_nano + offset[1] + i*abs(dis[0]))
                    x_i_wake = (x_nano + wake_offset[0] + i*abs(dis[1]))
                    y_i_wake = (y_nano + wake_offset[1] + i*abs(dis[0]))
                    self.fluid[x_i,y_i] = (1-self.fluid[x_i,y_i])
                    self.fluid[x_i_wake,y_i_wake] = (1-self.fluid[x_i_wake,y_i_wake])
                    self.nano[x_i,y_i] = (1-self.nano[x_i,y_i])
                    self.nano[x_i_wake,y_i_wake] = (1-self.nano[x_i_wake,y_i_wake])

                self.nano_list[i_nano] = ((x_nano+dis[0]),(y_nano+dis[1]))
        return delta_e

    '''
    Function that populates nanoparticle lattice with a number of nanoparticles
    Only attempts 100 random placements - can result in a lower fraction
    '''
    #Randomly populate nanoparticles
    def initialize_nano(self):
        for i in range(self.n_nano):
            tries = 0
            isdone = False
            while not isdone:
                tries += 1
                
                x_i = np.random.randint(1,self.x_dim-(self.nano_size))
                y_i = np.random.randint(1,self.y_dim-(self.nano_size))
                
                nano_sum = 0
                
                for i in range(self.nano_size):
                    for j in range(self.nano_size):
                        nano_sum += self.nano[(x_i+i),(y_i+j)]
                #Only place if no intersection
                if nano_sum == 0:
                    for i in range(self.nano_size):
                        for j in range(self.nano_size):
                            self.nano[(x_i+i),(y_i+j)] = 1
                            self.fluid[(x_i+i),(y_i+j)] = 0
                    self.nano_list.append((x_i,y_i))
                    self.n_nano_placed += 1
                    isdone = True
                elif tries > 100:
                    isdone = True
        self.total_energy = self.calculate_total_energy_neigh()
    
    '''
    Function that performs a single epoch
    '''
    def step_simulation(self):
        for i in range(self.solv_iter):
            self.step_fluid()
        for j in range(self.nano_mob):
            for i in range(self.n_nano_placed):
                self.step_nano()
