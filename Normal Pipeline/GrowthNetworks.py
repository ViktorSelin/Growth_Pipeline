import torch.nn as nn
import torch

#Network
class Net(nn.Module):
    def __init__(self,depth=1,width=100,input_size=1,output_size=2):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size,width)]+[nn.Linear(width,width) for i in range(depth-1)])
        #N hidden layers, N = depth-1: width -> width
        self.output = nn.Linear(width,output_size) #Output layer: width -> output_size
        self.act = nn.Tanh()
        self.elu = nn.ELU()
        
    def forward(self, x):
        #Forward pass through all hidden layers with relu activation
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
        x = self.output(x) #Output layer
        x[:,0] = self.elu(x[:,0]) + 1.0
        return x

#Network
class Net_old(nn.Module):
    def __init__(self,depth=1,width=100,input_size=1,output_size=1):
        super(Net_old, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size,width)]+[nn.Linear(width,width) for i in range(depth-1)])
        #N hidden layers, N = depth-1: width -> width
        self.output = nn.Linear(width,output_size) #Output layer: width -> output_size
        self.act = nn.Tanh()
        
    def forward(self, x):
        #Forward pass through all hidden layers with relu activation
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
        x = self.output(x) #Output layer
        return torch.sigmoid(x)*0.50 + 0.05
