from GrowthPipeline import Initialize_Net,Mutate_Net
from GrowthNetworks import Net

import copy
import torch
torch.set_grad_enabled(False)

net = Net()

Initialize_Net(net)

state_dict = copy.deepcopy(net.state_dict())

print(net.state_dict()['layers.0.weight'][0],state_dict['layers.0.weight'][0])

Mutate_Net(net,0.1)

state_dict_2 = net.state_dict()

print(net.state_dict()['layers.0.weight'][0],state_dict['layers.0.weight'][0],state_dict_2['layers.0.weight'][0])

net.load_state_dict(state_dict)


print(net.state_dict()['layers.0.weight'][0],state_dict['layers.0.weight'][0],state_dict_2['layers.0.weight'][0])


