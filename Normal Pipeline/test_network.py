from GrowthNetworks import Net
import torch

torch.set_grad_enabled(False)

net = Net()

net.output.bias[0] -= 1
net.output.bias[1] -= 2.5



test_input = torch.tensor([[0],[0.5],[1]])
print(test_input)
print(net(test_input))


