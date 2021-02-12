import torch.nn as nn
import torch
from torchvision import transforms

class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = num_inputs
        
        # self.batch_norm = nn.BatchNorm1d(num_features=nh)
        for nh in hidden_size:
            # self.affine_layers.append(nn.BatchNorm1d(num_features=last_dim))
            # self.affine_layers.append(transforms.Normalize(mean= 0, std= 1))
            # self.affine_layers.append(self.norm)
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.logic = nn.Linear(last_dim, 1)
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    def forward(self, x):
        # print(x.shape)
        for affine in self.affine_layers:
            # x -= torch.min(x)
            # x /= torch.max(x)

            x = self.activation(affine(x))
        
        # print(x)
        prob = torch.sigmoid(self.logic(x))
        return prob