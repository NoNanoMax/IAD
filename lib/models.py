import torchvision
import torch.nn as nn
import torch
from torch.optim import Adam
from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt
import time

class OneLayerModel(nn.Module):

    def __init__(self, input_shape, hidden_shape, out_shape):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([
            ("hidden_layer", nn.Linear(input_shape, hidden_shape)),
            ("norm", nn.BatchNorm1d(hidden_shape)),
            ("relu", nn.ReLU()),
            ("out_layer", nn.Linear(hidden_shape, out_shape))
        ]))

    def forward(self, x):
        return self.net(x)

    def init_weights(self):
        nn.init.kaiming_uniform_(self.net[0].weight)
        nn.init.ones_(self.net[0].bias)
        nn.init.kaiming_uniform_(self.net[-1].weight)
        nn.init.ones_(self.net[-1].bias)

class MultiLayerModel(nn.Module):
    
    def __init__(self, input_shape, out_shape, layers):
        super().__init__()
        list_of_layers = [nn.Linear(input_shape, layers[0]), nn.ReLU()]
        for i in range(1, len(layers)): 
            list_of_layers.append(nn.BatchNorm1d(layers[i - 1]))
            list_of_layers.append(nn.Linear(layers[i - 1], layers[i]))
            list_of_layers.append(nn.ReLU())    
        list_of_layers.append(nn.Linear(layers[-1], out_shape))
        self.net = nn.Sequential(*list_of_layers)
        
    def forward(self, x):
        return self.net(x)

    def init_weights(self):
        for i in range(0, len(self.net) - 1, 3):
            torch.nn.init.kaiming_uniform_(self.net[i].weight, nonlinearity="relu")
            torch.nn.init.normal_(self.net[i].bias)
        torch.nn.init.kaiming_uniform_(self.net[-1].weight)
        torch.nn.init.normal_(self.net[-1].bias)
