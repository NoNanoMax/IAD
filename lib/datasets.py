import torch
import torchvision
import numpy as np
import os

def load_MNIST():
    data_train = torchvision.datasets.MNIST('datasets', download=False)
    data_test = torchvision.datasets.MNIST('datasets', train=False, download=False)

    X_train = data_train.data.reshape(60000, -1).to(dtype=torch.float32)
    X_test = data_test.data.reshape( 10000, -1).to(dtype=torch.float32)

    Y_train = data_train.targets.to(dtype=torch.long)
    Y_test = data_test.targets.to(dtype=torch.long)

    return {"train": (X_train, Y_train), 
            "test": (X_test, Y_test)}

def func(x):
    por = np.linspace(0, 4, 5)
    ans = torch.zeros_like(x)
    for i in range(1, len(por)):
        ans[(x <= por[i]) * (x > por[i - 1])] = por[i] * 3 
    return ans + torch.rand_like(ans) * 0.01

def func1(x):
    return torch.sin(4 * x) + torch.sqrt(x) + 1 / 10. * torch.exp(x)

def func2(x):
    return torch.sin(x * 3) * x

def func3(x):
    return x**2

def func4(x):
    return x>2

def func5(x):
    return x + 1/2 * x**2 - 1/6 * x**3