from ast import arg
from inspect import ArgSpec

import numpy as np
import torch
import lib.models as models
import lib.train as train
import lib.datasets as data
import argparse

def parse():
    parcer = argparse.ArgumentParser(description='inputs')
    parcer.add_argument(
        '--model1',
        type=str,
        default='OneLayer'
    )
    parcer.add_argument(
        '--layers1',
        type=str,
        default='10'
    )
    parcer.add_argument(
        '--model2',
        type=str,
        default='MultiLayer'
    )
    parcer.add_argument(
        '--layers2',
        type=str,
        default='3,3,3'
    )
    parcer.add_argument(
        '--func',
        type=str,
        default='func'
    )
    parcer.add_argument(
        '--save_gif',
        type=str,
        default='ex1.gif'
    )
    return parcer.parse_args()

if __name__ == '__main__':
    p = parse()
    inn, out = 1, 1

    if p.func == 'MNIST':
        inn = 28 * 28
        out = 10
    if p.model1 == 'OneLayer':
        c1 = int(p.layers1)
        model1 = models.OneLayerModel(inn, c1, out)
    else:
        c1 = list(map(int, p.layers1.split(',')))
        model1 = models.MultiLayerModel(inn, out, c1)
    if p.model2 == 'OneLayer':
        c2 = int(p.layers1)
        model2 = models.OneLayerModel(inn, c2, out)
    else:
        c2 = list(map(int, p.layers2.split(',')))
        model2 = models.MultiLayerModel(inn, out, c2)
    model1.init_weights()
    model2.init_weights()

    if p.func != 'MNIST':
        x = torch.tensor(np.linspace(0, 4, 101))
        if p.func == 'func': y = data.func(x)
        if p.func == 'func1': y = data.func1(x)
        if p.func == 'func2': y = data.func2(x)
        if p.func == 'func3': y = data.func3(x)
        if p.func == 'func4': y = data.func4(x)
        if p.func == 'func5': y = data.func5(x)

        X = x.reshape(-1, 1).to(dtype=torch.float32)
        Y = y.reshape(-1, 1).to(dtype=torch.float32)

        train.train_hist([model1, model2], 20, 200, X, Y, 100)
        train.generate_gif(p.save_gif)

        

