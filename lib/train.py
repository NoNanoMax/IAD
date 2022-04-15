import torchvision
import torch.nn as nn
import torch
from torch.optim import Adam
from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt
import time
import os
from PIL import Image

torch.manual_seed(42)

def train(model, loss_fn, optimizer, epochs, x, y, step, x_test=None, y_test=None):
    model.train()
    ret = 0
    hist = []
    for i in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        ret = loss.item()
        if (i + 1) % step == 0:
            print("epoch = {}, loss = {}".format(i + 1, loss.item()))
            if x_test is None:
                print(40 * '-')
                continue
            model.eval()
            pred = torch.argmax(model(x_test), dim=1)
            acc = torch.sum(pred == y_test).item() / 10000
            print("accuracy = {}".format(acc))
            print(40 * '-')
            hist.append(acc)
    return hist

def train_hist(models, iters, epochs, X, y, step, x_test=None, y_test=None):
    optimizer0 = torch.optim.Adam(models[0].parameters(), lr=0.005)
    optimizer1 = torch.optim.Adam(models[1].parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    for elem in os.listdir("simple_history"):
        os.remove("simple_history/" + elem)
    for i in range(iters):
        print('ITER={}\n'.format(i + 1))
        loss1 = train(models[0], loss_fn, optimizer0, epochs, X, y, step, x_test, y_test)
        loss2 = train(models[1], loss_fn, optimizer1, epochs, X, y, step, x_test, y_test)
        plt.figure(figsize=(10, 10))
        plt.plot(X, y)
        plt.plot(X, models[0](X).reshape(-1).detach().numpy(), label='model1')
        plt.plot(X, models[1](X).reshape(   -1).detach().numpy(), label='model2')
        plt.legend()
        plt.savefig("simple_history/" + "img{}.jpg".format(i))  

def generate_gif(name):

    frames = []
 
    for i in range(len(os.listdir("simple_history"))):
        frame = Image.open('simple_history/img{}.jpg'.format(i))
        frames.append(frame)

    frames[0].save(
        "gifs/" + name,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=300,
        loop=0
    )

