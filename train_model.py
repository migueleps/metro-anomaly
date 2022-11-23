import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn, optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import copy
import time
import pickle as pkl
from LSTMAE import LSTM_AE

def train_model(model, train_tensors, val_tensors, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(),lr=lr)
    mse = nn.MSELoss(reduction="mean").to(device)
    loss_over_time = {"train": [], "val": []}

    best_model = copy.deepcopy(model.state_dict())
    best_loss = 100000.0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for train_tensor in train_tensors:
            #start = time.time()
            optimizer.zero_grad()

            reconstruction = model(train_tensor.to(device))
            loss = mse(reconstruction,train_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            #print(time.time() - start, end=" ")

        val_losses = []

        model.eval()
        with th.no_grad():
            for val_tensor in val_tensors:
                reconstruction = model(val_tensor.to(device))
                loss = mse(reconstruction,val_tensor)
                #print(loss.item())
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        loss_over_time['train'].append(train_loss)
        loss_over_time['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch+1}: train loss {train_loss} val loss {val_loss}')
    model.load_state_dict(best_model)
    return model.eval(), loss_over_time


with open("train_tensors_analog_feats.pkl", "rb") as pklfile:
    train_tensors = pkl.load(pklfile)

with open("val_tensors_analog_feats.pkl", "rb") as pklfile:
    val_tensors = pkl.load(pklfile)

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

lstm_ae = LSTM_AE(8, 8, 32, 0.2, device).to(device)
lstm_ae, loss_over_time = train_model(lstm_ae, train_tensors, val_tensors,epochs = 150 , lr = 1e-3)

with open("lstm_ae_analog_8_32_150_1e-3.pkl", "wb") as modelpkl:
    pkl.dump(lstm_ae,modelpkl)

with open("lstm_ae_analog_8_32_150_1e-3_losses.pkl", "wb") as lossespkl:
    pkl.dump(loss_over_time,lossespkl)
