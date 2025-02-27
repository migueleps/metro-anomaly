import numpy as np
import torch as th
from torch import nn, optim
import copy
import pickle as pkl
from LSTMAE import LSTM_AE
import tqdm
from EarlyStopper import EarlyStopping

def train_model(model, train_tensors, val_tensors, epochs, lr, device):

    optimizer = optim.Adam(model.parameters(),lr=lr)
    mse = nn.MSELoss(reduction="mean").to(device)
    loss_over_time = {"train": [], "val": []}

    best_model = copy.deepcopy(model.state_dict())
    best_loss = 100000.0

    #early_stopper = EarlyStopping(3, 1e-3, 1e-4)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        with tqdm.tqdm(train_tensors, unit="example") as tepoch:
            for train_tensor in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                optimizer.zero_grad()

                reconstruction = model(train_tensor)
                loss = mse(reconstruction,train_tensor)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                train_losses.append(loss.item())

        val_losses = []

        model.eval()
        with th.no_grad():
            for val_tensor in val_tensors:
                reconstruction = model(val_tensor)
                loss = mse(reconstruction,val_tensor)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        loss_over_time['train'].append(train_loss)
        loss_over_time['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch+1}: train loss {train_loss} val loss {val_loss}')

        #if early_stopper.stopping_condition(val_loss):
        #    print("Early stopping condition met, stopping training")
        #    break

    model.load_state_dict(best_model)
    return model.eval(), loss_over_time


with open("train_tensors_analog_feats.pkl", "rb") as pklfile:
    train_tensors = pkl.load(pklfile)

with open("val_tensors_analog_feats.pkl", "rb") as pklfile:
    val_tensors = pkl.load(pklfile)

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

for i in range(len(train_tensors)):
    train_tensors[i] = train_tensors[i].to(device)

for i in range(len(val_tensors)):
    val_tensors[i] = val_tensors[i].to(device)


lstm_ae = LSTM_AE(8, 8, 32, 0.2, device).to(device)
lstm_ae, loss_over_time = train_model(lstm_ae, train_tensors, val_tensors, epochs = 100, lr = 1e-3, device = device)

with open("lstm_ae_analog_8_32_100_1e-3.pkl", "wb") as modelpkl:
    pkl.dump(lstm_ae, modelpkl)

with open("lstm_ae_analog_8_32_100_1e-3_losses.pkl", "wb") as lossespkl:
    pkl.dump(loss_over_time, lossespkl)
