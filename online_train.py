import numpy as np
import torch as th
from torch import nn, optim
import pickle as pkl
from LSTMAE import LSTM_AE
import tqdm
import copy


def yield_tensors(list_of_cycles, cycle_inds, blacklist, device):
    first_cycle, last_cycle = cycle_inds # last cycle is meant to not be included, the interval is [start,end)
    tensor_list = []
    for cycle in range(first_cycle, last_cycle):
        if cycle in blacklist:
            continue
        tensor_list.append(list_of_cycles[cycle].to(device))
    return tensor_list


def train_model(model, train_tensors, val_tensors, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(reduction="mean").to(device)
    loss_over_time = {"train": [], "val": []}
    best_loss = 100000.0
    best_model = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        train_losses = []
        val_losses = []
        with tqdm.tqdm(train_tensors, unit="examples") as tepoch:
            for train_tensor in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                optimizer.zero_grad()
                reconstruction = model(train_tensor)
                loss = mse(reconstruction, train_tensor)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3)
                optimizer.step()
                train_losses.append(loss.item())

        with th.no_grad():
            model.eval()
            for val_tensor in val_tensors:
                reconstruction = model(val_tensor)
                loss = mse(reconstruction, val_tensor)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        loss_over_time['train'].append(train_loss)
        loss_over_time['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch+1}: train loss {train_loss} val loss {val_loss}')
        if train_loss == np.nan:
            exit(1)

    model.load_state_dict(best_model)
    return model, loss_over_time


def predict(model, test_tensors, device, tqdm_desc):
    mse = nn.MSELoss(reduction="mean").to(device)
    test_losses = []
    with th.no_grad():
        model.eval()
        with tqdm.tqdm(test_tensors, unit="examples") as tepoch:
            for test_tensor in tepoch:
                tepoch.set_description(tqdm_desc)
                reconstruction = model(test_tensor)
                loss = mse(reconstruction, test_tensor)
                test_losses.append(loss.item())
    return test_losses


def extreme_anomaly(dist):
    q25, q75 = np.quantile(dist, [0.25,0.75])
    return q75 + 3*(q75-q25)


def anomaly_inds(anomalies, test_inds):
    i_first, i_last = test_inds
    anom_inds = np.where(anomalies)[0]
    return anom_inds + i_first


with open("all_tensors_analog_feats.pkl", "rb") as tensorpkl:
    all_tensors = pkl.load(tensorpkl)

with open("online_train_val_test_inds.pkl", "rb") as indspkl:
    train_inds, val_inds, test_inds = pkl.load(indspkl)

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
lstm_ae = LSTM_AE(8, 8, 32, 0.2, device).to(device)

losses_over_time = {}
blacklist = set()

for loop in range(len(train_inds)):
    print(f"STARTING LOOP {loop+1}")
    train_tensors = yield_tensors(all_tensors, train_inds[loop], blacklist, device)
    val_tensors = yield_tensors(all_tensors, val_inds[loop], blacklist, device)
    lstm_ae, loss_over_time = train_model(lstm_ae, train_tensors, val_tensors, epochs = 100, lr = 1e-4, device = device)
    train_losses = predict(lstm_ae, train_tensors, device, "Calculating training error distribution")

    test_tensors = yield_tensors(all_tensors, test_inds[loop], [], device)
    test_losses = predict(lstm_ae, test_tensors, device, "Testing on new data")

    anomaly_thres = extreme_anomaly(train_losses)
    anomalies = np.array(test_losses) > anomaly_thres

    blacklist.update(anomaly_inds(anomalies,test_inds[loop]))

    losses_over_time[loop] = {"train": train_losses, "test": test_losses}

with open("online_losses_lstm_ae_analog_feats_8_32_100_1e-4.pkl", "rb") as lossfile:
    pkl.dump(losses_over_time, lossfile)

th.save(lstm_ae.state_dict(), "online_lstm_ae_analog_feats_8_32_100_1e-4.pkl")
