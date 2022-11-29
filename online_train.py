import numpy as np
import pandas as pd
import torch as th
from torch import nn, optim
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from LSTMAE import LSTM_AE
import tqdm


def generate_cycles(df):
    comp_change = list((t := df.COMP.diff().eq(-1))[t == True].index)
    return [[comp_change[i], comp_change[i+1]] for i in range(len(comp_change)-1)]


def test_and_train_dates(df):
    init_train_s = df.timestamp_day[0]
    init_train_e = df.timestamp_day[0] + pd.offsets.DateOffset(months=1)
    init_test_s = init_train_e
    init_test_e = init_test_s + pd.offsets.DateOffset(weeks=1)

    train_dates = [[init_train_s, init_train_e]]
    test_dates = [[init_test_s, init_test_e]]

    last_date = df.timestamp_day.iloc[-1]
    prev_test_end = init_test_e
    while prev_test_end < last_date:
        new_test_end = prev_test_end + pd.offsets.DateOffset(weeks=1)
        new_train_start = prev_test_end - pd.offsets.DateOffset(months=1)
        train_dates.append([new_train_start,prev_test_end])
        test_dates.append([prev_test_end,new_test_end])
        prev_test_end = new_test_end
    return train_dates, test_dates


def match_cycles_to_dates(cycle, df):
    cycle_start, cycle_end = cycle
    cycle_start_date = df.iloc[cycle_start, :].timestamp_day
    cycle_end_date = df.iloc[cycle_end, :].timestamp_day
    return [cycle_start_date, cycle_end_date]


def test_and_train_cycles(cycle_dates, train_dates, test_dates):
    train_inds, test_inds = [], []
    for j in range(len(train_dates)):
        t_start, t_end = train_dates[j]
        test_s, test_e = test_dates[j]
        i = 0
        while cycle_dates[i][0] < t_start:
            i += 1
        train_start_ind = i
        while cycle_dates[i][0] < t_end:
            i += 1
        train_end_ind = i
        test_start_ind = train_end_ind
        while i < len(cycle_dates) and cycle_dates[i][0] < test_e:
            i += 1
        test_end_ind = i
        train_inds.append([train_start_ind, train_end_ind])
        test_inds.append([test_start_ind, test_end_ind])
    return train_inds, test_inds


def yield_cycles(df, cycle_inds, cols, blacklist, device):
    scaler = StandardScaler()
    first_cycle, last_cycle = cycle_inds # last cycle is meant to not be included, the interval is [start,end)
    tensor_list = []
    for cycle in range(first_cycle, last_cycle):
        if cycle in blacklist:
            continue
        i_s, i_f = all_cycles[cycle]
        df_slice = df.iloc[i_s:i_f, :]
        df_slice = df_slice.loc[:, cols]
        df_slice[df_slice.columns] = scaler.fit_transform(df_slice[df_slice.columns])
        tensor_chunk = th.tensor(df_slice.values).unsqueeze(0).float().to(device)
        tensor_list.append(tensor_chunk)
    return tensor_list


def train_model(model, train_tensors, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(reduction="mean").to(device)
    loss_over_time = {"train": []}
    for epoch in range(epochs):
        model.train()
        train_losses = []
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

        train_loss = np.mean(train_losses)
        loss_over_time['train'].append(train_loss)
        print(f'Epoch {epoch+1}: train loss {train_loss}')
        if train_loss == np.nan:
            exit(1)
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


metro = pd.read_csv("dataset_train.csv")
metro["timestamp"] = pd.to_datetime(metro["timestamp"],dayfirst=True)
metro["timestamp_day"] = metro.timestamp - np.timedelta64(2,"h")-np.timedelta64(1,"s")

all_cycles = generate_cycles(metro)
train_dates, test_dates = test_and_train_dates(metro)
all_cycles_dates = list(map(lambda x: match_cycles_to_dates(x,metro), all_cycles))
train_inds, test_inds = test_and_train_cycles(all_cycles_dates, train_dates, test_dates)

analog_sensors = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Flowmeter', 'Motor_current']

digital_sensors = ['COMP', 'DV_eletric', 'Towers', 'MPG', 'LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses']
all_sensors = analog_sensors + digital_sensors

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
lstm_ae = LSTM_AE(8, 8, 32, 0.2, device).to(device)

losses_over_time = {}
blacklist = set()

for loop in range(len(train_inds)):
    train_tensors = yield_cycles(metro,train_inds[loop], analog_sensors, blacklist, device)
    lstm_ae, loss_over_time = train_model(lstm_ae, train_tensors, epochs = 100, lr = 1e-3, device = device)
    train_losses = predict(lstm_ae, train_tensors, device, "Calculating training error distribution")

    test_tensors = yield_cycles(metro, test_inds[loop], analog_sensors, [], device)
    test_losses = predict(lstm_ae, test_tensors, device, "Testing on new data")

    anomaly_thres = extreme_anomaly(train_losses)
    anomalies = np.array(test_losses) > anomaly_thres

    blacklist.update(anomaly_inds(anomalies,test_inds[loop]))

    losses_over_time[loop] = {"train": train_losses, "test": test_losses}

with open("online_losses_lstm_ae_analog_feats_8_32_100_1e-4.pkl", "rb") as lossfile:
    pkl.dump(losses_over_time, lossfile)

th.save(lstm_ae.state_dict(), "online_lstm_ae_analog_feats_8_32_100_1e-4.pkl")
