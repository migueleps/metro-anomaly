import pandas as pd
import numpy as np
import torch as th
from sklearn.preprocessing import StandardScaler
import pickle as pkl


def get_tensors(df, time_window, cols, step_size=20):
    scaler = StandardScaler()
    time_start, time_end = time_window
    df_red = df.loc[(df.timestamp_day >= time_start) & (df.timestamp_day < time_end), cols].reset_index(drop=True)
    index_start, index_end = min(df_red.index), max(df_red.index)
    tensor_list = []
    for i in range(index_start, index_end, step_size):
        normalized_values = scaler.fit_transform(df_red.iloc[i:i+120, :].values)
        tensor_chunk = th.tensor(normalized_values).float()
        if tensor_chunk.shape[0] < 120:
            continue
        tensor_list.append(tensor_chunk)
    return tensor_list


metro = pd.read_csv("data/dataset_train.csv")
metro["timestamp"] = pd.to_datetime(metro["timestamp"], dayfirst=True)
metro["timestamp_day"] = metro.timestamp - np.timedelta64(2, "h")-np.timedelta64(1, "s")

analog_sensors = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs',
                  'Oil_temperature', 'Flowmeter', 'Motor_current']

digital_sensors = ['COMP', 'DV_eletric',
                   'Towers', 'MPG', 'LPS', 'Pressure_switch', 'Oil_level',
                   'Caudal_impulses']
all_sensors = analog_sensors + digital_sensors

init_train_s = metro.timestamp_day[0] + pd.offsets.DateOffset(days=2)
init_train_e = metro.timestamp_day[0] + pd.offsets.DateOffset(months=1)
init_val_s = metro.timestamp_day[0]
init_val_e = metro.timestamp_day[0] + pd.offsets.DateOffset(days=2)
init_test_s = init_train_e
init_test_e = init_test_s + pd.offsets.DateOffset(weeks=1)

train_dates = [[init_train_s, init_train_e]]
test_dates = [[init_test_s, init_test_e]]
val_dates = [init_val_s, init_val_e]

last_date = metro.timestamp_day.iloc[-1]
prev_test_end = init_test_e
while prev_test_end < last_date:
    new_test_end = prev_test_end + pd.offsets.DateOffset(weeks=1)
    new_train_start = prev_test_end - pd.offsets.DateOffset(months=1)
    train_dates.append([new_train_start, prev_test_end])
    test_dates.append([prev_test_end, new_test_end])
    prev_test_end = new_test_end

val_tensors = get_tensors(metro, val_dates, analog_sensors, step_size=40)
with open(f"data/val_tensors_1min_chunks_offline_analog_feats.pkl", "wb") as tensorpkl:
    pkl.dump(val_tensors, tensorpkl)

train_tensors = get_tensors(metro, train_dates[0], analog_sensors, step_size=40)
with open(f"data/train_tensors_1min_chunks_offline_analog_feats.pkl", "wb") as tensorpkl:
    pkl.dump(train_tensors, tensorpkl)

for loop in range(len(test_dates)):
    test_tensors = get_tensors(metro, test_dates[loop], analog_sensors, step_size=40)
    print(len(test_tensors))
    with open(f"data/test_tensors_1min_chunks_{loop}_analog_feats.pkl", "wb") as tensorpkl:
        pkl.dump(test_tensors, tensorpkl)
