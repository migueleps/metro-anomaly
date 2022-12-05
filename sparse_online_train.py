import numpy as np
import torch as th
from torch import nn, optim
import pickle as pkl
from LSTM_SAE import LSTM_SAE
import tqdm
import copy


sparsity_weight = 0.01
sparsity_parameter = 0.05


def sparsity_loss(target_activation, hidden_outputs, device):
    average_activation = th.mean(hidden_outputs, 1)
    target_activations = th.tensor([target_activation] * average_activation.shape[1]).to(device)
    kl_div_part1 = th.log(target_activations/average_activation)
    kl_div_part2 = th.log((1-target_activations)/(1-average_activation))
    return th.sum(target_activation * kl_div_part1 + (1-target_activation) * kl_div_part2)


def sparsity_l1_loss(hidden_outputs, device):
    

def filter_tensors(list_of_cycles, cycle_inds, blacklist, device):
    first_cycle, _ = cycle_inds # last cycle is meant to not be included, the interval is [start,end)
    tensor_list = []
    for ind, cycle in enumerate(list_of_cycles):
        if ind+first_cycle in blacklist:
            continue
        tensor_list.append(cycle.to(device))
    return tensor_list


def train_model(model, train_tensors, val_tensors, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    mse = nn.MSELoss(reduction="mean").to(device)
    loss_over_time = {"train": [], "val": []}
    best_loss = 100000.0
    best_model = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        train_losses = []
        val_losses = []
        with tqdm.tqdm(train_tensors, unit="cycles") as tepoch:
            for train_tensor in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                optimizer.zero_grad()
                reconstruction, hidden_output = model(train_tensor)
                sparsity_comp = sparsity_loss(sparsity_parameter, hidden_output, device)
                loss = mse(reconstruction, train_tensor) + sparsity_weight * sparsity_comp
                loss.backward()
                #nn.utils.clip_grad_norm_(model.parameters(), 3)
                optimizer.step()
                train_losses.append(loss.item())

        with th.no_grad():
            model.eval()
            for val_tensor in val_tensors:
                reconstruction, hidden_output = model(val_tensor)
                sparsity_comp = sparsity_loss(sparsity_parameter, hidden_output, device)
                loss = mse(reconstruction, val_tensor) + sparsity_weight * sparsity_comp
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        loss_over_time['train'].append(train_loss)
        loss_over_time['val'].append(val_loss)

        #if val_loss < best_loss:
        #    best_loss = val_loss
        #    best_model = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch+1}: train loss {train_loss} val loss {val_loss}')

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
                reconstruction, _ = model(test_tensor)
                loss = mse(reconstruction, test_tensor)
                test_losses.append(loss.item())
    return test_losses


def simple_lowpass_filter(arr, alpha):
    y = arr[0]
    filtered_arr = []
    for elem in arr[1:]:
        y = y + alpha * (elem - y)
        filtered_arr.append(y)
    return filtered_arr


def extreme_anomaly(dist):
    q25, q75 = np.quantile(dist, [0.25,0.75])
    return q75 + 3*(q75-q25)


def anomaly_inds(anomalies, test_inds):
    i_first, i_last = test_inds
    anom_inds = np.where(anomalies)[0]
    return anom_inds + i_first


with open("online_train_val_test_inds.pkl", "rb") as indspkl:
    train_inds, val_inds, test_inds = pkl.load(indspkl)

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
lstm_sae = LSTM_SAE(8, 16, 32, 0.2, device).to(device)

losses_over_time = {}
blacklist = set()

for loop in range(6):

    print(f"STARTING LOOP {loop+1}")

    with open(f"train_tensors_{loop}_analog_feats.pkl", "rb") as tensorpkl:
        train_tensors = pkl.load(tensorpkl)

    with open(f"test_tensors_{loop}_analog_feats.pkl", "rb") as tensorpkl:
        test_tensors = pkl.load(tensorpkl)

    with open(f"val_tensors_{loop}_analog_feats.pkl", "rb") as tensorpkl:
        val_tensors = pkl.load(tensorpkl)

    train_tensors = filter_tensors(train_tensors, train_inds[loop], blacklist, device)
    val_tensors = filter_tensors(val_tensors, val_inds[loop], blacklist, device)
    lstm_sae, loss_over_time = train_model(lstm_sae, train_tensors, val_tensors, epochs = 100, lr = 1e-3, device = device)
    train_losses = predict(lstm_sae, train_tensors, device, "Calculating training error distribution")

    test_tensors = filter_tensors(test_tensors, test_inds[loop], [], device)
    test_losses = predict(lstm_sae, test_tensors, device, "Testing on new data")

    anomaly_thres = extreme_anomaly(train_losses)

    filtered_test_losses = simple_lowpass_filter(test_losses, 0.05)

    anomalies = np.array(filtered_test_losses) > anomaly_thres

    blacklist.update(anomaly_inds(anomalies, test_inds[loop]))

    losses_over_time[loop] = {"train": train_losses, "test": test_losses, "filtered": filtered_test_losses, "blacklist": blacklist}

    with open(f"online_{loop}_losses_lstm_sae_analog_feats_32_16_50_1e-3_nosigmoid.pkl", "wb") as lossfile:
        pkl.dump(losses_over_time, lossfile)

    th.save(lstm_sae.state_dict(), f"online_{loop}_lstm_sae_analog_feats_32_16_50_1e-3_nosigmoid.pt")
