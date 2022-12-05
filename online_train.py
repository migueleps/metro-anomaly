import numpy as np
import torch as th
from torch import nn, optim
import pickle as pkl
from LSTMAE import LSTM_AE
from LSTM_SAE import LSTM_SAE
import tqdm
import copy

th.autograd.set_detect_anomaly(True)
INIT_LOOP = 0
END_LOOP = 5

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
EPOCHS = 100
LR = 1e-3
LPF_ALPHA = 0.05
DROPOUT = 0.2
HIDDEN = 32
EMBEDDING = 8
FEATS = "analog_feats"
FEATS_TO_NUMBER = {"analog_feats": 8, "digital_feats": 8, "all_feats": 16}
NFEATS = FEATS_TO_NUMBER[FEATS]

MODELS = {"lstm_ae": LSTM_AE, "lstm_sae": LSTM_SAE}
MODEL_NAME = "lstm_ae"

model = MODELS[MODEL_NAME](NFEATS, EMBEDDING, HIDDEN, DROPOUT, device).to(device)
blacklist = set()

model_string = f"{MODEL_NAME}_{FEATS}_{EMBEDDING}_{HIDDEN}"


with open("online_train_val_test_inds.pkl", "rb") as indspkl:
    train_inds, val_inds, test_inds = pkl.load(indspkl)


def filter_tensors(list_of_cycles, cycle_inds, blacklist):
    first_cycle, _ = cycle_inds # last cycle is meant to not be included, the interval is [start,end)
    tensor_list = []
    for ind, cycle in enumerate(list_of_cycles):
        if ind+first_cycle in blacklist:
            continue
        tensor_list.append(cycle.to(device))
    return tensor_list


def train_model(model, train_tensors, val_tensors, epochs, lr, prev_best_loss):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(reduction="mean").to(device)
    loss_over_time = {"train": [], "val": []}
    best_loss = prev_best_loss
    best_model = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        val_losses = []
        with tqdm.tqdm(train_tensors, unit="cycles") as tepoch:
            for train_tensor in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                optimizer.zero_grad()
                reconstruction = model(train_tensor)
                loss = mse(reconstruction, train_tensor)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
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

        if np.isnan(train_loss) or np.isnan(val_loss):
            print("Found nan in loss")

        loss_over_time['train'].append(train_loss)
        loss_over_time['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        print(f'Epoch {epoch+1}: train loss {train_loss} val loss {val_loss}')

    print(f"Loading model from epoch {best_epoch+1} with validation loss {best_loss}")
    model.load_state_dict(best_model)
    return model, loss_over_time, best_loss


def predict(model, test_tensors, tqdm_desc):
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


def execute_train_test_loop(loop_no, prev_best_loss, model, load_model, blacklist):

    print(f"STARTING LOOP {loop_no+1}")

    with open(f"train_tensors_{loop}_{FEATS}.pkl", "rb") as tensorpkl:
        train_tensors = pkl.load(tensorpkl)

    with open(f"test_tensors_{loop}_{FEATS}.pkl", "rb") as tensorpkl:
        test_tensors = pkl.load(tensorpkl)

    with open(f"val_tensors_{loop}_{FEATS}.pkl", "rb") as tensorpkl:
        val_tensors = pkl.load(tensorpkl)

    if load_model != "":
        model.load_state_dict(th.load(load_model))

    train_tensors = filter_tensors(train_tensors, train_inds[loop_no], blacklist)
    val_tensors = filter_tensors(val_tensors, val_inds[loop_no], blacklist)

    model, loss_over_time, new_best_loss = train_model(model, train_tensors, val_tensors, epochs = EPOCHS, lr = LR, prev_best_loss = prev_best_loss)

    train_losses = predict(model, train_tensors, "Calculating training error distribution")

    test_tensors = filter_tensors(test_tensors, test_inds[loop_no], [])
    test_losses = predict(model, test_tensors, "Testing on new data")

    anomaly_thres = extreme_anomaly(train_losses)

    filtered_test_losses = simple_lowpass_filter(test_losses, LPF_ALPHA)

    anomalies = np.array(filtered_test_losses) > anomaly_thres

    blacklist.update(anomaly_inds(anomalies, test_inds[loop_no]))

    losses_over_time = {"train": train_losses, "test": test_losses, "filtered": filtered_test_losses, "blacklist": blacklist}

    with open(f"online_{loop_no}_losses_{model_string}_{EPOCHS}_{LR}.pkl", "wb") as lossfile:
        pkl.dump(losses_over_time, lossfile)

    th.save(model.state_dict(), f"online_{loop_no}_{model_string}_{EPOCHS}_{LR}.pt")

    return blacklist, new_best_loss


for loop in range(INIT_LOOP, END_LOOP+1):

    model_string = "" if loop == INIT_LOOP else f"online_{loop-1}_{model_string}_{EPOCHS}_{LR}.pt"
    prev_best_loss = best_loss if loop > INIT_LOOP else 10000.
    blacklist, best_loss = execute_train_test_loop(loop, prev_best_loss, model, model_string, blacklist)
