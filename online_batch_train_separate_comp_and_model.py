import numpy as np
import torch as th
from torch import nn, optim
import pickle as pkl
from LSTMAE_mini_batch import LSTM_AE
from LSTM_SAE_mini_batch import LSTM_SAE
from torch.nn.utils.rnn import pack_padded_sequence
import tqdm
import copy

#th.autograd.set_detect_anomaly(True)
INIT_LOOP = 0
END_LOOP = 17

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
EPOCHS = 100
LR = 1e-3
DROPOUT = 0.2
EMBEDDING = 64
BATCH_SIZE = 32
LSTM_LAYERS = 2
sparsity_weight = 1
sparsity_parameter = 0.05
FEATS = "analog_feats"
FEATS_TO_NUMBER = {"analog_feats": 8, "digital_feats": 8, "all_feats": 16}
NFEATS = FEATS_TO_NUMBER[FEATS]

MODELS = {"lstm_ae": LSTM_AE, "lstm_sae": LSTM_SAE}
MODEL_NAME = "lstm_sae"

results_folder = "results/"
data_folder = "data/"

model_comp0 = MODELS[MODEL_NAME](NFEATS, EMBEDDING,  DROPOUT, LSTM_LAYERS, device, sparsity_weight, sparsity_parameter).to(device)
model_comp1 = MODELS[MODEL_NAME](NFEATS, EMBEDDING,  DROPOUT, LSTM_LAYERS, device, sparsity_weight, sparsity_parameter).to(device)
model_string = f"{MODEL_NAME}_{FEATS}_{EMBEDDING}"

blacklist = set()
if INIT_LOOP > 0:
    try:
        with open(f"{results_folder}online_{INIT_LOOP-1}_losses_{model_string}_{EPOCHS}_{LR}.pkl", "wb") as lossfile:
            loss_over_time = pkl.load(lossfile)
            blacklist = loss_over_time["blacklist"]
    except:
        pass

with open(f"{data_folder}online_train_val_test_inds.pkl", "rb") as indspkl:
    train_inds, val_inds, test_inds = pkl.load(indspkl)


def create_batch(tensor_list, batch_size):
    packed_batches = []
    for batch_number in range(0,len(tensor_list),batch_size):
        tensors_to_batch = tensor_list[batch_number:(batch_number+batch_size)]
        batch_tensor_lengths = th.tensor([tensor.shape[1] for tensor in tensors_to_batch])
        longest_seq = max(batch_tensor_lengths)
        mini_batch = []
        for tensor in tensors_to_batch:
            tensor = tensor.squeeze()
            padded_tensor = th.cat([tensor, th.zeros(longest_seq - tensor.shape[0], tensor.shape[1]).to(device)])
            mini_batch.append(padded_tensor)
        tensor_mini_batch = th.stack(mini_batch)
        packed = pack_padded_sequence(tensor_mini_batch, batch_tensor_lengths, batch_first=True, enforce_sorted=False)
        packed_batches.append(packed.to(device))
    return packed_batches


def filter_tensors(list_of_cycles, cycle_inds, blacklist):
    first_cycle, _ = cycle_inds # last cycle is meant to not be included, the interval is [start,end)
    tensor_list = []
    for ind, cycle in enumerate(list_of_cycles):
        if ind+first_cycle in blacklist:
            continue
        tensor_list.append(cycle.to(device))
    return tensor_list


def train_model(model_comp0, model_comp1, batch_train_tensors_comp0, batch_train_tensors_comp1,
                batch_val_tensors_comp0, batch_val_tensors_comp1, epochs, lr, prev_best_loss):
    optimizer = optim.Adam(list(model_comp0.parameters()) + list(model_comp1.parameters()), lr=lr)
    loss_over_time = {"train": [], "val": []}
    best_loss = prev_best_loss
    #best_model = copy.deepcopy(model.state_dict())
    #best_epoch = 0

    for epoch in range(epochs):
        model_comp0.train()
        model_comp1.train()
        train_losses = []
        val_losses = []
        with tqdm.tqdm(range(len(batch_train_tensors_comp0)), unit="cycles") as tepoch:
            for i in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                optimizer.zero_grad()
                batch_loss_comp0, _ = model_comp0(batch_train_tensors_comp0[i])
                batch_loss_comp1, _ = model_comp1(batch_train_tensors_comp1[i])
                batch_loss = batch_loss_comp0 + batch_loss_comp1
                batch_loss.backward()
                nn.utils.clip_grad_norm_(list(model_comp0.parameters()) + list(model_comp1.parameters()), 1)
                optimizer.step()
                train_losses.append(batch_loss.item())

        with th.no_grad():
            model_comp0.eval()
            model_comp1.eval()
            for i in range(len(batch_val_tensors_comp0)):
                batch_loss_comp0, _ = model_comp0(batch_val_tensors_comp0[i])
                batch_loss_comp1, _ = model_comp1(batch_val_tensors_comp1[i])
                batch_loss = batch_loss_comp0 + batch_loss_comp1
                val_losses.append(batch_loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        loss_over_time['train'].append(train_loss)
        loss_over_time['val'].append(val_loss)

        #if val_loss < best_loss:
        #    best_loss = val_loss
        #    best_model = copy.deepcopy(model.state_dict())
        #    best_epoch = epoch

        print(f'Epoch {epoch+1}: train loss {train_loss} val loss {val_loss}')

    #print(f"Loading model from epoch {best_epoch+1} with validation loss {best_loss}")
    #model.load_state_dict(best_model)
    return model_comp0, model_comp1, loss_over_time, best_loss


def predict(model, test_tensors_comp0, test_tensors_comp1, tqdm_desc):
    test_losses = []
    with th.no_grad():
        model.eval()
        with tqdm.tqdm(range(len(test_tensors_comp0)), unit="examples") as tepoch:
            for i in tepoch:
                tepoch.set_description(tqdm_desc)
                loss, _ = model(test_tensors_comp0[0], test_tensors_comp1[i])
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

    with open(f"{data_folder}train_tensors_comp0_{loop}_{FEATS}.pkl", "rb") as tensorpkl:
        train_tensors_comp0 = pkl.load(tensorpkl)

    with open(f"{data_folder}train_tensors_comp1_{loop}_{FEATS}.pkl", "rb") as tensorpkl:
        train_tensors_comp1 = pkl.load(tensorpkl)

    with open(f"{data_folder}test_tensors_comp0_{loop}_{FEATS}.pkl", "rb") as tensorpkl:
        test_tensors_comp0 = pkl.load(tensorpkl)

    with open(f"{data_folder}test_tensors_comp1_{loop}_{FEATS}.pkl", "rb") as tensorpkl:
        test_tensors_comp1 = pkl.load(tensorpkl)

    with open(f"{data_folder}val_tensors_comp0_{loop}_{FEATS}.pkl", "rb") as tensorpkl:
        val_tensors_comp0 = pkl.load(tensorpkl)

    with open(f"{data_folder}val_tensors_comp1_{loop}_{FEATS}.pkl", "rb") as tensorpkl:
        val_tensors_comp1 = pkl.load(tensorpkl)

    if load_model != "":
        model.load_state_dict(th.load(load_model))

    train_tensors_comp0 = filter_tensors(train_tensors_comp0, train_inds[loop_no], blacklist)
    train_tensors_comp1 = filter_tensors(train_tensors_comp1, train_inds[loop_no], blacklist)
    batched_train_tensors_comp0 = create_batch(train_tensors_comp0, BATCH_SIZE)
    batched_train_tensors_comp1 = create_batch(train_tensors_comp1, BATCH_SIZE)

    val_tensors_comp0 = filter_tensors(val_tensors_comp0, val_inds[loop_no], blacklist)
    val_tensors_comp1 = filter_tensors(val_tensors_comp1, val_inds[loop_no], blacklist)
    batched_val_tensors_comp0 = create_batch(val_tensors_comp0, BATCH_SIZE)
    batched_val_tensors_comp1 = create_batch(val_tensors_comp1, BATCH_SIZE)

    model, loss_over_time, new_best_loss = train_model(model, batched_train_tensors_comp0, batched_train_tensors_comp1,
                                                       batched_val_tensors_comp0, batched_val_tensors_comp1,
                                                       epochs = EPOCHS, lr = LR, prev_best_loss = prev_best_loss)

    train_losses = predict(model, create_batch(train_tensors_comp0, 1), create_batch(train_tensors_comp1, 1),
                           "Calculating training error distribution")

    test_tensors_comp0 = create_batch(filter_tensors(test_tensors_comp0, test_inds[loop_no], []), 1)
    test_tensors_comp1 = create_batch(filter_tensors(test_tensors_comp1, test_inds[loop_no], []), 1)
    test_losses = predict(model, test_tensors_comp0, test_tensors_comp1, "Testing on new data")

    anomaly_thres = extreme_anomaly(train_losses)

    #filtered_test_losses = simple_lowpass_filter(test_losses, LPF_ALPHA)

    anomalies = np.array(test_losses) > anomaly_thres

    blacklist.update(anomaly_inds(anomalies, test_inds[loop_no]))

    losses_over_time = {"train": train_losses, "test": test_losses, "blacklist": blacklist} #"filtered": filtered_test_losses,

    with open(f"{results_folder}online_{loop_no}_losses_{model_string}_{EPOCHS}_{LR}_separate_comp.pkl", "wb") as lossfile:
        pkl.dump(losses_over_time, lossfile)

    th.save(model.state_dict(), f"{results_folder}online_{loop_no}_{model_string}_{EPOCHS}_{LR}_separate_comp.pt")

    return blacklist, new_best_loss


for loop in range(INIT_LOOP, END_LOOP+1):

    load_model = "" if True else f"{results_folder}online_{loop-1}_{model_string}_{EPOCHS}_{LR}.pt"
    prev_best_loss = best_loss if loop > INIT_LOOP else 10000.
    blacklist, best_loss = execute_train_test_loop(loop, prev_best_loss, model, load_model, blacklist)
