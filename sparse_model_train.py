
import numpy as np
import torch as th
from torch import nn, optim
import copy
import pickle as pkl
from LSTM_SAE import LSTM_SAE
import tqdm

sparsity_weight = 0.01
sparsity_parameter = 0.05


def sparsity_loss(target_activation, hidden_outputs, device):
    average_activation = th.mean(th.sigmoid(hidden_outputs), 1)
    target_activations = th.tensor([target_activation] * average_activation.shape[1]).to(device)
    kl_div_part1 = th.log(target_activations/average_activation)
    kl_div_part2 = th.log((1-target_activations)/(1-average_activation))
    return th.sum(target_activation * kl_div_part1 + (1-target_activation) * kl_div_part2)


def train_model(model, train_tensors, val_tensors, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=1e-3)
    mse = nn.MSELoss(reduction="mean").to(device)
    loss_over_time = {"train": [], "val": []}

    best_model = copy.deepcopy(model.state_dict())
    best_loss = 100000.0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        with tqdm.tqdm(train_tensors, unit="example") as tepoch:
            for train_tensor in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                optimizer.zero_grad()

                reconstruction, hidden_output = model(train_tensor)
                sparsity_comp = sparsity_loss(sparsity_parameter, hidden_output, device)
                loss = mse(reconstruction,train_tensor) + sparsity_weight * sparsity_comp
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                train_losses.append(loss.item())

        val_losses = []

        model.eval()
        with th.no_grad():
            for val_tensor in val_tensors:
                reconstruction, hidden_output = model(val_tensor)
                sparsity_comp = sparsity_loss(sparsity_parameter, hidden_output, device)
                loss = mse(reconstruction,val_tensor) + sparsity_weight * sparsity_comp
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

for i in range(len(train_tensors)):
    train_tensors[i] = train_tensors[i].to(device)

for i in range(len(val_tensors)):
    val_tensors[i] = val_tensors[i].to(device)


lstm_sae = LSTM_SAE(8, 64, 32, 0.2, device).to(device)
lstm_sae, loss_over_time = train_model(lstm_sae, train_tensors, val_tensors, epochs = 50, lr = 1e-3, device = device)

with open("lstm_sae_analog_64_32_150_1e-3.pkl", "wb") as modelpkl:
    pkl.dump(lstm_sae, modelpkl)

with open("lstm_sae_analog_64_32_150_1e-3_losses.pkl", "wb") as lossespkl:
    pkl.dump(loss_over_time, lossespkl)
