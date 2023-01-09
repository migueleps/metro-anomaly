import numpy as np
import torch as th
from torch import nn, optim
import pickle as pkl
from LSTMAE import LSTM_AE
from LSTM_SAE import LSTM_SAE
import tqdm
from types import SimpleNamespace
from EarlyStopper import EarlyStopping


def filter_tensors(list_of_cycles, cycle_indices, blacklist, args):
    first_cycle, _ = cycle_indices  # last cycle is meant to not be included, the interval is [start,end)
    tensor_list = []
    for ind, cycle in enumerate(list_of_cycles):
        if ind+first_cycle in blacklist:
            continue
        tensor_list.append(cycle.to(args.device))
    return tensor_list


def train_model(model,
                train_tensors,
                val_tensors,
                epochs,
                lr,
                args):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    loss_over_time = {"train": [], "val": []}
    early_stopper = EarlyStopping(args.succesive_iters,
                                  args.delta_worse,
                                  args.delta_better)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        val_losses = []
        with tqdm.tqdm(train_tensors, unit="cycles") as tqdm_epoch:
            for train_tensor in tqdm_epoch:
                if epochs > 1:
                    tqdm_epoch.set_description(f"Epoch {epoch+1}")
                else:
                    tqdm_epoch.set_description(f"Training on online data")
                optimizer.zero_grad()
                loss, _ = model(train_tensor)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                train_losses.append(loss.item())

        with th.no_grad():
            model.eval()
            for val_tensor in val_tensors:
                loss, _ = model(val_tensor)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses) if len(val_tensors) > 0 else 0

        loss_over_time['train'].append(train_loss)
        loss_over_time['val'].append(val_loss)

        print(f'Epoch {epoch+1}: train loss {train_loss} val loss {val_loss}')

        if early_stopper.stopping_condition(val_loss):
            break

    return model, loss_over_time


def predict(model, test_tensors, tqdm_desc):
    test_losses = []
    with th.no_grad():
        model.eval()
        with tqdm.tqdm(test_tensors, unit="cycles") as tqdm_epoch:
            for test_tensor in tqdm_epoch:
                tqdm_epoch.set_description(tqdm_desc)
                loss, _ = model(test_tensor)
                test_losses.append(loss.item())
    return test_losses


def extreme_anomaly(loop, args):
    dist_window = args.train_losses[args.train_indices[loop]]
    dist = dist_window[np.where(dist_window > -1)[0]]
    q25, q75 = np.quantile(dist, [0.25, 0.75])
    return q75 + 3*(q75-q25)


def anomaly_indices(anomalies, test_indices):
    i_first, i_last = test_indices
    anom_indices = np.where(anomalies)[0]
    return anom_indices + i_first


def offline_train(model, args):

    print(f"Starting offline training")

    with open(f"{args.data_folder}train_tensors_offline_{args.FEATS}.pkl", "rb") as tensor_pkl:
        train_tensors = pkl.load(tensor_pkl)
        train_tensors = [tensor.to(args.device) for tensor in train_tensors]

    with open(f"{args.data_folder}val_tensors_offline_{args.FEATS}.pkl", "rb") as tensor_pkl:
        val_tensors = pkl.load(tensor_pkl)
        val_tensors = [tensor.to(args.device) for tensor in val_tensors]

    model, loss_over_time = train_model(model,
                                        train_tensors,
                                        val_tensors,
                                        epochs=args.EPOCHS,
                                        lr=args.LR,
                                        args=args)

    train_losses = predict(model, train_tensors, "Calculating training error distribution")

    with open(args.offline_results_string, "wb") as loss_file:
        pkl.dump(loss_over_time, loss_file)

    th.save(model.state_dict(), args.model_saving_string("offline"))

    return model, np.array([-1]*len(val_tensors) + train_losses)


def execute_online_loop(loop_no, model, args):

    print(f"Starting online loop {loop_no+1}")

    with open(f"{args.data_folder}test_tensors_{loop_no}_{args.FEATS}.pkl", "rb") as tensor_pkl:
        test_tensors = pkl.load(tensor_pkl)
        test_tensors = [tensor.to(args.device) for tensor in test_tensors]

    test_losses = predict(model, test_tensors, "Testing on new data")

    anomaly_threshold = extreme_anomaly(loop_no, args)
    anomalies = np.array(test_losses) > anomaly_threshold
    detected_anomalies = anomaly_indices(anomalies, args.test_indices[loop_no])
    args.blacklist.update(detected_anomalies)

    train_tensors = filter_tensors(test_tensors, args.test_indices[loop_no], args.blacklist, args)
    model, _ = train_model(model,
                           train_tensors,
                           [],
                           epochs=1,
                           lr=args.LR,
                           args=args)

    train_losses = predict(model, test_tensors, "Calculating new training error distribution")
    args.train_losses = np.append(args.train_losses, train_losses)
    args.train_losses[detected_anomalies] = -1

    losses_over_time = {"test": test_losses, "blacklist": args.blacklist}

    with open(args.results_string(loop_no), "wb") as loss_file:
        pkl.dump(losses_over_time, loss_file)

    th.save(model.state_dict(), args.model_saving_string(loop_no))

    return model


def load_parameters():

    FEATS_TO_NUMBER = {"analog_feats": 8, "digital_feats": 8, "all_feats": 16}

    args = SimpleNamespace()

    args.INIT_LOOP = 0
    args.END_LOOP = 17

    args.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    args.MODEL_NAME = "lstm_ae"

    args.EPOCHS = 1000
    args.LR = 1e-3
    args.weight_decay = 0 if args.MODEL_NAME == "lstm_ae" else 1e-3

    args.succesive_iters = 10
    args.delta_worse = 0.05
    args.delta_better = 0.005

    args.DROPOUT = 0.2
    args.EMBEDDING = 6
    args.LSTM_LAYERS = 3
    args.HIDDEN_DIMS = [16, 8]

    args.sparsity_weight = 1
    args.sparsity_parameter = 0.05

    args.FEATS = "analog_feats"
    args.NUMBER_FEATURES = FEATS_TO_NUMBER[args.FEATS]

    args.results_folder = "results/"
    args.data_folder = "data/"

    args.model_string = f"{args.MODEL_NAME}_{args.FEATS}_{args.EMBEDDING}"

    args.blacklist = set()

    args.results_string = lambda loop_no: f"{args.results_folder}online_{loop_no}_losses_{args.model_string}_{args.EPOCHS}_{args.LR}.pkl"
    args.model_saving_string = lambda loop_no: f"{args.results_folder}online_{loop_no}_{args.model_string}_{args.EPOCHS}_{args.LR}.pt"
    args.offline_results_string = f"{args.results_folder}online_offline_losses_{args.model_string}_{args.EPOCHS}_{args.LR}.pkl"

    if args.INIT_LOOP > 0:
        try:
            with open(args.results_string(args.INIT_LOOP-1), "rb") as loss_file:
                loss_over_time = pkl.load(loss_file)
                args.blacklist = loss_over_time["blacklist"]
        except FileNotFoundError:
            print("Tried loading blacklist from previous results but failed, starting with empty blacklist.")
            pass

    with open(f"{args.data_folder}online_train_val_test_inds.pkl", "rb") as indices_pkl:
        args.train_indices, args.val_indices, args.test_indices = pkl.load(indices_pkl)

    return args


def main(arguments):

    MODELS = {"lstm_ae": LSTM_AE, "lstm_sae": LSTM_SAE}

    model = MODELS[arguments.MODEL_NAME](arguments.NUMBER_FEATURES,
                                         arguments.EMBEDDING,
                                         arguments.HIDDEN_DIMS,
                                         arguments.DROPOUT,
                                         arguments.LSTM_LAYERS,
                                         arguments.device,
                                         arguments.sparsity_weight,
                                         arguments.sparsity_parameter).to(arguments.device)

    if arguments.INIT_LOOP == 0:
        model, train_losses = offline_train(model, arguments)
        arguments.train_losses = train_losses
    else:
        model.load_state_dict(th.load(arguments.model_saving_string(arguments.INIT_LOOP-1)))

    for loop in range(arguments.INIT_LOOP, arguments.END_LOOP+1):
        model = execute_online_loop(loop, model, arguments)


if __name__ == "__main__":
    argument_dict = load_parameters()
    main(argument_dict)
