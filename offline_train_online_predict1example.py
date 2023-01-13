import os
import numpy as np
import torch as th
from torch import nn, optim
import pickle as pkl
from LSTMAE import LSTM_AE
from LSTM_SAE import LSTM_SAE
from LSTM_AllLayerSAE import LSTM_AllLayerSAE
from LSTM_SAE_multi_encoder import LSTM_SAE_MultiEncoder
from LSTM_AE_multi_encoder import LSTM_AE_MultiEncoder
import tqdm
from EarlyStopper import EarlyStopping
from ArgumentParser import parse_arguments
from collections import deque


def train_model(model,
                train_tensors,
                val_tensors,
                epochs,
                lr,
                args):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    loss_over_time = {"train": [], "val": []}
    early_stopper = EarlyStopping(args.successive_iters,
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

        if epoch > 100 and early_stopper.stopping_condition(val_loss):
            early_stopper.print_stop_reason()
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


def extreme_anomaly(dist):
    q25, q75 = np.quantile(dist, [0.25, 0.75])
    return q75 + 3*(q75-q25)


def calculate_train_losses(model, args):

    if args.separate_comp:
        with open(f"{args.data_folder}train_tensors_comp0_offline_{args.FEATS}.pkl", "rb") as tensor_pkl:
            train_tensors_comp0 = pkl.load(tensor_pkl)
        with open(f"{args.data_folder}train_tensors_comp1_offline_{args.FEATS}.pkl", "rb") as tensor_pkl:
            train_tensors_comp1 = pkl.load(tensor_pkl)
        train_tensors = [[train_tensors_comp0[i].to(args.device),
                          train_tensors_comp1[i].to(args.device)] for i in range(len(train_tensors_comp0))]

    else:
        with open(f"{args.data_folder}train_tensors_offline_{args.FEATS}.pkl", "rb") as tensor_pkl:
            train_tensors = pkl.load(tensor_pkl)
            train_tensors = [tensor.to(args.device) for tensor in train_tensors]

    train_losses = predict(model, train_tensors, "Calculating training error distribution")

    return train_losses


def offline_train(model, args):

    print(f"Starting offline training")

    if args.separate_comp:
        with open(f"{args.data_folder}train_tensors_comp0_offline_{args.FEATS}.pkl", "rb") as tensor_pkl:
            train_tensors_comp0 = pkl.load(tensor_pkl)
        with open(f"{args.data_folder}train_tensors_comp1_offline_{args.FEATS}.pkl", "rb") as tensor_pkl:
            train_tensors_comp1 = pkl.load(tensor_pkl)
        train_tensors = [[train_tensors_comp0[i].to(args.device),
                          train_tensors_comp1[i].to(args.device)] for i in range(len(train_tensors_comp0))]

        with open(f"{args.data_folder}val_tensors_comp0_offline_{args.FEATS}.pkl", "rb") as tensor_pkl:
            val_tensors_comp0 = pkl.load(tensor_pkl)
        with open(f"{args.data_folder}val_tensors_comp1_offline_{args.FEATS}.pkl", "rb") as tensor_pkl:
            val_tensors_comp1 = pkl.load(tensor_pkl)
        val_tensors = [[val_tensors_comp0[i].to(args.device),
                        val_tensors_comp1[i].to(args.device)] for i in range(len(val_tensors_comp0))]

    else:
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

    with open(args.results_string("offline"), "wb") as loss_file:
        pkl.dump(loss_over_time, loss_file)

    th.save(model.state_dict(), args.model_saving_string("offline"))

    return model, train_losses


def execute_online_loop(model, args):

    all_test_tensors = []
    if args.separate_comp:
        for loop in range(args.END_LOOP+1):
            t = []
            with open(f"{args.data_folder}test_tensors_comp0_{loop}_{args.FEATS}.pkl", "rb") as tensor_pkl:
                test_tensors = pkl.load(tensor_pkl)
                t.append(test_tensors)
            with open(f"{args.data_folder}test_tensors_comp1_{loop}_{args.FEATS}.pkl", "rb") as tensor_pkl:
                test_tensors = pkl.load(tensor_pkl)
                t.append(test_tensors)
            all_test_tensors.extend([[t[0][i].to(args.device), t[1][i].to(args.device)] for i in range(len(t[0]))])
    else:
        for loop in range(args.END_LOOP+1):
            with open(f"{args.data_folder}test_tensors_{loop}_{args.FEATS}.pkl", "rb") as tensor_pkl:
                test_tensors = pkl.load(tensor_pkl)
                all_test_tensors.extend([tensor.to(args.device) for tensor in test_tensors])

    optimizer = optim.Adam(model.parameters(), lr=args.LR, weight_decay=args.weight_decay)
    losses_for_anomaly = deque(args.train_losses)

    test_losses = []
    with tqdm.tqdm(all_test_tensors, unit="cycles") as tqdm_epoch:
        for test_tensor in tqdm_epoch:
            tqdm_epoch.set_description("Testing on new data")
            anomaly_threshold = extreme_anomaly(losses_for_anomaly)
            with th.no_grad():
                model.eval()
                loss, _ = model(test_tensor)
                test_losses.append(loss.item())

            if loss.item() < anomaly_threshold:
                model.train()
                optimizer.zero_grad()
                loss, _ = model(test_tensor)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                _ = losses_for_anomaly.popleft()
                losses_for_anomaly.append(loss.item())

    losses_over_time = {"test": test_losses, "train": args.train_losses}

    with open(args.results_string("complete"), "wb") as loss_file:
        pkl.dump(losses_over_time, loss_file)

    return model


def load_parameters(arguments):

    FEATS_TO_NUMBER = {"analog_feats": 8, "digital_feats": 8, "all_feats": 16}

    arguments.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    arguments.FEATS = f"{arguments.FEATS}_feats"
    arguments.NUMBER_FEATURES = FEATS_TO_NUMBER[arguments.FEATS]

    arguments.results_folder = "results/"
    arguments.data_folder = "data/"

    print_hidden = "_".join([f"{hidden}" for hidden in arguments.HIDDEN_DIMS])
    arguments.model_string = f"{arguments.MODEL_NAME}_{arguments.FEATS}_{arguments.EMBEDDING}_{print_hidden}"

    print(f"Starting execution of model: {arguments.model_string}")

    arguments.results_string = lambda loop_no: f"{arguments.results_folder}1example_{loop_no}_losses_{arguments.model_string}_{arguments.EPOCHS}_{arguments.LR}.pkl"
    arguments.model_saving_string = lambda loop_no: f"{arguments.results_folder}online_{loop_no}_{arguments.model_string}_{arguments.EPOCHS}_{arguments.LR}.pt"

    with open(f"{arguments.data_folder}online_train_val_test_inds.pkl", "rb") as indices_pkl:
        arguments.train_indices, arguments.val_indices, arguments.test_indices = pkl.load(indices_pkl)

    return arguments


def main(arguments):

    MODELS = {"lstm_ae": LSTM_AE, "lstm_sae": LSTM_SAE, "lstm_all_layer_sae": LSTM_AllLayerSAE,
              "multi_enc_sae": LSTM_SAE_MultiEncoder, "multi_enc_ae": LSTM_AE_MultiEncoder}

    model = MODELS[arguments.MODEL_NAME](arguments.NUMBER_FEATURES,
                                         arguments.EMBEDDING,
                                         arguments.HIDDEN_DIMS,
                                         arguments.DROPOUT,
                                         arguments.LSTM_LAYERS,
                                         arguments.device,
                                         arguments.sparsity_weight,
                                         arguments.sparsity_parameter).to(arguments.device)

    if arguments.INIT_LOOP == 0:
        if os.path.exists(arguments.model_saving_string("offline")) and not arguments.force_training:
            model.load_state_dict(th.load(arguments.model_saving_string("offline")))
            arguments.train_losses = calculate_train_losses(model, arguments)
        else:
            model, train_losses = offline_train(model, arguments)
            arguments.train_losses = train_losses
    else:
        model.load_state_dict(th.load(arguments.model_saving_string(arguments.INIT_LOOP-1)))

    execute_online_loop(model, arguments)


if __name__ == "__main__":
    argument_dict = parse_arguments()
    argument_dict = load_parameters(argument_dict)
    main(argument_dict)
