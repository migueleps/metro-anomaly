import os
import numpy as np
import torch as th
from torch import nn, optim
import pickle as pkl
from LSTMAE import LSTM_AE
from LSTM_SAE import LSTM_SAE
from LSTM_AllLayerSAE import LSTM_AllLayerSAE
import tqdm
from EarlyStopper import EarlyStopping
from ArgumentParser import parse_arguments


def create_batch(tensor_list, batch_size, args):
    batches = []
    for batch_number in range(0, len(tensor_list), batch_size):
        tensors_to_batch = tensor_list[batch_number:(batch_number+batch_size)]
        tensor_mini_batch = th.stack(tensors_to_batch)
        batches.append(tensor_mini_batch.to(args.device))
    return batches


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
        with tqdm.tqdm(train_tensors, unit="batches") as tqdm_epoch:
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
        with tqdm.tqdm(test_tensors, unit="chunks") as tqdm_epoch:
            for test_tensor in tqdm_epoch:
                tqdm_epoch.set_description(tqdm_desc)
                loss, _ = model(test_tensor.unsqueeze(0))
                test_losses.append(loss.item())
    return test_losses


def offline_train(model, args):

    print(f"Starting offline training")

    with open(f"{args.data_folder}train_tensors_1min_chunks_offline_{args.FEATS}.pkl", "rb") as tensor_pkl:
        train_tensors = pkl.load(tensor_pkl)
        train_tensors = [tensor.to(args.device) for tensor in train_tensors]

    with open(f"{args.data_folder}val_tensors_1min_chunks_offline_{args.FEATS}.pkl", "rb") as tensor_pkl:
        val_tensors = pkl.load(tensor_pkl)
        val_tensors = [tensor.to(args.device) for tensor in val_tensors]

    batched_train = create_batch(train_tensors, args.BATCH_SIZE, args)
    batched_val = create_batch(val_tensors, args.BATCH_SIZE, args)

    model, loss_over_time = train_model(model,
                                        batched_train,
                                        batched_val,
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
    for loop in range(args.END_LOOP+1):
        with open(f"{args.data_folder}test_tensors_1min_chunks_{loop}_{args.FEATS}.pkl", "rb") as tensor_pkl:
            test_tensors = pkl.load(tensor_pkl)
            all_test_tensors.extend([tensor.to(args.device) for tensor in test_tensors])

    test_losses = predict(model, all_test_tensors, "Testing on new data")

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

    arguments.blacklist = set()

    arguments.results_string = lambda loop_no: f"{arguments.results_folder}no_cycle_{loop_no}_losses_{arguments.model_string}_{arguments.EPOCHS}_{arguments.LR}.pkl"
    arguments.model_saving_string = lambda loop_no: f"{arguments.results_folder}no_cycle_{loop_no}_{arguments.model_string}_{arguments.EPOCHS}_{arguments.LR}.pt"

    if arguments.INIT_LOOP > 0:
        try:
            with open(arguments.results_string(arguments.INIT_LOOP-1), "rb") as loss_file:
                loss_over_time = pkl.load(loss_file)
                arguments.blacklist = loss_over_time["blacklist"]
        except FileNotFoundError:
            print("Tried loading blacklist from previous results but failed, starting with empty blacklist.")
            pass

    with open(f"{arguments.data_folder}online_train_val_test_inds.pkl", "rb") as indices_pkl:
        arguments.train_indices, arguments.val_indices, arguments.test_indices = pkl.load(indices_pkl)

    return arguments


def main(arguments):

    MODELS = {"lstm_ae": LSTM_AE, "lstm_sae": LSTM_SAE, "lstm_all_layer_sae": LSTM_AllLayerSAE}

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
