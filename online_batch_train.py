import numpy as np
import torch as th
from torch import nn, optim
import pickle as pkl
from LSTMAE_mini_batch import LSTM_AE
from LSTM_SAE_mini_batch import LSTM_SAE
from torch.nn.utils.rnn import pack_padded_sequence
import tqdm
from types import SimpleNamespace


def create_batch(tensor_list, batch_size, args):
    packed_batches = []
    for batch_number in range(0, len(tensor_list), batch_size):
        tensors_to_batch = tensor_list[batch_number:(batch_number+batch_size)]
        batch_tensor_lengths = th.tensor([tensor.shape[1] for tensor in tensors_to_batch])
        longest_seq = max(batch_tensor_lengths)
        mini_batch = []
        for tensor in tensors_to_batch:
            tensor = tensor.squeeze()
            padded_tensor = th.cat([tensor, th.zeros(longest_seq - tensor.shape[0], tensor.shape[1]).to(args.device)])
            mini_batch.append(padded_tensor)
        tensor_mini_batch = th.stack(mini_batch)
        packed = pack_padded_sequence(tensor_mini_batch, batch_tensor_lengths, batch_first=True, enforce_sorted=False)
        packed_batches.append(packed.to(args.device))
    return packed_batches


def filter_tensors(list_of_cycles, cycle_indices, blacklist, args):
    first_cycle, _ = cycle_indices  # last cycle is meant to not be included, the interval is [start,end)
    tensor_list = []
    for ind, cycle in enumerate(list_of_cycles):
        if ind+first_cycle in blacklist:
            continue
        tensor_list.append(cycle.to(args.device))
    return tensor_list


def train_model(model,
                batch_train_tensors,
                batch_val_tensors,
                epochs,
                lr):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_over_time = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        val_losses = []
        with tqdm.tqdm(batch_train_tensors, unit="cycles") as tqdm_epoch:
            for packed_train_batch in tqdm_epoch:
                tqdm_epoch.set_description(f"Epoch {epoch+1}")
                optimizer.zero_grad()
                batch_loss, _ = model(packed_train_batch)
                batch_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                train_losses.append(batch_loss.item())

        with th.no_grad():
            model.eval()
            for packed_val_batch in batch_val_tensors:
                batch_loss, _ = model(packed_val_batch)
                val_losses.append(batch_loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        loss_over_time['train'].append(train_loss)
        loss_over_time['val'].append(val_loss)

        print(f'Epoch {epoch+1}: train loss {train_loss} val loss {val_loss}')

    return model, loss_over_time


def predict(model, test_tensors, tqdm_desc):
    test_losses = []
    with th.no_grad():
        model.eval()
        with tqdm.tqdm(test_tensors, unit="examples") as tqdm_epoch:
            for test_tensor in tqdm_epoch:
                tqdm_epoch.set_description(tqdm_desc)
                loss, _ = model(test_tensor)
                test_losses.append(loss.item())
    return test_losses


def extreme_anomaly(dist):
    q25, q75 = np.quantile(dist, [0.25, 0.75])
    return q75 + 3*(q75-q25)


def anomaly_indices(anomalies, test_indices):
    i_first, i_last = test_indices
    anom_indices = np.where(anomalies)[0]
    return anom_indices + i_first


def execute_train_test_loop(loop_no, model, args):

    print(f"STARTING LOOP {loop_no+1}")

    with open(f"{args.data_folder}train_tensors_{loop_no}_{args.FEATS}.pkl", "rb") as tensor_pkl:
        train_tensors = pkl.load(tensor_pkl)

    with open(f"{args.data_folder}test_tensors_{loop_no}_{args.FEATS}.pkl", "rb") as tensor_pkl:
        test_tensors = pkl.load(tensor_pkl)

    with open(f"{args.data_folder}val_tensors_{loop_no}_{args.FEATS}.pkl", "rb") as tensor_pkl:
        val_tensors = pkl.load(tensor_pkl)

    train_tensors = filter_tensors(train_tensors, args.train_indices[loop_no], args.blacklist, args)
    batched_train_tensors = create_batch(train_tensors, args.BATCH_SIZE, args)
    val_tensors = filter_tensors(val_tensors, args.val_indices[loop_no], args.blacklist, args)
    batched_val_tensors = create_batch(val_tensors, args.BATCH_SIZE, args)

    model, loss_over_time = train_model(model,
                                        batched_train_tensors,
                                        batched_val_tensors,
                                        epochs=args.EPOCHS,
                                        lr=args.LR)

    train_losses = predict(model, create_batch(train_tensors, 1, args), "Calculating training error distribution")

    test_tensors = create_batch(filter_tensors(test_tensors, args.test_indices[loop_no], [], args), 1, args)
    test_losses = predict(model, test_tensors, "Testing on new data")

    anomaly_threshold = extreme_anomaly(train_losses)
    anomalies = np.array(test_losses) > anomaly_threshold

    args.blacklist.update(anomaly_indices(anomalies, args.test_indices[loop_no]))

    losses_over_time = {"train": train_losses, "test": test_losses, "blacklist": args.blacklist}

    with open(args.results_string(loop_no), "wb") as loss_file:
        pkl.dump(losses_over_time, loss_file)

    th.save(model.state_dict(), args.model_saving_string(loop_no))

    return model


def load_parameters():

    FEATS_TO_NUMBER = {"analog_feats": 8, "digital_feats": 8, "all_feats": 16}

    args = SimpleNamespace()

    args.INIT_LOOP = 14
    args.END_LOOP = 17

    args.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    args.EPOCHS = 100
    args.LR = 1e-3
    args.BATCH_SIZE = 32

    args.DROPOUT = 0.2
    args.EMBEDDING = 64
    args.LSTM_LAYERS = 2

    args.sparsity_weight = 1
    args.sparsity_parameter = 0.05

    args.FEATS = "analog_feats"
    args.NUMBER_FEATURES = FEATS_TO_NUMBER[args.FEATS]

    args.MODEL_NAME = "lstm_ae"

    args.results_folder = "results/"
    args.data_folder = "data/"

    args.model_string = f"{args.MODEL_NAME}_{args.FEATS}_{args.EMBEDDING}"

    args.blacklist = set()

    args.results_string = lambda loop_no: f"{args.results_folder}online_{loop_no}_losses_{args.model_string}\
    _{args.EPOCHS}_{args.LR}.pkl"
    args.model_saving_string = lambda loop_no: f"{args.results_folder}online_{loop_no}\
    _{args.model_string}_{args.EPOCHS}_{args.LR}.pt"

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
                                         arguments.DROPOUT,
                                         arguments.LSTM_LAYERS,
                                         arguments.device,
                                         arguments.sparsity_weight,
                                         arguments.sparsity_parameter).to(arguments.device)
    if arguments.INIT_LOOP > 0:
        model.load_state_dict(th.load(arguments.model_saving_string(arguments.INIT_LOOP-1)))

    for loop in range(arguments.INIT_LOOP, arguments.END_LOOP+1):
        model = execute_train_test_loop(loop, model, arguments)


if __name__ == "__main__":
    argument_dict = load_parameters()
    main(argument_dict)
