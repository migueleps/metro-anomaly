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


def train_model(model_comp0,
                model_comp1,
                batch_train_tensors_comp0,
                batch_train_tensors_comp1,
                batch_val_tensors_comp0,
                batch_val_tensors_comp1,
                epochs,
                lr):
    optimizer = optim.Adam(list(model_comp0.parameters()) + list(model_comp1.parameters()), lr=lr)
    loss_over_time = {"train": [], "val": []}

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
                nn.utils.clip_grad_norm_(model_comp0.parameters(), 1)
                nn.utils.clip_grad_norm_(model_comp1.parameters(), 1)
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

        print(f'Epoch {epoch+1}: train loss {train_loss} val loss {val_loss}')

    return model_comp0, model_comp1, loss_over_time


def predict(model_comp0, model_comp1, test_tensors_comp0, test_tensors_comp1, tqdm_desc):
    test_losses = []
    with th.no_grad():
        model_comp0.eval()
        model_comp1.eval()
        with tqdm.tqdm(range(len(test_tensors_comp0)), unit="examples") as tepoch:
            for i in tepoch:
                tepoch.set_description(tqdm_desc)
                loss_comp0, _ = model_comp0(test_tensors_comp0[0])
                loss_comp1, _ = model_comp1(test_tensors_comp1[i])
                loss = loss_comp0 + loss_comp1
                test_losses.append(loss.item())
    return test_losses


def extreme_anomaly(dist):
    q25, q75 = np.quantile(dist, [0.25, 0.75])
    return q75 + 3*(q75-q25)


def anomaly_inds(anomalies, test_inds):
    i_first, i_last = test_inds
    anom_inds = np.where(anomalies)[0]
    return anom_inds + i_first


def execute_train_test_loop(loop_no, model_comp0, model_comp1, args):

    print(f"STARTING LOOP {loop_no+1}")

    with open(f"{args.data_folder}train_tensors_comp0_{loop_no}_{args.FEATS}.pkl", "rb") as tensorpkl:
        train_tensors_comp0 = pkl.load(tensorpkl)

    with open(f"{args.data_folder}train_tensors_comp1_{loop_no}_{args.FEATS}.pkl", "rb") as tensorpkl:
        train_tensors_comp1 = pkl.load(tensorpkl)

    with open(f"{args.data_folder}test_tensors_comp0_{loop_no}_{args.FEATS}.pkl", "rb") as tensorpkl:
        test_tensors_comp0 = pkl.load(tensorpkl)

    with open(f"{args.data_folder}test_tensors_comp1_{loop_no}_{args.FEATS}.pkl", "rb") as tensorpkl:
        test_tensors_comp1 = pkl.load(tensorpkl)

    with open(f"{args.data_folder}val_tensors_comp0_{loop_no}_{args.FEATS}.pkl", "rb") as tensorpkl:
        val_tensors_comp0 = pkl.load(tensorpkl)

    with open(f"{args.data_folder}val_tensors_comp1_{loop_no}_{args.FEATS}.pkl", "rb") as tensorpkl:
        val_tensors_comp1 = pkl.load(tensorpkl)

    train_tensors_comp0 = filter_tensors(train_tensors_comp0, args.train_inds[loop_no], args.blacklist, args)
    train_tensors_comp1 = filter_tensors(train_tensors_comp1, args.train_inds[loop_no], args.blacklist, args)
    batched_train_tensors_comp0 = create_batch(train_tensors_comp0, args.BATCH_SIZE, args)
    batched_train_tensors_comp1 = create_batch(train_tensors_comp1, args.BATCH_SIZE, args)

    val_tensors_comp0 = filter_tensors(val_tensors_comp0, args.val_inds[loop_no], args.blacklist, args)
    val_tensors_comp1 = filter_tensors(val_tensors_comp1, args.val_inds[loop_no], args.blacklist, args)
    batched_val_tensors_comp0 = create_batch(val_tensors_comp0, args.BATCH_SIZE, args)
    batched_val_tensors_comp1 = create_batch(val_tensors_comp1, args.BATCH_SIZE, args)

    model_comp0, model_comp1, loss_over_time = train_model(model_comp0,
                                                           model_comp1,
                                                           batched_train_tensors_comp0,
                                                           batched_train_tensors_comp1,
                                                           batched_val_tensors_comp0,
                                                           batched_val_tensors_comp1,
                                                           epochs=args.EPOCHS,
                                                           lr=args.LR)

    train_losses = predict(model_comp0,
                           model_comp1,
                           create_batch(train_tensors_comp0, 1, args),
                           create_batch(train_tensors_comp1, 1, args),
                           "Calculating training error distribution")

    test_tensors_comp0 = create_batch(filter_tensors(test_tensors_comp0, args.test_inds[loop_no], [], args), 1, args)
    test_tensors_comp1 = create_batch(filter_tensors(test_tensors_comp1, args.test_inds[loop_no], [], args), 1, args)
    test_losses = predict(model_comp0, model_comp1, test_tensors_comp0, test_tensors_comp1, "Testing on new data")

    anomaly_threshold = extreme_anomaly(train_losses)

    anomalies = np.array(test_losses) > anomaly_threshold

    args.blacklist.update(anomaly_inds(anomalies, args.test_inds[loop_no]))

    losses_over_time = {"train": train_losses, "test": test_losses, "blacklist": args.blacklist}

    with open(args.results_string(loop_no), "wb") as lossfile:
        pkl.dump(losses_over_time, lossfile)

    th.save(model_comp0.state_dict(), args.model_saving_string(loop_no, "_comp0"))
    th.save(model_comp1.state_dict(), args.model_saving_string(loop_no, "_comp1"))

    return model_comp0, model_comp1


def load_parameters():

    FEATS_TO_NUMBER = {"analog_feats": 8, "digital_feats": 8, "all_feats": 16}

    args = SimpleNamespace()

    args.INIT_LOOP = 0
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

    args.MODEL_NAME = "lstm_sae"

    args.results_folder = "results/"
    args.data_folder = "data/"

    args.model_string = f"{args.MODEL_NAME}_{args.FEATS}_{args.EMBEDDING}"

    args.blacklist = set()

    args.results_string = lambda loop_no: f"{args.results_folder}online_{loop_no}_losses_{args.model_string}\
    _{args.EPOCHS}_{args.LR}_separate_comp.pkl"
    args.model_saving_string = lambda loop_no, model_label: f"{args.results_folder}online_{loop_no}\
    _{args.model_string}{model_label}_{args.EPOCHS}_{args.LR}_separate_comp.pt"
    if args.INIT_LOOP > 0:
        try:
            with open(args.results_string(args.INIT_LOOP-1), "wb") as lossfile:
                loss_over_time = pkl.load(lossfile)
                args.blacklist = loss_over_time["blacklist"]
        except FileNotFoundError:
            print("Tried loading blacklist from previous results but failed, starting with empty blacklist.")
            pass

    with open(f"{args.data_folder}online_train_val_test_inds.pkl", "rb") as indspkl:
        args.train_inds, args.val_inds, args.test_inds = pkl.load(indspkl)

    return args


def main(arguments):

    MODELS = {"lstm_ae": LSTM_AE, "lstm_sae": LSTM_SAE}

    model_comp0 = MODELS[arguments.MODEL_NAME](arguments.NUMBER_FEATURES,
                                               arguments.EMBEDDING,
                                               arguments.DROPOUT,
                                               arguments.LSTM_LAYERS,
                                               arguments.device,
                                               arguments.sparsity_weight,
                                               arguments.sparsity_parameter).to(arguments.device)
    model_comp1 = MODELS[arguments.MODEL_NAME](arguments.NUMBER_FEATURES,
                                               arguments.EMBEDDING,
                                               arguments.DROPOUT,
                                               arguments.LSTM_LAYERS,
                                               arguments.device,
                                               arguments.sparsity_weight,
                                               arguments.sparsity_parameter).to(arguments.device)

    for loop in range(arguments.INIT_LOOP, arguments.END_LOOP+1):
        model_comp0, model_comp1 = execute_train_test_loop(loop, model_comp0, model_comp1, arguments)


if __name__ == "__main__":
    argument_dict = load_parameters()
    main(argument_dict)
