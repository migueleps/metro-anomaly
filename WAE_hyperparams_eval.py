import sys
import os
from torch.distributions.multivariate_normal import MultivariateNormal
import torch as th
from TCN_AAE import SimpleDiscriminator_TCN, LSTMDiscriminator_TCN, ConvDiscriminator_TCN
from LSTM_AAE import LSTMDiscriminator
import pickle as pkl
import numpy as np
import pandas as pd
from operator import itemgetter
from itertools import groupby
from scipy.stats import zscore

import warnings
warnings.filterwarnings("ignore")


def detect_failures(anom_indices):
    failure_list = []
    failure = set()
    for i in range(len(anom_indices) - 1):
        if anom_indices[i] == 1 and anom_indices[i + 1] == 1:
            failure.add(i)
            failure.add(i + 1)
        elif len(failure) > 0:
            failure_list.append(failure)
            failure = set()

    if len(failure) > 0:
        failure_list.append(failure)

    return failure_list


def collate_intervals(interval_list):
    diff_consecutive_intervals = [(interval_list[i+1].left - interval_list[i].right).days for i in range(len(interval_list)-1)]
    lt_1day = np.where(np.array(diff_consecutive_intervals) <= 1)[0]
    collated_intervals = []
    for k, g in groupby(enumerate(lt_1day), lambda ix: ix[0]-ix[1]):
        collated = list(map(itemgetter(1), g))
        collated_intervals.append(pd.Interval(interval_list[collated[0]].left, interval_list[collated[-1]+1].right, closed="both"))

    collated_intervals.extend([interval_list[i] for i in range(len(interval_list)) if i not in lt_1day and i-1 not in lt_1day])
    return sorted(collated_intervals)


def failure_list_to_interval(cycle_dates, failures):
    failure_intervals = []
    for failure in failures:
        failure = sorted(failure)
        failure_intervals.append(pd.Interval(cycle_dates[failure[0]].left, cycle_dates[failure[-1]].right, closed="both"))
    return failure_intervals


def print_failures_interval(cycle_dates, output, label):
    failures = detect_failures(output)
    failure_intervals = failure_list_to_interval(cycle_dates, failures)
    collated_intervals = collate_intervals(failure_intervals)
    if len(collated_intervals) <= 5:
        print(label)
        for interval in collated_intervals:
            print(interval)


def generate_intervals(granularity, start_timestamp, end_timestamp):
    current_timestamp = start_timestamp
    interval_length = pd.offsets.DateOffset(**granularity)
    interval_list = []
    while current_timestamp < end_timestamp:
        interval_list.append(pd.Interval(current_timestamp, current_timestamp + interval_length, closed="left"))
        current_timestamp = current_timestamp + interval_length
    return interval_list


def generate_chunks(df, chunk_size, chunk_stride):

    from numpy.lib.stride_tricks import sliding_window_view

    gaps = list((g:=df.timestamp.diff().gt(pd.Timedelta(minutes=1)))[g].index)
    window_start_date = []
    start = 0
    for gap in gaps:
        tdf = df.iloc[start:gap,:]
        if len(tdf) < chunk_size:
            start = gap
            continue
        window_start_date.append(sliding_window_view(tdf.timestamp.values, chunk_size)[::chunk_stride,[0,-1]])
        start = gap
    tdf = df.iloc[start:, :]
    if len(tdf) >= chunk_size:
        window_start_date.append(sliding_window_view(tdf.timestamp.values, chunk_size)[::chunk_stride,[0,-1]])

    return np.concatenate(window_start_date)


def map_cycles_to_intervals(interval_list, chunk_dates):
    cycles_dates = list(map(lambda x: pd.Interval(pd.Timestamp(x[0]), pd.Timestamp(x[1]), closed="both"), chunk_dates))
    return list(map(lambda x: np.where([x.overlaps(i) for i in cycles_dates])[0], interval_list))


def extreme_anomaly(dist):
    q25, q75 = np.quantile(dist, [0.25, 0.75])
    return q75 + 3*(q75-q25)


def simple_lowpass_filter(arr, alpha):
    y = arr[0]
    filtered_arr = [y]
    for elem in arr[1:]:
        y = y + alpha * (elem - y)
        filtered_arr.append(y)
    return filtered_arr


def print_failures_reconstruction(train_losses, test_losses):

    median_train_losses = np.array([np.median(np.array(train_losses["reconstruction"])[tc]) for tc in train_chunks_to_intervals if len(tc) > 0])
    median_test_losses = np.array([np.median(np.array(test_losses["reconstruction"])[tc]) for tc in test_chunks_to_intervals if len(tc) > 0])
    anomaly_threshold = extreme_anomaly(median_train_losses)
    binary_output = np.array(np.array(median_test_losses) > anomaly_threshold, dtype=int)
    print_failures_interval(intervals_of_interest,
                            binary_output,
                            f"Binary output")
    output_lpf = {alpha: np.array(simple_lowpass_filter(binary_output,alpha)) for alpha in alpha_range}
    for alpha in alpha_range:
        for threshold in [0.4, 0.5]:
            print_failures_interval(intervals_of_interest,
                                    np.array(np.array(output_lpf[alpha]) > threshold, dtype=int),
                                    f"Alpha: {alpha} - threshold: {threshold}")


def print_failures_critic(train_losses, test_losses):

    median_train_losses = np.array([np.median(np.array(train_losses["reconstruction"])[tc]) for tc in train_chunks_to_intervals if len(tc) > 0])
    median_test_losses = np.array([np.median(np.array(test_losses["reconstruction"])[tc]) for tc in test_chunks_to_intervals if len(tc) > 0])

    median_train_critic = np.array([np.median(np.array(train_losses["critic"])[tc]) for tc in train_chunks_to_intervals if len(tc) > 0])
    median_test_critic = np.array([np.median(np.array(test_losses["critic"])[tc]) for tc in test_chunks_to_intervals if len(tc) > 0])

    combine_critic_reconstruction = np.abs(zscore(median_test_critic, ddof=1)) * median_test_losses
    combine_critic_reconstruction_train = np.abs(zscore(median_train_critic, ddof=1)) * median_train_losses

    anomaly_threshold = extreme_anomaly(combine_critic_reconstruction_train)
    binary_output = np.array(np.array(combine_critic_reconstruction) > anomaly_threshold, dtype=int)
    print_failures_interval(intervals_of_interest,
                            binary_output,
                            f"Binary output")

    output_lpf = {alpha: np.array(simple_lowpass_filter(binary_output, alpha)) for alpha in alpha_range}
    for alpha in alpha_range:
        for threshold in [0.4, 0.5]:
            print_failures_interval(intervals_of_interest,
                                    np.array(np.array(output_lpf[alpha]) > threshold, dtype=int),
                                    f"Alpha: {alpha} - threshold: {threshold}")



final_metro = pd.read_csv("~/final2.csv")
final_metro["timestamp"] = pd.to_datetime(final_metro["timestamp"])

alpha_range = [0.01, 0.05, 0.1, 0.2, 0.33, 0.6]

chunk_dates = generate_chunks(final_metro, 1800, 300)
training_chunk_dates = chunk_dates[np.where(chunk_dates[:,1] < np.datetime64("2022-06-01T00:00:00.000000000"))[0]]
test_chunk_dates = chunk_dates[np.where(chunk_dates[:,0] >= np.datetime64("2022-06-01T00:00:00.000000000"))[0]]

train_intervals = generate_intervals({"minutes": 5}, pd.Timestamp(training_chunk_dates[0][0]), pd.Timestamp(training_chunk_dates[-1][0]))
test_intervals = generate_intervals({"minutes": 5}, pd.Timestamp(test_chunk_dates[0][0]), pd.Timestamp(test_chunk_dates[-1][0]))

train_chunks_to_intervals = map_cycles_to_intervals(train_intervals, training_chunk_dates)
test_chunks_to_intervals = map_cycles_to_intervals(test_intervals, test_chunk_dates)

intervals_of_interest = [interval for i, interval in enumerate(test_intervals) if len(test_chunks_to_intervals[i]) > 0]


epochs = 150
lrs = [0.001, 0.0001]
disc_lr = [1, 0.5, 0.1]
emb_size = 4

encdec_layers = [(7, 9), (10, 3)]
encdec_hidden_units = [6, 30]

disc_layers = [1, 2, 3]
disc_hidden = [6, 32]

model = "LSTMDiscriminator_TCN"

multivariate_normal = MultivariateNormal(th.zeros(4), th.eye(4))


def model_string(model, tcn_layers, tcn_hidden, tcn_kernel, disc_layers, disc_hidden):
    first_part = lambda tcn_layers: f"LSTMDiscriminator_TCN_analog_feats_4_{tcn_layers}"
    last_part = lambda disc_layers, disc_hidden: f"10.0_{disc_layers}_{disc_hidden}"
    model_specific_part = lambda tcn_hidden, tcn_kernel: f"_{tcn_hidden}_{tcn_kernel}"
    return f"{model}_{first_part(tcn_layers)}{model_specific_part(tcn_hidden, tcn_kernel)}_{last_part(disc_layers, disc_hidden)}"


def results_string(LR, disc_lr, tcn_layers, tcn_hidden, tcn_kernel, disc_layers, disc_hidden):
    model_part = model_string('WAE', tcn_layers, tcn_hidden, tcn_kernel, disc_layers, disc_hidden)
    return f"results/final_chunks_complete_losses_{model_part}_150_{LR}_{disc_lr}_64.pkl"


def model_loading_string(LR, disc_lr, tcn_layers, tcn_hidden, tcn_kernel, disc_layers, disc_hidden):
    model_part = model_string('WAE_discriminator', tcn_layers, tcn_hidden, tcn_kernel, disc_layers, disc_hidden)
    return f"results/final_chunks_offline_{model_part}_150_{LR}_{disc_lr}_64.pt"


for disc_layer in disc_layers:
    for discriminator_hidden in disc_hidden:
        discriminator = LSTMDiscriminator_TCN(4, 0.2, n_layers=disc_layer, disc_hidden=discriminator_hidden).to(th.device("cuda"))
        for lr in lrs:
            for r in disc_lr:
                dl = r * lr
                for tcn_layers, tcn_kernel in encdec_layers:
                    for encdec_hidden in encdec_hidden_units:
                        model_s = model_loading_string(lr, dl, tcn_layers, encdec_hidden, tcn_kernel, disc_layer, discriminator_hidden)
                        results_s = results_string(lr, dl, tcn_layers, encdec_hidden, tcn_kernel, disc_layer, discriminator_hidden)
                        discriminator.load_state_dict(th.load(model_s, map_location=th.device("cuda")))
                        discriminator.eval()
                        with th.no_grad():
                            disc_scores = []
                            for _ in range(10000):
                                disc_scores.append(discriminator(multivariate_normal.sample(th.Size([1, 1800])).to(th.device("cuda"))).item())
                        if np.max(disc_scores) - np.min(disc_scores) < 0.6 and np.median(disc_scores) < 0.7:
                            print("Discriminator could not separate random latent space")
                            continue

                        with open(results_s, "rb") as loss_file:
                            tl = pkl.load(loss_file)
                            test_losses = tl["test"]
                            train_losses = tl["train"]

                        print_failures_reconstruction(train_losses, test_losses)
                        print_failures_critic(train_losses, test_losses)