import numpy as np
import pandas as pd
import pickle as pkl
import os
import argparse


def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("-hours", dest="hours", type=int, default=1)
    parser.add_argument("-minutes", dest="minutes", type=int, default=0)
    parser.add_argument("-days", dest="days", type=int, default=0)
    parser.add_argument("-file", dest="file", required=True)
    parser.add_argument("-print_flag", dest="print_flag", action="store_true")
    args = parser.parse_args()
    return args


def generate_intervals(granularity):
    first_timestamp = pd.to_datetime("03-01-2022 04:00:00",
                                     dayfirst=True)
    last_timestamp = pd.to_datetime("02-06-2022 14:00:00", dayfirst=True)
    current_timestamp = first_timestamp
    interval_length = pd.offsets.DateOffset(**granularity)
    interval_list = []
    while current_timestamp < last_timestamp:
        interval_list.append(pd.Interval(current_timestamp, current_timestamp + interval_length, closed="left"))
        current_timestamp = current_timestamp + interval_length
    return interval_list


def map_cycles_to_intervals(interval_list, cycles_dates):
    return list(map(lambda x: np.where([x.overlaps(i) for i in cycles_dates])[0], interval_list))


def simple_lowpass_filter(arr, alpha):
    y = arr[0]
    filtered_arr = [y]
    for elem in arr[1:]:
        y = y + alpha * (elem - y)
        filtered_arr.append(y)
    return filtered_arr


def anomaly_inds(anomalies, test_inds):
    i_first, i_last = test_inds
    anom_inds = np.where(anomalies)[0]
    return anom_inds + i_first


def extreme_anomaly(dist):
    q25, q75 = np.quantile(dist, [0.25, 0.75])
    return q75 + 3*(q75-q25)


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


def failure_list_to_interval(intervals, failures):
    failure_intervals = []
    for failure in failures:
        failure = sorted(failure)
        failure_intervals.append(pd.Interval(intervals[failure[0]].left, intervals[failure[-1]].right, closed="left"))
    return failure_intervals


def evaluate_model(failures, ground_truth):
    tp, fn, fp = 0, 0, 0
    for failure in failures:
        if not any(map(lambda x: failure.overlaps(x), ground_truth)):
            fp += 1

    for true_failure in ground_truth:
        if any(map(lambda x: true_failure.overlaps(x), failures)):
            tp += 1
        else:
            fn += 1

    return dict(tp=tp, fn=fn, fp=fp)


def prec(d):
    if d["tp"] == 0:
        return 0
    return d["tp"] / (d["tp"] + d["fp"])


def rec(d):
    if d["tp"] == 0:
        return 0
    return d["tp"] / (d["tp"] + d["fn"])


def f1(d):
    if d["tp"] == 0:
        return 0
    return 2 * d["tp"] / (2 * d["tp"] + d["fp"] + d["fn"])


def output_metrics(bin_output, ground_truth, intervals, print_flag=True, print_label="Early Detection"):
    failures = detect_failures(bin_output)
    failure_intervals = failure_list_to_interval(intervals, failures)
    metrics = evaluate_model(failure_intervals, ground_truth)
    if print_flag:
        print(f"[{print_label}]\nFalse Positives: {metrics['fp']}\nPrecision: {prec(metrics):.3f}\n\
Recall: {rec(metrics):.3f}\nF1: {f1(metrics):.3f}\nNumber of predicted failures: {len(failures)}")
    return dict(precision=prec(metrics), recall=rec(metrics), f1=f1(metrics), false_positives=metrics['fp'],
                false_negatives=metrics['fn'])


def best_alpha(metric, metric_label, dictionary):
    best_metric, alpha_of_metric = 0, 0.01
    for alpha in alpha_range:
        value = dictionary[alpha][metric]
        if value > best_metric:
            best_metric = value
            alpha_of_metric = alpha
    print(f"Best {metric_label} was {best_metric:.3f} with alpha = {alpha_of_metric:.2f}")
    if metric == "f1":
        print(f"Achieved with precision of {dictionary[alpha_of_metric]['precision']} and recall of {dictionary[alpha_of_metric]['recall']}")


def best_alpha_min(metric, metric_label, dictionary):
    best_metric, alpha_of_metric = 400, 0
    for alpha in alpha_range:
        value = dictionary[alpha][metric]
        if value < best_metric:
            best_metric = value
            alpha_of_metric = alpha
    print(f"Least {metric_label} was {best_metric:.3f} with alpha = {alpha_of_metric:.2f}")


if __name__ == "__main__":
    air_leak1_dates = {"start": pd.to_datetime("28-02-2022 21:53:00",
                                               dayfirst=True) - pd.offsets.DateOffset(hours=2, seconds=1),
                       "end": pd.to_datetime("01-03-2022 02:00:00",
                                             dayfirst=True) - pd.offsets.DateOffset(hours=2, seconds=1)}
    air_leak2_dates = {"start": pd.to_datetime("23-03-2022 14:54:00",
                                               dayfirst=True) - pd.offsets.DateOffset(hours=2, seconds=1),
                       "end": pd.to_datetime("23-03-2022 15:24:00",
                                             dayfirst=True) - pd.offsets.DateOffset(hours=2, seconds=1)}
    oil_leak_dates = {"start": pd.to_datetime("30-05-2022 12:00:00",
                                              dayfirst=True) - pd.offsets.DateOffset(hours=2, seconds=1),
                      "end": pd.to_datetime("02-06-2022 06:18:00",
                                            dayfirst=True) - pd.offsets.DateOffset(hours=2, seconds=1)}

    true_failures = [pd.Interval(air_leak1_dates["start"], air_leak1_dates["end"], closed="both"),
                     pd.Interval(air_leak2_dates["start"], air_leak2_dates["end"], closed="both"),
                     pd.Interval(oil_leak_dates["start"], oil_leak_dates["end"], closed="both")]

    early_detection = [pd.Interval(air_leak1_dates["start"] - pd.offsets.DateOffset(hours=2), air_leak1_dates["start"],
                                   closed="left"),
                       pd.Interval(air_leak2_dates["start"] - pd.offsets.DateOffset(hours=2), air_leak2_dates["start"],
                                   closed="left"),
                       pd.Interval(oil_leak_dates["start"] - pd.offsets.DateOffset(hours=2), oil_leak_dates["start"],
                                   closed="left")]

    both = [pd.Interval(air_leak1_dates["start"] - pd.offsets.DateOffset(hours=2), air_leak1_dates["end"],
                        closed="both"),
            pd.Interval(air_leak2_dates["start"] - pd.offsets.DateOffset(hours=2), air_leak2_dates["end"],
                        closed="both"),
            pd.Interval(oil_leak_dates["start"] - pd.offsets.DateOffset(hours=2), oil_leak_dates["end"],
                        closed="both")]

    with open("data/all_cycles_dates.pkl", "rb") as cycle_dates:
        all_cycles_dates = pkl.load(cycle_dates)

    alpha_range = np.append(np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1, 0.1))
    argument_dict = argument_parser()
    time_granularity = {"hours": argument_dict.hours, "days": argument_dict.days, "minutes": argument_dict.minutes}
    all_intervals = generate_intervals(time_granularity)
    cycles_per_interval = map_cycles_to_intervals(all_intervals, all_cycles_dates)

    with open(argument_dict.file, "rb") as loss_file:
        tl = pkl.load(loss_file)
        test_losses = tl["test"]
        train_losses = tl["train"]

    complete_losses = np.append(train_losses, test_losses)
    average_loss_per_interval = np.array([np.mean(complete_losses[interval]) for interval in cycles_per_interval])
    train_time = pd.Interval(pd.to_datetime('03-01-2022 03:59:59', dayfirst=True),
                             pd.to_datetime('01-02-2022 03:59:59', dayfirst=True))
    train_intervals = np.where([interval.overlaps(train_time) for interval in all_intervals])[0]
    train_interval_losses = average_loss_per_interval[train_intervals]
    test_interval_losses = average_loss_per_interval[train_intervals[-1]+1:]
    anomaly_threshold = extreme_anomaly(train_interval_losses)
    binary_output = np.array(np.array(test_interval_losses) > anomaly_threshold, dtype=int)

    output_lpf_binary = {alpha: np.array(np.array(simple_lowpass_filter(binary_output, alpha)) > 0.5, dtype=int)
                         for alpha in alpha_range}

    dicts_early = {}
    dicts_both = {}

    for alpha in alpha_range:
        if argument_dict.print_flag:
            print(f"[Alpha: {alpha:.2f}]")
        d_both = output_metrics(output_lpf_binary[alpha], both, all_intervals[train_intervals[-1]+1:],
                                print_label="Early and actual anomaly", print_flag=argument_dict.print_flag)
        dicts_both[alpha] = d_both

    for alpha in alpha_range:
        if argument_dict.print_flag:
            print(f"[Alpha: {alpha:.2f}]")
        d_early = output_metrics(output_lpf_binary[alpha], early_detection, all_intervals[train_intervals[-1]+1:],
                                 print_flag=argument_dict.print_flag)
        dicts_early[alpha] = d_early

    best_alpha("f1", "F1 - Early detection", dicts_early)
    best_alpha("f1", "F1 - Early and actual anomaly", dicts_both)
