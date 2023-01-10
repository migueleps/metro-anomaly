import numpy as np
import pickle as pkl
import sys

ground_truth_early = [{1882, 1883, 1884, 1885, 1886, 1887}, {3389, 3390, 3391, 3392, 3393, 3394},
                      {9098, 9099, 9100, 9101, 9102, 9103, 9104, 9105}]

ground_truth_both = [{1882, 1883, 1884, 1885, 1886, 1887, 1888}, {3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396},
                     set(list(range(9098, 9357)))]

alpha_range = np.append(np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1, 0.1))


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
    return q75 + 3 * (q75 - q25)


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


def evaluate_model(failures, ground_truth):
    tp, fn, fp = 0, 0, 0
    for failure in failures:
        if not any(map(lambda x: bool(failure & x), ground_truth)):
            fp += 1

    for true_failure in ground_truth:
        if any(map(lambda x: bool(true_failure & x), failures)):
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


def output_metrics(bin_output, ground_truth, print_flag=True, print_label="Early Detection"):
    failures = detect_failures(bin_output)
    metrics = evaluate_model(failures, ground_truth)
    if print_flag:
        print(f"[{print_label}] False Positives: {metrics['fp']}\nPrecision: {prec(metrics):.3f}\n\
Recall: {rec(metrics):.3f}\nF1: {f1(metrics):.3f}\n")
    return dict(precision=prec(metrics), recall=rec(metrics), f1=f1(metrics), false_positives=metrics['fp'],
                false_negatives=metrics['fn'])


def best_alpha(metric, metric_label, dictionary):
    best_metric, alpha_of_metric = 0, 0
    for alpha in alpha_range:
        value = dictionary[alpha][metric]
        if value > best_metric:
            best_metric = value
            alpha_of_metric = alpha
    print(f"Best {metric_label} was {best_metric:.3f} with alpha = {alpha_of_metric:.2f}")


if __name__ == "__main__":
    input_file = sys.argv[1]
    with open(input_file, "rb") as loss_file:
        tl = pkl.load(loss_file)
        test_losses = tl["test"]
        train_losses = tl["train"]
    anomaly_value = extreme_anomaly(train_losses)
    binary_test_losses = np.array(np.array(test_losses) > anomaly_value, dtype=int)

    output_lpf_binary = {alpha: np.array(np.array(simple_lowpass_filter(binary_test_losses, alpha)) > 0.5, dtype=int)
                         for alpha in alpha_range}

    dicts_early = {}
    dicts_both = {}

    for alpha in alpha_range:
        print(f"[Alpha: {alpha:.2f}]")
        d_both = output_metrics(output_lpf_binary[alpha], ground_truth_both, print_label="Early and actual anomaly")
        dicts_both[alpha] = d_both

    for alpha in alpha_range:
        print(f"[Alpha: {alpha:.2f}]")
        d_early = output_metrics(output_lpf_binary[alpha], ground_truth_early)
        dicts_early[alpha] = d_early

    best_alpha("f1", "F1 - Early detection", dicts_early)
    best_alpha("f1", "F1 - Early and actual anomaly", dicts_both)
    best_alpha("recall", "Recall - Early detection", dicts_early)
    best_alpha("recall", "Recall - Early and actual anomaly", dicts_both)

