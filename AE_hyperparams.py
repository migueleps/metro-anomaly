import os
import sys

epochs = 150
lrs = [0.001, 0.0001]

lstm_params = lambda n_layers: f"-n_layers {n_layers}"
tcn_params = lambda n_layers, kernel_size, hidden_units: f"-n_layers {n_layers} -tcn_kernel {kernel_size} -tcn_hidden {hidden_units}"

tcn_layers = [(7, 9, 6), (7, 9, 30), (10, 3, 6), (10, 3, 30)]

lstm_layers = range(1, 6)

embedding_sizes = [4, 6]
model = sys.argv[1]

base_string = lambda learning_rate, embedding_dim, enc_dec_params: f"python final_chunks_offile.py \
-feats noflow -model {model} -embedding {embedding_dim} -epochs {epochs} -lr {learning_rate} -batch_size 64 \
{enc_dec_params} -force-training"

for lr in lrs:
    for emb_size in embedding_sizes:
        to_iterate = lstm_layers if "lstm" in model else tcn_layers
        for elem in to_iterate:
            enc_dec_param_string = lstm_params(elem) if "lstm" in model else tcn_params(elem[0], elem[1], elem[2])
            os.system(base_string(lr, emb_size, enc_dec_param_string))
