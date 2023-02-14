import sys
import os

epochs = [100, 150, 200, 250]
lrs = [1e-3, 1e-4, 1e-2]
emb_size = {"lstm_ae": [4, 6],
            "lstm_sae": [16, 32, 64],
            "multi_enc_sae": [16, 32, 64],
            "multi_enc_ae": [4, 6],
            "diff_comp_sae": [16, 32, 64],
            "diff_comp_ae": [4, 6]}

reg_string = {"lstm_ae": "",
              "lstm_sae": "-l2reg 0.00001",
              "multi_enc_sae": "-l2reg 0.00001",
              "multi_enc_ae": "",
              "diff_comp_sae": "-l2reg 0.00001",
              "diff_comp_ae": ""}

sep_string = {"lstm_ae": "",
              "lstm_sae": "",
              "multi_enc_sae": "-separate_comp",
              "multi_enc_ae": "-separate_comp",
              "diff_comp_sae": "-separate_comp",
              "diff_comp_ae": "-separate_comp"}

lstm_layers = [2, 3, 4, 5]

model = sys.argv[1]

base_string = lambda embedding, layers, n_epochs, learning_rate, reg, separate: f"python offline_only_train.py \
-model {model} -embedding {embedding} -n_layers {layers} -epochs {n_epochs} -lr {learning_rate} \
{reg_string[model]} {sep_string[model]} -force-training"

for lr in lrs:
    for nepochs in epochs:
        for emb in emb_size[model]:
            for nlayers in lstm_layers:
                os.system(base_string(emb, nlayers, nepochs, lr))
