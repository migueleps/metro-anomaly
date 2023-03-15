import sys
import os

epochs = 150
lr = float(sys.argv[1])
disc_lr = [1, 0.5, 0.1]
emb_size = 4

disc_params_string = lambda n_layers, hidden_units: f"-disc_hidden {hidden_units} -disc_layers {n_layers}"
enc_dec_params = lambda n_layers, kernel_size, hidden_units: f"-n_layers {n_layers} -tcn_kernel {kernel_size} -tcn_hidden {hidden_units}"

encdec_layers = [(7, 9), (10, 3)]
encdec_hidden_units = [6, 30]

disc_layers = [1, 2, 3]
disc_hidden = [6, 32]

model = "LSTMDiscriminator_TCN"

base_string = lambda discriminator_lr, disc_params, enc_dec_params: f"python final_chunks_offile.py \
-encoder TCN -decoder TCN -model {model} -embedding {emb_size} -epochs {epochs} -lr {lr} -batch_size 64 \
-disc_lr {discriminator_lr} {disc_params} {enc_dec_params} -use_discriminator -WAEreg 10 -force-training"

for r in disc_lr:
    dl = r * lr
    for tcn_layers, tcn_kernel in encdec_layers:
        for encdec_hidden in encdec_hidden_units:
            enc_dec_param_string = enc_dec_params(tcn_layers, tcn_kernel, encdec_hidden)
            for discriminator_layers in disc_layers:
                for discriminator_hidden in disc_hidden:
                    disc_param_string = disc_params_string(discriminator_layers, discriminator_hidden)
                    os.system(base_string(dl, enc_dec_param_string, disc_param_string))
