from torch.distributions.multivariate_normal import MultivariateNormal
import torch as th
from TCN_AAE import SimpleDiscriminator_TCN, LSTMDiscriminator_TCN, ConvDiscriminator_TCN
from LSTM_AAE import LSTMDiscriminator
import pickle as pkl
import numpy as np
import sys

#LSTM_enc_discriminators = ["results/final_chunks_offline_WAE_discriminator_LSTMDiscriminator_analog_feats_6_4_10.0_250_0.0001_64.pt",
#                           "results/final_chunks_offline_WAE_discriminator_LSTMDiscriminator_analog_feats_6_4_10.0_200_0.0002_64.pt"]

#disc_models = [LSTMDiscriminator(6, 0.2, disc_hidden=6, n_layers=4),
#               LSTMDiscriminator(6, 0.2, disc_hidden=64, n_layers=4)]

multivariate_normal = MultivariateNormal(th.zeros(4), th.eye(4))

#discriminator_random_scores = {}

#for i, discriminator in enumerate(disc_models):
#    discriminator = discriminator.to(th.device("cuda"))
#    discriminator.load_state_dict(th.load(LSTM_enc_discriminators[i], map_location=th.device("cuda")))
#    discriminator.eval()
#    with th.no_grad():
#        disc_scores = []
#        for _ in range(10000):
#            disc_scores.append(discriminator(multivariate_normal.sample(th.Size([1, 1])).to(th.device("cuda"))).item())
#    discriminator_random_scores[LSTM_enc_discriminators[i]] = disc_scores


#TCN_enc_discriminators = ["results/final_chunks_offline_WAE_discriminator_LSTMDiscriminator_TCN_analog_feats_6_7_10.0_200_0.0001_64.pt", #LSTMDiscriminator_TCN(6, 0.2, disc_hidden=32, n_layers=4)
#                          "results/final_chunks_offline_WAE_discriminator_LSTMDiscriminator_TCN_analog_feats_6_7_10.0_150_0.0001_64.pt",
#                          "results/final_chunks_offline_WAE_discriminator_SimpleDiscriminator_TCN_analog_feats_6_7_10.0_200_0.0001_64.pt",
#                          "results/final_chunks_offline_WAE_discriminator_SimpleDiscriminator_TCN_analog_feats_6_10_5.0_200_1e-05_64.pt",
#                          "results/final_chunks_offline_WAE_discriminator_SimpleDiscriminator_TCN_analog_feats_6_10_10.0_200_1e-05_64.pt",
#                          "results/final_chunks_offline_WAE_discriminator_SimpleDiscriminator_TCN_analog_feats_6_10_0.5_200_1e-05_64.pt",
#                          "results/final_chunks_offline_WAE_discriminator_LSTMDiscriminator_TCN_analog_feats_6_10_5.0_200_1e-05_64.pt",
#                          "results/final_chunks_offline_WAE_discriminator_LSTMDiscriminator_TCN_analog_feats_6_10_10.0_200_1e-05_64.pt",
#                          "results/final_chunks_offline_WAE_discriminator_LSTMDiscriminator_TCN_analog_feats_6_10_0.5_200_1e-05_64.pt",
#                          "results/final_chunks_offline_WAE_discriminator_ConvDiscriminator_TCN_analog_feats_6_7_10.0_200_1e-05_64.pt"]

#disc_models = [LSTMDiscriminator_TCN(6, 0.2, disc_hidden=32, n_layers=4),
#               LSTMDiscriminator_TCN(6, 0.2, disc_hidden=32, n_layers=2),
#               SimpleDiscriminator_TCN(6, 0.2, disc_hidden=64, n_layers=3, window_size=1800),
#               SimpleDiscriminator_TCN(6, 0.2, disc_hidden=32, n_layers=5, window_size=1800),
#               SimpleDiscriminator_TCN(6, 0.2, disc_hidden=32, n_layers=5, window_size=1800),
#               SimpleDiscriminator_TCN(6, 0.2, disc_hidden=32, n_layers=5, window_size=1800),
#               LSTMDiscriminator_TCN(6, 0.2, disc_hidden=32, n_layers=5, window_size=1800),
#               LSTMDiscriminator_TCN(6, 0.2, disc_hidden=32, n_layers=5, window_size=1800),
#               LSTMDiscriminator_TCN(6, 0.2, disc_hidden=32, n_layers=5, window_size=1800),
#               ConvDiscriminator_TCN(6, 0.2, disc_hidden=30, n_layers=7, kernel_size= 7, window_size=1800)]

model_file = sys.argv[1]

disc_hidden = int(model_file.split("_")[15])
n_layers = int(model_file.split("_")[14])

discriminator = LSTMDiscriminator_TCN(4, 0.2, disc_hidden=disc_hidden, n_layers=n_layers).to(th.device("cuda"))
discriminator.load_state_dict(th.load(model_file, map_location=th.device("cuda")))
discriminator.eval()
with th.no_grad():
    disc_scores = []
    for _ in range(10000):
        disc_scores.append(discriminator(multivariate_normal.sample(th.Size([1, 1800])).to(th.device("cuda"))).item())
print(np.min(disc_scores), np.median(disc_scores), np.max(disc_scores))

