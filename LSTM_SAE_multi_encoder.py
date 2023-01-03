import torch as th
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, input_dim, embedding_dim, dropout, lstm_layers):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = embedding_dim

        self.lstm = nn.LSTM(input_size = self.input_dim,
                            hidden_size = self.emb_dim,
                            batch_first = True,
                            num_layers = lstm_layers,
                            dropout = dropout)

    def forward(self, x):
        activations, (hidden, _) = self.lstm(x)
        return hidden[-1], activations


class Decoder(nn.Module):

    def __init__(self, embedding_dim, dropout, lstm_layers):
        super(Decoder, self).__init__()

        self.emb_dim = embedding_dim

        self.lstm = nn.LSTM(input_size = self.emb_dim,
                            hidden_size = self.emb_dim,
                            batch_first = True,
                            num_layers = lstm_layers,
                            dropout = dropout)

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        return x


class LSTM_SAE_MultiEncoder(nn.Module):

    def __init__(self, n_features, emb_dim, dropout, lstm_layers, device, sparsity_weight, sparsity_parameter):
        super(LSTM_SAE_MultiEncoder, self).__init__()

        self.embedding_dim = emb_dim
        self.dropout = dropout
        self.n_features = n_features
        self.device = device
        self.sparsity_weight = sparsity_weight
        self.sparsity_parameter = sparsity_parameter

        self.encode_comp0 = Encoder(input_dim = n_features,
                                    embedding_dim = self.embedding_dim,
                                    dropout = self.dropout,
                                    lstm_layers = lstm_layers).to(device)

        self.encode_comp1 = Encoder(input_dim = n_features,
                                    embedding_dim = self.embedding_dim,
                                    dropout = self.dropout,
                                    lstm_layers = lstm_layers).to(device)

        self.decode = Decoder(embedding_dim = 2*self.embedding_dim,
                              dropout = self.dropout,
                              lstm_layers = lstm_layers).to(device)

        self.output_layer = nn.Linear(in_features = 2*self.embedding_dim,
                                      out_features = self.n_features)

    def sparsity_penalty(self, activations):
        average_activation = th.mean(th.abs(activations), 0)
        target_activations = th.tensor([self.sparsity_parameter] * average_activation.shape[0]).to(self.device)
        kl_div_part1 = th.log(target_activations/average_activation)
        kl_div_part2 = th.log((1-target_activations)/(1-average_activation))
        return th.sum(self.sparsity_parameter * kl_div_part1 + (1-self.sparsity_parameter) * kl_div_part2)

    def forward(self, comp0, comp1):

        unpacked_original0, seq_lengths0 = pad_packed_sequence(comp0, batch_first = True)
        unpacked_original1, seq_lengths1 = pad_packed_sequence(comp1, batch_first = True)

        latent_vector0, activations0 = self.encode_comp0(comp0)
        latent_vector1, activations1 = self.encode_comp1(comp1)

        sparsity_loss = 0
        unpacked_activations, _ = pad_packed_sequence(activations0, batch_first=True)
        for i,activation_tensor in enumerate(unpacked_activations):
            sparsity_loss += self.sparsity_penalty(activation_tensor[:seq_lengths0[i]])

        unpacked_activations, _ = pad_packed_sequence(activations1, batch_first=True)
        for i,activation_tensor in enumerate(unpacked_activations):
            sparsity_loss += self.sparsity_penalty(activation_tensor[:seq_lengths1[i]])

        sparsity_loss = self.sparsity_weight * sparsity_loss

        seq_lengths_decoder = seq_lengths0 + seq_lengths1
        max_seq_length = max(seq_lengths_decoder)
        new_mini_batch = []
        for i in range(latent_vector0.shape[0]):
            tensor_comp0 = latent_vector0[i]
            tensor_comp1 = latent_vector1[i]
            tensor = th.cat((tensor_comp0, tensor_comp1))
            padded_tensor = th.cat([tensor.repeat(seq_lengths_decoder[i], 1),
                                    th.zeros(max_seq_length - seq_lengths_decoder[i],
                                             2*self.embedding_dim).to(self.device)])
            new_mini_batch.append(padded_tensor)

        mini_batch_LV = th.stack(new_mini_batch).to(self.device)

        packed_LV = pack_padded_sequence(mini_batch_LV, seq_lengths_decoder, batch_first=True, enforce_sorted=False)

        decoded_x = self.decode(packed_LV)

        unpacked_decoded, _ = pad_packed_sequence(decoded_x, batch_first=True)

        reconstructions = []
        mse_loss = 0
        for i, tensor in enumerate(unpacked_decoded):
            linear_out = self.output_layer(tensor[:seq_lengths_decoder[i]])
            original_cycle = th.cat((unpacked_original0[i, :seq_lengths0[i]], unpacked_original1[i, :seq_lengths1[i]]))
            mse_loss += F.mse_loss(linear_out, original_cycle)
            reconstructions.append(linear_out)

        avg_mse_loss = mse_loss/seq_lengths_decoder.shape[0]
        loss = avg_mse_loss + sparsity_loss if self.training else avg_mse_loss

        return loss, reconstructions
