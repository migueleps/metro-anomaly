from torch import nn
import torch as th
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dims,
                 embedding_dim,
                 dropout,
                 lstm_layers):

        super(Encoder, self).__init__()

        self.dropout = nn.Dropout(dropout)

        input_dims = [input_dim] + hidden_dims
        output_dims = hidden_dims + [embedding_dim]

        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size=input_dims[i],
                                                  hidden_size=output_dims[i],
                                                  batch_first=True) for i in range(lstm_layers)])

    def forward(self, x):
        hidden_outs = []
        for lstm_cell in self.lstm_layers[:-1]:
            x, (_, _) = lstm_cell(x)
            hidden_outs.append(x)
            x = self.dropout(x)
        x, (hidden, _) = self.lstm_layers[-1](x)
        hidden_outs.append(x)
        return hidden, hidden_outs


class Decoder(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dims,
                 output_dim,
                 dropout,
                 lstm_layers):
        super(Decoder, self).__init__()

        self.dropout = nn.Dropout(dropout)

        input_dims = [embedding_dim, embedding_dim] + hidden_dims[:-1]
        output_dims = [embedding_dim] + hidden_dims

        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size=input_dims[i],
                                                  hidden_size=output_dims[i],
                                                  batch_first=True) for i in range(lstm_layers)])

        self.output_layer = nn.Linear(in_features=hidden_dims[-1],
                                      out_features=output_dim)

    def forward(self, x):
        hidden_outs = []
        for lstm_cell in self.lstm_layers:
            x, (_, _) = lstm_cell(x)
            hidden_outs.append(x)
            x = self.dropout(x)
        return self.output_layer(x), hidden_outs


class LSTM_AllLayerSAE(nn.Module):

    def __init__(self,
                 n_features,
                 emb_dim,
                 hidden_dims,
                 dropout,
                 lstm_layers,
                 device,
                 sparsity_weight,
                 sparsity_parameter):

        super(LSTM_AllLayerSAE, self).__init__()

        self.embedding_dim = emb_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.n_features = n_features
        self.device = device
        self.lstm_layers = lstm_layers
        self.sparsity_weight = sparsity_weight
        self.sparsity_parameter = sparsity_parameter

        self.encode = Encoder(input_dim=self.n_features,
                              hidden_dims=self.hidden_dims,
                              embedding_dim=self.embedding_dim,
                              dropout=self.dropout,
                              lstm_layers=self.lstm_layers).to(device)

        self.decode = Decoder(embedding_dim=self.embedding_dim,
                              hidden_dims=self.hidden_dims,
                              output_dim=self.n_features,
                              dropout=self.dropout,
                              lstm_layers=self.lstm_layers).to(device)

    def sparsity_penalty(self, activations):
        average_activation = th.mean(th.abs(activations.squeeze()), 0)
        target_activations = th.tensor([self.sparsity_parameter] * average_activation.shape[0]).to(self.device)
        kl_div_part1 = th.log(target_activations / average_activation)
        kl_div_part2 = th.log((1 - target_activations) / (1 - average_activation))
        return th.sum(self.sparsity_parameter * kl_div_part1 + (1 - self.sparsity_parameter) * kl_div_part2)

    def forward(self, x):
        n_examples = x.shape[1]
        assert x.shape[2] == self.n_features

        latent_vector, hidden_outs_encoder = self.encode(x)

        stacked_LV = latent_vector.repeat(1, n_examples, 1).to(self.device)
        reconstructed_x, hidden_outs_decoder = self.decode(stacked_LV)

        loss = F.mse_loss(reconstructed_x, x)
        if self.training:
            for activations in hidden_outs_encoder+hidden_outs_decoder:
                loss += self.sparsity_penalty(activations)

        return loss, reconstructed_x
