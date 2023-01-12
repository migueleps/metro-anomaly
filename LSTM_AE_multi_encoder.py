import torch as th
from torch import nn
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
        for lstm_cell in self.lstm_layers[:-1]:
            x, (_, _) = lstm_cell(x)
            x = self.dropout(x)
        hidden_outs, (hidden, _) = self.lstm_layers[-1](x)
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
        for lstm_cell in self.lstm_layers:
            x, (_, _) = lstm_cell(x)
            x = self.dropout(x)
        return self.output_layer(x)


class LSTM_AE_MultiEncoder(nn.Module):

    def __init__(self,
                 n_features,
                 emb_dim,
                 hidden_dims,
                 dropout,
                 lstm_layers,
                 device,
                 *kwargs):

        super(LSTM_AE_MultiEncoder, self).__init__()

        self.embedding_dim = emb_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.n_features = n_features
        self.device = device
        self.lstm_layers = lstm_layers

        self.encode_comp0 = Encoder(input_dim=self.n_features,
                                    hidden_dims=self.hidden_dims,
                                    embedding_dim=self.embedding_dim,
                                    dropout=self.dropout,
                                    lstm_layers=self.lstm_layers).to(device)

        self.encode_comp1 = Encoder(input_dim=self.n_features,
                                    hidden_dims=self.hidden_dims,
                                    embedding_dim=self.embedding_dim,
                                    dropout=self.dropout,
                                    lstm_layers=self.lstm_layers).to(device)

        self.decode = Decoder(embedding_dim=self.embedding_dim,
                              hidden_dims=self.hidden_dims,
                              output_dim=self.n_features,
                              dropout=self.dropout,
                              lstm_layers=self.lstm_layers).to(device)

    def forward(self, x):

        comp0, comp1 = x

        n_examples_comp0 = comp0.shape[1]
        assert comp0.shape[2] == self.n_features

        n_examples_comp1 = comp1.shape[1]
        assert comp1.shape[2] == self.n_features

        total_n_examples = n_examples_comp0 + n_examples_comp1

        latent_vector0, hidden_outs0 = self.encode_comp0(comp0)
        latent_vector1, hidden_outs1 = self.encode_comp1(comp1)

        latent_vector = th.cat((latent_vector0, latent_vector1))

        stacked_LV = th.repeat_interleave(latent_vector, total_n_examples,
                                          dim=1).reshape(-1, total_n_examples, self.embedding_dim).to(self.device)
        reconstructed_x = self.decode(stacked_LV)

        original_cycle = th.cat((comp0, comp1), dim=1)
        loss = F.mse_loss(reconstructed_x, original_cycle)

        return loss, reconstructed_x
