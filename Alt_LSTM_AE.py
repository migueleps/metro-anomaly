from torch import nn
import torch as th
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 dropout,
                 lstm_layers):

        super(Encoder, self).__init__()

        self.lstm_layers = nn.LSTM(input_size=input_dim,
                                   hidden_size=embedding_dim,
                                   batch_first=True,
                                   num_layers=lstm_layers,
                                   dropout=dropout)

    def forward(self, x):
        hidden_outs, (_, _) = self.lstm_layers(x)
        return hidden_outs


class Decoder(nn.Module):

    def __init__(self,
                 embedding_dim,
                 output_dim,
                 dropout,
                 lstm_layers):
        super(Decoder, self).__init__()

        self.lstm_layers = nn.LSTM(input_size=embedding_dim,
                                   hidden_size=embedding_dim,
                                   batch_first=True,
                                   num_layers=lstm_layers,
                                   bidirectional=True,
                                   dropout=dropout)

        self.output_layer = nn.Linear(in_features=2*embedding_dim,
                                      out_features=output_dim)

    def forward(self, x):
        x, (_, _) = self.lstm_layers(x)
        return self.output_layer(x)


class Alt_LSTM_AE(nn.Module):

    def __init__(self,
                 n_features,
                 emb_dim,
                 dropout,
                 lstm_layers,
                 device, *kwargs):

        super(Alt_LSTM_AE, self).__init__()

        self.embedding_dim = emb_dim
        self.dropout = dropout
        self.n_features = n_features
        self.device = device
        self.lstm_layers = lstm_layers

        self.encode = Encoder(input_dim=self.n_features,
                              embedding_dim=self.embedding_dim,
                              dropout=self.dropout,
                              lstm_layers=self.lstm_layers).to(device)

        self.decode = Decoder(embedding_dim=self.embedding_dim,
                              output_dim=self.n_features,
                              dropout=self.dropout,
                              lstm_layers=self.lstm_layers).to(device)

    def forward(self, x):

        reconstructed_x = self.decode(self.encode(x))

        loss = F.mse_loss(reconstructed_x, x)

        return loss, reconstructed_x

