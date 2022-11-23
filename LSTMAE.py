import torch as th
from torch import nn


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, embedding_dim, dropout):

        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)

        self.lstm1 = nn.LSTM(input_size = self.input_dim,
                            hidden_size = self.hidden_dim,
                            batch_first = True)
        self.lstm2 = nn.LSTM(input_size = self.hidden_dim,
                            hidden_size = embedding_dim,
                            batch_first = True)


    def forward(self, x):
        x, (_, _) = self.lstm1(x)
        x = self.dropout(x)
        _, (hidden, _) = self.lstm2(x)
        return hidden


class Decoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout):
        super(Decoder, self).__init__()

        self.emb_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.dropout = nn.Dropout(dropout)
        self.lstm1 = nn.LSTM(input_size = self.emb_dim,
                            hidden_size = self.emb_dim,
                            batch_first = True)
        self.lstm2 = nn.LSTM(input_size = self.emb_dim,
                            hidden_size = self.hidden_dim,
                            batch_first = True)
        self.output_layer = nn.Linear(in_features = self.hidden_dim,
                                      out_features = self.output_dim)

    def forward(self, x):
        x, (_, _) = self.lstm1(x)
        x = self.dropout(x)
        x, (_, _) = self.lstm2(x)
        return self.output_layer(x)


class LSTM_AE(nn.Module):

    def __init__(self, n_features, emb_dim, hidden_dim, dropout, device):
        super(LSTM_AE,self).__init__()

        self.embedding_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_features = n_features
        self.device = device

        self.encode = Encoder(input_dim = n_features,
                              hidden_dim = self.hidden_dim,
                              embedding_dim = self.embedding_dim,
                              dropout = self.dropout).to(device)

        self.decode = Decoder(embedding_dim = self.embedding_dim,
                              hidden_dim = self.hidden_dim,
                              output_dim = n_features,
                              dropout = self.dropout).to(device)

    def forward(self, x):

        n_examples = x.shape[1]
        assert x.shape[2] == self.n_features

        latent_vector = self.encode(x)
        stacked_LV = latent_vector.repeat(1,n_examples,1).to(device)
        reconstructed_x = self.decode(stacked_LV)
        return reconstructed_x
