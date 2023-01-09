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

        self.lstm_layers = [nn.LSTM(input_size=input_dims[i],
                                    hidden_size=output_dims[i],
                                    batch_first=True) for i in range(lstm_layers)]

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

        input_dims = [embedding_dim] + hidden_dims
        output_dims = hidden_dims + [output_dim]

        self.lstm_layers = [nn.LSTM(input_size=input_dims[i],
                                    hidden_size=output_dims[i],
                                    batch_first=True) for i in range(lstm_layers)]

    def forward(self, x):
        for lstm_cell in self.lstm_layers[:-1]:
            x, (_, _) = lstm_cell(x)
            x = self.dropout(x)
        hidden_outs, (_, _) = self.lstm_layers[-1](x)
        return hidden_outs


class LSTM_AE(nn.Module):

    def __init__(self,
                 n_features,
                 emb_dim,
                 hidden_dims,
                 dropout,
                 lstm_layers,
                 device, *kwargs):

        super(LSTM_AE, self).__init__()

        self.embedding_dim = emb_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.n_features = n_features
        self.device = device
        self.lstm_layers = lstm_layers

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

    def forward(self, x):
        n_examples = x.shape[1]
        assert x.shape[2] == self.n_features

        latent_vector, hidden_outs = self.encode(x)

        stacked_LV = latent_vector.repeat(1, n_examples, 1).to(self.device)
        reconstructed_x = self.decode(stacked_LV)

        loss = F.mse_loss(reconstructed_x, x)

        return loss, reconstructed_x
