import torch as th
import torch.nn as nn


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
        hidden_outs, _ = self.lstm_layers(x)
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
                                   dropout=dropout,
                                   bidirectional=True)

        self.output_layer = nn.Linear(in_features=2*embedding_dim,
                                      out_features=output_dim)

    def forward(self, x):
        x, (_, _) = self.lstm_layers(x)
        return self.output_layer(x)


def generate_square_subsequent_mask(dim):
    return th.triu(th.ones(dim, dim) * float("-inf"), diagonal=1)


class CriticEncoder(nn.Module):

    def __init__(self, embedding_dim, lstm_layers, attention_heads, dropout):

        super(CriticEncoder, self).__init__()

        self.lstm_layers = nn.LSTM(input_size=embedding_dim,
                                   hidden_size=2*embedding_dim,
                                   batch_first=True,
                                   num_layers=lstm_layers,
                                   dropout=dropout)
        self.attention_layer = nn.MultiheadAttention(embed_dim=2*embedding_dim,
                                                     num_heads=attention_heads,
                                                     dropout=dropout,
                                                     batch_first=True)
        self.output_layer = nn.Linear(in_features=2*embedding_dim,
                                      out_features=1)

    def forward(self, x):

        x, (_, _) = self.lstm_layers(x)
        att_mask = generate_square_subsequent_mask(x.shape[1])
        _, attention_weights = self.attention_layer(x, x, x, attn_mask=att_mask)
        x = attention_weights * x
        return self.output_layer(x)


class CriticDecoder(nn.Module):

    def __init__(self, input_dim, lstm_layers, attention_heads, dropout):

        super(CriticDecoder, self).__init__()
        self.lstm_layers = nn.LSTM(input_size=input_dim,
                                   hidden_size=2*input_dim,
                                   batch_first=True,
                                   num_layers=lstm_layers,
                                   dropout=dropout,
                                   bidirectional=True)
        self.attention_layer = nn.MultiheadAttention(embed_dim=4*input_dim,
                                                     num_heads=attention_heads,
                                                     dropout=dropout,
                                                     batch_first=True)
        self.output_layer = nn.Linear(in_features=4*input_dim,
                                      out_features=1)

    def forward(self, x):
        x, (_, _) = self.lstm_layers(x)
        _, attention_weights = self.attention_layer(x, x, x)
        x = attention_weights * x
        return self.output_layer(x)

