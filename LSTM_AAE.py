from torch import nn
import torch as th
from torch.nn.utils import weight_norm


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
        hidden_outs, (hidden, _) = self.lstm_layers(x)
        return hidden[-1]


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

        self.output_layer = nn.Linear(in_features=2 * embedding_dim,
                                      out_features=output_dim)

    def forward(self, x):
        x, (_, _) = self.lstm_layers(x)
        return self.output_layer(x)


class SimpleDiscriminator(nn.Module):

    def __init__(self, input_dim, dropout):
        super(SimpleDiscriminator, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation_func = nn.LeakyReLU()

        self.fc1 = nn.Linear(in_features=input_dim,
                             out_features=128)
        self.fc2 = nn.Linear(in_features=128,
                             out_features=64)
        self.fc3 = nn.Linear(in_features=64,
                             out_features=32)
        self.output_layer = nn.Linear(in_features=32,
                                      out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_func(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation_func(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.activation_func(x)
        x = self.dropout(x)
        return th.sigmoid(self.output_layer(x))


class LSTMDiscriminator(nn.Module):

    def __init__(self, input_dim, dropout):
        super(LSTMDiscriminator, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=64,
                            dropout=dropout,
                            bidirectional=True,
                            batch_first=True,
                            num_layers=2)

        self.output_layer = nn.Linear(in_features=128,
                                      out_features=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        return th.sigmoid(self.output_layer(x))


######################################################################################
#
#    TCN from https://github.com/locuslab/TCN/
#
######################################################################################


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.padding = padding
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        x = x[:, :, :-self.padding].contiguous()
        x = self.dropout(self.relu(x))
        x = self.conv2(x)
        x = x[:, :, :-self.padding].contiguous()
        out = self.dropout(self.relu(x))

        return self.relu(out + res)


class ConvDiscriminator(nn.Module):

    def __init__(self, input_dim, dropout):
        super(ConvDiscriminator, self).__init__()
        # use same default parameters as TCN repo

        hidden_dim = 30
        num_layers = 8
        channels = num_layers * [hidden_dim]
        self.TCN_layers = []
        kernel_size = 5

        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else channels[i - 1]
            out_channels = channels[i]
            padding = (kernel_size - 1) * dilation
            self.TCN_layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                                 dilation=dilation, padding=padding, dropout=dropout))

        self.output_layer = nn.Linear(in_features=channels[-1],
                                      out_features=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        for layer in self.TCN_layers:
            x = layer(x)

        return th.sigmoid(self.output_layer(x))
