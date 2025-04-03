import torch
import torch.nn as nn



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * sequence_length, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
            )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size)
        out, _ = self.rnn(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)

        return out
