import numpy as np
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class LSTM(nn.Module):
    """An LSTM."""
    def __init__(self, num_inputs, num_outputs, num_hid, num_layers):
        super(LSTM, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hid = num_hid
        self.num_layers = num_layers
        self.prev_state = None

        print('!!! USING VANILLA LSTM !!!')
        self.lstm = nn.LSTM(input_size=self.num_inputs,
                            hidden_size=self.num_hid,
                            num_layers=self.num_layers)
        self.fc = nn.Linear(self.num_hid, self.num_outputs, bias=True)


        # The hidden state is a learned parameter
        self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_hid) * 0.05)
        self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_hid) * 0.05)

        self.reset_parameters()

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.prev_state = self.create_new_state(batch_size)

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))
                nn.init.uniform(p, -stdev, stdev)

        for p in self.fc.parameters():
            if p.dim() == 1:
                nn.init.constant(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))
                nn.init.uniform(p, -stdev, stdev)

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x=None):
        if x is None:
            x = Variable(torch.zeros(self.batch_size, self.num_inputs))

        x = x.unsqueeze(0)
        o, self.prev_state = self.lstm(x, self.prev_state)
        o = self.fc(o)
        o = F.sigmoid(o)

        return o.squeeze(0), self.prev_state


    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params


