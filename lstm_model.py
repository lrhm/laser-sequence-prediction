
import torch as t


class VanillaLSTM(t.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = t.nn.LSTM(input_size, hidden_size)
        self.act = t.nn.Tanh()
        # self.linear = t.nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        # tanh on output
        output = self.act(output)
        return output, hidden

    def init_hidden(self):
        return t.autograd.Variable(t.zeros(1, 1, self.hidden_size))


