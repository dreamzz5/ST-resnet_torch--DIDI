import torch.nn as nn
import torch
rnn = nn.LSTM(10, 20, 1)
input = torch.randn(5, 1, 10)
h0 = torch.randn(2, 1, 20)
c0 = torch.randn(2, 1, 20)
output, (hn, cn) = rnn(input, (h0, c0))