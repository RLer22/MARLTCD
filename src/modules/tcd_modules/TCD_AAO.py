import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import math
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class Encoder(nn.Module):  
    def __init__(self, din=32, hidden_dim=128):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(din, hidden_dim)

    def forward(self, x):  
        embedding = F.relu(self.fc(x))  
        return embedding


# in your AttModel class
class AttModel(nn.Module):
    def __init__(self, n_node, din, hidden_dim, dout):
        super(AttModel, self).__init__()
        self.fcv = nn.Linear(din, hidden_dim)
        self.fck = nn.Linear(din, hidden_dim)
        self.fcq = nn.Linear(din, hidden_dim)

        self.scale = math.sqrt(hidden_dim)

    def forward(self, x, mask, return_attention=False):

        v = self.fcv(x)  
        q = self.fcq(x)
        k = self.fck(x).permute(0, 2, 1)


        att_scores = torch.bmm(q, k) / self.scale


        att_scores = att_scores.masked_fill(mask == 0, -1e9)


        attention_weights = F.softmax(att_scores, dim=2)


        out = torch.bmm(attention_weights, v)


        return out, attention_weights




class CommModel(nn.Module):
    def __init__(self, n_node, din, hidden_dim, dout):
        super(CommModel, self).__init__()
        self.rnn = torch.nn.GRU(input_size=din, hidden_size=int(hidden_dim / 2), bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, dout)
        self.n_node = n_node
        self.din = din

    def forward(self, x, mask):
        size = x.shape
        aid = torch.eye(self.n_node).cuda().unsqueeze(0).expand(size[0], -1, -1).unsqueeze(2).reshape(
            size[0] * self.n_node, 1, self.n_node)
        x = x.unsqueeze(1).expand(-1, self.n_node, -1, -1).reshape(size[0] * self.n_node, size[1], size[2])
        mask = mask.reshape(size[0] * self.n_node, self.n_node).unsqueeze(-1).expand(-1, -1, self.din)
        y = torch.bmm(aid, self.rnn(x * mask)[0]).squeeze(1).reshape(size[0], self.n_node, self.din)
        return y


# in your RNDNetwork class
class RNDNetwork(nn.Module):
    def __init__(self, n_agent, state_size, hidden_dim=64, model_type=1):
        super(RNDNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_agent = n_agent
        self.model_type = model_type
        self.encoder = Encoder(state_size, hidden_dim)
        self.att = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.outLayer = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, state, mask, return_attention=1):
        h1 = self.encoder(state)

        h2, attention_weights = self.att(h1, mask, return_attention=True)
        h3 = self.outLayer(self.relu(h2))
        return h3, attention_weights



