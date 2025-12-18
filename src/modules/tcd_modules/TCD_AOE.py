import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class AttModel(nn.Module):
    def __init__(self, n_node, din, hidden_dim, dout):
        super(AttModel, self).__init__()
        self.fcv = nn.Linear(din, hidden_dim)
        self.fck = nn.Linear(din, hidden_dim)
        self.fcq = nn.Linear(din, hidden_dim)

    def forward(self, x, mask):  
        v = F.relu(self.fcv(x)) 
        q = F.relu(self.fcq(x)) 
        k = F.relu(self.fck(x)).permute(0, 2, 1)  

       
        att = torch.bmm(q, k)
        att = att.masked_fill(mask == 0, -1e9)
        att = F.softmax(att, dim=2)

        out = torch.bmm(att, v)  

        return out


class ATCPred(nn.Module):
    def __init__(self, n_agent, input_dim, hidden_dim, latent_dim):
        super(ATCPred, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_agent = n_agent
        self.L1 = nn.Linear(input_dim, hidden_dim)
        self.att = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.compress = nn.Linear(hidden_dim, latent_dim)
        self.residual_predictor = nn.Linear(latent_dim, latent_dim)
        self.output_layer = nn.Linear(latent_dim, latent_dim)
        self.relu = nn.ReLU()  

    def forward(self, x, mask):
        x = self.relu(self.L1(x))
        x = self.att(x, mask)
        z = self.relu(x)
        c = self.relu(self.compress(z))
        p = self.relu(self.residual_predictor(c)) + c
        return self.output_layer(p)


class ATCTarget(nn.Module):
    def __init__(self, n_agent, input_dim, hidden_dim, latent_dim, out_dim=None):
        super(ATCTarget, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_agent = n_agent

        self.L1 = nn.Linear(input_dim, hidden_dim)
        self.att = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.compress = nn.Linear(hidden_dim, latent_dim)
        self.W = nn.Linear(latent_dim, latent_dim, bias=False) 
        self.relu = nn.ReLU()  

    def forward(self, x, mask, is_pos=False):
        x = self.relu(self.L1(x))
        x = self.att(x, mask)
        z = self.relu(x)
        c = self.relu(self.compress(z))
        if is_pos:
            c = self.W(c)
        return c
