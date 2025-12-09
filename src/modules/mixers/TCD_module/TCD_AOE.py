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

    def forward(self, x, mask):  # x是全局观测(2272,5,64),mask是矩阵(2272,5,5)
        v = F.relu(self.fcv(x))  # (2272,5,64)
        q = F.relu(self.fcq(x))  # (2272,5,64)
        k = F.relu(self.fck(x)).permute(0, 2, 1)  # 维度转换，将k调整为(2272,64,5)

        # 原版
        # att = F.softmax(torch.mul(torch.bmm(q, k), mask) - 9e15 * (1 - mask), dim=2)
        # GPT版
        att = torch.bmm(q, k)
        att = att.masked_fill(mask == 0, -1e9)
        att = F.softmax(att, dim=2)

        out = torch.bmm(att, v)  # (2272,5,64)

        return out


class ATCPred(nn.Module):
    def __init__(self, n_agent, input_dim, hidden_dim, latent_dim):
        super(ATCPred, self).__init__()
        # 编码器部分
        self.hidden_dim = hidden_dim
        self.n_agent = n_agent
        self.L1 = nn.Linear(input_dim, hidden_dim)
        self.att = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.compress = nn.Linear(hidden_dim, latent_dim)
        self.residual_predictor = nn.Linear(latent_dim, latent_dim)
        self.output_layer = nn.Linear(latent_dim, latent_dim)
        self.relu = nn.ReLU()  # 使用 ReLU 作为激活函数

    def forward(self, x, mask):
        x = self.relu(self.L1(x))
        x = self.att(x, mask)
        z = self.relu(x)
        c = self.relu(self.compress(z))
        p = self.relu(self.residual_predictor(c)) + c
        return self.output_layer(p)


# self.args.n_agents, self.args.obs_shape, self.args.RND_hidden_dim
class ATCTarget(nn.Module):
    def __init__(self, n_agent, input_dim, hidden_dim, latent_dim, out_dim=None):
        super(ATCTarget, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_agent = n_agent
        # 编码器部分
        self.L1 = nn.Linear(input_dim, hidden_dim)
        self.att = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.compress = nn.Linear(hidden_dim, latent_dim)
        self.W = nn.Linear(latent_dim, latent_dim, bias=False)  # 对比变换矩阵
        self.relu = nn.ReLU()  # 使用 ReLU 作为激活函数

    def forward(self, x, mask, is_pos=False):
        x = self.relu(self.L1(x))
        x = self.att(x, mask)
        z = self.relu(x)
        c = self.relu(self.compress(z))
        if is_pos:
            c = self.W(c)
        return c
