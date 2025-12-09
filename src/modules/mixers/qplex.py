import torch as th
import torch.nn as nn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QPLEXMixer(nn.Module):
    def __init__(self, args):
        super(QPLEXMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w = nn.Linear(self.state_dim, self.n_agents)
            self.hyper_v = nn.Linear(self.state_dim, self.n_agents)
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                         nn.ReLU(),
                                         nn.Linear(hypernet_embed, self.n_agents))
            self.hyper_v = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                         nn.ReLU(),
                                         nn.Linear(hypernet_embed, self.n_agents))
            self.V = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                   nn.ReLU(),
                                   nn.Linear(hypernet_embed, 1))
        else:
            raise Exception("wrong")

    def forward(self, agent_qs, states):
        if agent_qs.dim() == 4 and agent_qs.size(-1) == 1:
            agent_qs = agent_qs.squeeze(-1)

        if agent_qs.dim() == 3:
            bs, ts, n_agents = agent_qs.size()
            flat_agent_qs = agent_qs.view(bs * ts, n_agents)
            out_bs = bs
            out_ts = ts
        elif agent_qs.dim() == 2:
            bs, n_agents = agent_qs.size()
            flat_agent_qs = agent_qs.view(bs, n_agents)
            out_bs = bs
            out_ts = None
        else:
            flat_agent_qs = agent_qs.view(-1, self.n_agents)
            out_bs = None
            out_ts = None

        states_flat = states.reshape(-1, self.state_dim)

        v_i = self.hyper_v(states_flat)
        if flat_agent_qs.size(0) != v_i.size(0):
            flat_agent_qs = flat_agent_qs.expand(v_i.size(0), -1)
        advantages = flat_agent_qs - v_i

        w_logits = self.hyper_w(states_flat)
        w = F.softmax(w_logits, dim=1)

        weighted_adv = (advantages * w).sum(dim=1, keepdim=True)

        v_global = self.V(states_flat)

        y = v_global + weighted_adv

        if out_ts is not None:
            q_tot = y.view(out_bs, out_ts, 1)
        else:
            q_tot = y.view(-1, 1)

        return q_tot
