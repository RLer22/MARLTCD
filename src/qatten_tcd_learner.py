import os
import copy
import torch
import random
import torch as th
import numpy as np
from torch.optim import RMSprop
import torch.nn.functional as F
from modules.mixers.TCD_module.TCD_AAO import AAO_PnT
from modules.mixers.TCD_module.TCD_AOE import AOE_Pred, AOE_Tgt
from components.episode_buffer import EpisodeBatch
from modules.mixers.qatten import QattenMixer


def set_seed(seed=0, env=None):
    # Python, NumPy
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


class Qatten_TCD_Leaner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.params = list(mac.parameters())
        self.last_entropy_log_t = 0
        self.t2 = 0
        self.last_target_update_episode = 0
        set_seed(args.seed)
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "qatten":
                self.mixer = QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.target_atc_model = AOE_Tgt(self.args.n_agents, self.args.obs_shape, self.args.RND_hidden_dim, self.args.RND_hidden_dim).cuda()
        self.pred_atc_model = AOE_Pred(self.args.n_agents, self.args.obs_shape, self.args.RND_hidden_dim, self.args.RND_hidden_dim).cuda()
        for param in self.target_atc_model.parameters():
            param.requires_grad = False
        self.optimiser_atc = th.optim.Adam(self.pred_atc_model.parameters(), lr=self.args.atc_lr)

        self.predictor_rnd_model = AAO_PnT(self.args.n_agents, self.args.obs_shape, self.args.RND_hidden_dim, self.args.model_type).cuda()
        self.target_rnd_model = AAO_PnT(self.args.n_agents, self.args.obs_shape, self.args.RND_hidden_dim, self.args.model_type).cuda()

        for param in self.target_rnd_model.parameters():
            param.requires_grad = False

        self.optimiser_rnd = th.optim.Adam(self.predictor_rnd_model.parameters(), lr=float(4e-5))
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.n_actions = self.args.n_actions

    def generate_negative_samples(self, anchor_data, anchor_adj, neg_samples_num):
        batch_size, time_step, n_agent, data_dim = anchor_data.shape
        batch_size, time_step, n_agent, adj_data_dim = anchor_adj.shape
        negative_samples = torch.zeros((neg_samples_num, batch_size, time_step, n_agent, data_dim), dtype=anchor_data.dtype, device=anchor_data.device)
        neg_adj = torch.zeros((neg_samples_num, batch_size, time_step, n_agent, adj_data_dim), dtype=anchor_adj.dtype, device=anchor_adj.device)
        for idx in range(batch_size):
            neg_indices = np.random.choice([i for i in range(batch_size) if i != idx], size=neg_samples_num, replace=False)
            for neg_sample_idx in range(neg_samples_num):
                negative_samples[neg_sample_idx, idx] = anchor_data[neg_indices[neg_sample_idx]]
                neg_adj[neg_sample_idx, idx] = anchor_adj[neg_indices[neg_sample_idx]]
        return negative_samples, neg_adj

    def atc_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings, temperature=0.9):
        batch_size = anchor_embeddings.shape[0]
        n_agents = anchor_embeddings.shape[1]
        embedding_dim = anchor_embeddings.shape[2]
        neg_nums = negative_embeddings.shape[0]
        positive_scores = (anchor_embeddings * positive_embeddings).sum(dim=2) / temperature
        negative_scores = torch.zeros(neg_nums, batch_size, n_agents).cuda()
        for j in range(neg_nums):
            negative_scores[j] = (anchor_embeddings * negative_embeddings[j]).sum(dim=2) / temperature
        all_scores = torch.cat([positive_scores.unsqueeze(0), negative_scores], dim=0)
        probs = F.softmax(all_scores, dim=0)  # (neg_num + 1, batch_size, n_agents)
        positive_prob = probs[0]  # (batch_size, n_agents)
        contrastive_loss = -torch.mean(torch.log(positive_prob))
        return contrastive_loss

    def sub_train(self,
                  batch: EpisodeBatch,
                  t_env: int,
                  episode_num: int,
                  mac, mixer,
                  optimiser,
                  params,
                  show_demo=False,
                  save_data=None):
        rewards = batch["reward"][:, :-1]

        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        obs_t = batch["obs"][:, :-1]
        adj_t = batch["adj"][:, :-1]
        obs = batch["obs"][:, 1:]
        adj = batch["adj"][:, 1:]

        # (0) data
        neg_samples_num = 3
        negative_obs, negative_adj = self.generate_negative_samples(obs, adj, neg_samples_num)
        negative_obs = negative_obs.reshape(neg_samples_num, -1, self.args.n_agents, self.args.obs_shape)
        negative_adj = negative_adj.reshape(neg_samples_num, -1, self.args.n_agents, self.args.n_agents)

        obs_t = obs_t.reshape(-1, self.args.n_agents, self.args.obs_shape)
        adj_t = adj_t.reshape(-1, self.args.n_agents, self.args.n_agents)
        obs = obs.reshape(-1, self.args.n_agents, self.args.obs_shape)
        adj = adj.reshape(-1, self.args.n_agents, self.args.n_agents)

        anchor_obs = obs
        anchor_adj = adj
        positive_obs = obs_t
        positive_adj = adj_t

        if t_env > self.args.rnd_start_train:

            target_feature, _ = self.target_rnd_model(obs.cuda(), adj.cuda())
            predictor_feature, _ = self.predictor_rnd_model(obs.cuda(), adj.cuda())
            predictor_feature_t, _ = self.predictor_rnd_model(obs_t.cuda(), adj_t.cuda())
            # (3) loss_AAO
            forward_loss = ((target_feature - predictor_feature) ** 2).mean(-1)
            # loss
            if self.args.model_type == 0:
                kl_loss = th.log((predictor_feature_t + 0.0001) / (predictor_feature + 0.0001)).mean(-1)
                loss_rnd = (forward_loss.sum() + self.args.kl_lambda * kl_loss.sum()) / mask.sum()
            if self.args.model_type == 1:
                loss_rnd = forward_loss.sum() / mask.sum()

            if t_env - self.last_entropy_log_t >= 50000:
                self.last_entropy_log_t = t_env
                try:
                    self.predictor_rnd_model.eval()
                    with th.no_grad():
                        _, attention_weights = self.predictor_rnd_model(obs.cuda(), adj.cuda(), return_attention=True)
                        epsilon = 1e-8
                        p_log_p = attention_weights * th.log(attention_weights + epsilon)
                        entropy_per_agent = -th.sum(p_log_p, dim=-1)
                        mean_entropy = th.mean(entropy_per_agent).item()
                        self.logger.log_stat("attention_entropy", mean_entropy, t_env)
                except Exception as e:
                    print(f"Error calculating attention entropy at step {t_env}: {e}")
                finally:
                    self.predictor_rnd_model.train()

            intrinsic_rewards = (self.args.intrinsic_scale * forward_loss).sum(-1)
            intrinsic_rewards = intrinsic_rewards.reshape(rewards.shape[0], rewards.shape[1], rewards.shape[2])
            total_rewards = intrinsic_rewards.clone()
            total_rewards += rewards.cuda()

            mac_out = []
            mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)

            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

            x_mac_out = mac_out.clone().detach()
            x_mac_out[avail_actions == 0] = -9999999
            max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)
            max_action_index = max_action_index.detach().unsqueeze(3)
            is_max_action = (max_action_index == actions).int().float()
            if show_demo:
                q_i_data = chosen_action_qvals.detach().cuda().numpy()
                q_data = (max_action_qvals - chosen_action_qvals).detach().cuda().numpy()

            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            target_mac_out = th.stack(target_mac_out[1:], dim=1)

            target_mac_out[avail_actions[:, 1:] == 0] = -9999999

            if self.args.double_q:
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
                target_max_qvals = target_mac_out.max(dim=3)[0]
                target_next_actions = cur_max_actions.detach()
                cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)).cuda()
                cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
            else:
                target_mac_out = []
                self.target_mac.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    target_agent_outs = self.target_mac.forward(batch, t=t)
                    target_mac_out.append(target_agent_outs)
                target_mac_out = th.stack(target_mac_out[1:], dim=1)
                target_max_qvals = target_mac_out.max(dim=3)[0]

            if mixer is not None:
                ans_chosen, q_attend_regs, head_entropies = mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv, _, _ = mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot, max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv
                if self.args.double_q:
                    target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], actions=cur_max_actions_onehot, max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
                else:
                    target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)

            targets = total_rewards + self.args.gamma * (1 - terminated) * target_max_qvals

            if show_demo:
                tot_q_data = chosen_action_qvals.detach().cpu().numpy()
                tot_target = targets.detach().cpu().numpy()
                print('action_pair_%d_%d' % (save_data[0], save_data[1]),
                      np.squeeze(q_data[:, 0]),
                      np.squeeze(q_i_data[:, 0]),
                      np.squeeze(tot_q_data[:, 0]),
                      np.squeeze(tot_target[:, 0]))
                self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(tot_q_data[:, 0]), t_env)
                return

            td_error = (chosen_action_qvals - targets.detach())
            mask = mask.expand_as(td_error)
            masked_td_error = td_error * mask
            loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
            masked_hit_prob = th.mean(is_max_action, dim=2) * mask
            hit_prob = masked_hit_prob.sum() / mask.sum()

            optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
            optimiser.step()

            # (4)
            if self.args.model_type == 1:
                for param in self.predictor_rnd_model.att.parameters():
                    param.requires_grad = False
            # (5)
            self.optimiser_rnd.zero_grad()
            loss_rnd.backward()
            self.optimiser_rnd.step()

        if t_env < self.args.atc_pretrain:
            # (6)
            if self.args.model_type == 1:
                positive_embeddings = self.target_atc_model(positive_obs.cuda(), positive_adj.cuda())
                a, b, c = positive_embeddings.shape
                negative_embeddings = torch.empty((0, a, b, c), dtype=torch.float, device=positive_embeddings.device)
                for i in range(neg_samples_num):
                    single_neg_obs = negative_obs[i].cuda()
                    single_neg_adj = negative_adj[i].cuda()
                    single_neg_embeddings = self.target_atc_model(single_neg_obs, single_neg_adj)
                    negative_embeddings = torch.cat((negative_embeddings, single_neg_embeddings.unsqueeze(0)), dim=0)

                # (7)
                anchor_embeddings = self.pred_atc_model(anchor_obs.cuda(), anchor_adj.cuda())
                # (8)
                loss_ATC = self.atc_loss(anchor_embeddings, positive_embeddings, negative_embeddings) / mask.sum()
                # (9)
                self.optimiser_atc.zero_grad()
                loss_ATC.backward()
                self.optimiser_atc.step()

                if t_env > self.args.atc_decay:
                    self.args.tau = self.args.tau * self.args.decay_coefficient
                # (10)
                with torch.no_grad():
                    for param_target, param_pred in zip(self.target_atc_model.parameters(), self.pred_atc_model.parameters()):
                        param_target.data.copy_(self.args.tau * param_pred.data + (1.0 - self.args.tau) * param_target.data)
                # (11)
                if t_env - self.t2 >= 100000:
                    self.t2 = t_env
                    with torch.no_grad():
                        for param_RND, param_ATC in zip(self.target_rnd_model.att.parameters(), self.target_atc_model.att.parameters()):
                            param_RND.data.copy_(self.args.tau_tgt * param_ATC.data + (1.0 - self.args.tau_tgt) * param_RND.data)
                # (12)
                with torch.no_grad():
                    for param_RND, param_ATC in zip(self.predictor_rnd_model.att.parameters(), self.pred_atc_model.att.parameters()):
                        param_RND.data.copy_(self.args.tau_pred * param_ATC.data + (1.0 - self.args.tau_pred) * param_RND.data)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # (13)
            if t_env < self.args.atc_pretrain:
                if self.args.model_type == 1:
                    self.logger.log_stat("loss_atc", loss_ATC.item(), t_env)
            if t_env > self.args.rnd_start_train:
                self.logger.log_stat("loss_rnd", loss_rnd.item(), t_env)
                self.logger.log_stat("loss", loss.item(), t_env)
                self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
                self.logger.log_stat("grad_norm", grad_norm, t_env)
                mask_elems = mask.sum().item()
                self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
                self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
                self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
                self.log_stats_t = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params, show_demo=show_demo, save_data=save_data)
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
