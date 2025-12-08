# Qtran_rnd

from modules.mixers.RND_ATC import ATCPred, ATCTarget
import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_qatten import DMAQ_QattenMixer
from modules.mixers.RND_net import RNDNetwork
import torch.nn.functional as F
import torch as th
import torch
from torch.optim import RMSprop
import numpy as np
import os
import random

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
    # torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

class DMAQ_qatten_rnd_Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.params = list(mac.parameters())
        self.last_entropy_log_t = 0
        self.t2=0
        self.last_target_update_episode = 0
        print("33333333333", args.seed)
        set_seed(args.seed)
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "dmaq_qatten":
                self.mixer = DMAQ_QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # 定义atc网络和优化器
        self.target_atc_model = ATCTarget(self.args.n_agents, self.args.obs_shape, self.args.RND_hidden_dim, self.args.RND_hidden_dim).cuda()
        self.pred_atc_model = ATCPred(self.args.n_agents, self.args.obs_shape, self.args.RND_hidden_dim, self.args.RND_hidden_dim).cuda()
        for param in self.target_atc_model.parameters():
            param.requires_grad = False
        self.optimiser_atc = th.optim.Adam(self.pred_atc_model.parameters(), lr=self.args.atc_lr)  # 59:2e-5

        self.predictor_rnd_model = RNDNetwork(self.args.n_agents, self.args.obs_shape, self.args.RND_hidden_dim, self.args.model_type).cuda()
        self.target_rnd_model = RNDNetwork(self.args.n_agents, self.args.obs_shape, self.args.RND_hidden_dim, self.args.model_type).cuda()

        # 设置RND目标网络的参数冻结
        for param in self.target_rnd_model.parameters():
            param.requires_grad = False

        # 定义
        self.optimiser_rnd = th.optim.Adam(self.predictor_rnd_model.parameters(), lr=float(4e-5))
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.n_actions = self.args.n_actions

    def generate_negative_samples(self, anchor_data, anchor_adj, neg_samples_num):
        batch_size, time_step, n_agent, data_dim = anchor_data.shape
        batch_size, time_step, n_agent, adj_data_dim = anchor_adj.shape
        # 初始化负样本张量，形状为 (neg_samples_num, batch_size, n_agent, embedding_dim)
        negative_samples = torch.zeros((neg_samples_num, batch_size, time_step, n_agent, data_dim), dtype=anchor_data.dtype, device=anchor_data.device)
        neg_adj = torch.zeros((neg_samples_num, batch_size, time_step, n_agent, adj_data_dim), dtype=anchor_adj.dtype, device=anchor_adj.device)
        for idx in range(batch_size):
            # 生成负样本的随机索引，排除当前锚点样本的索引
            neg_indices = np.random.choice([i for i in range(batch_size) if i != idx], size=neg_samples_num, replace=False)
            for neg_sample_idx in range(neg_samples_num):
                # 将选中的负样本复制到负样本张量中
                negative_samples[neg_sample_idx, idx] = anchor_data[neg_indices[neg_sample_idx]]
                neg_adj[neg_sample_idx, idx] = anchor_adj[neg_indices[neg_sample_idx]]
        return negative_samples, neg_adj

    def atc_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings, temperature=0.9):
        batch_size = anchor_embeddings.shape[0]
        n_agents = anchor_embeddings.shape[1]
        embedding_dim = anchor_embeddings.shape[2]
        neg_nums = negative_embeddings.shape[0]
        positive_scores = (anchor_embeddings * positive_embeddings).sum(dim=2) / temperature  # (32*100, 5)
        negative_scores = torch.zeros(neg_nums, batch_size, n_agents).cuda()
        for j in range(neg_nums):
            negative_scores[j] = (anchor_embeddings * negative_embeddings[j]).sum(dim=2) / temperature  # 使用 element-wise 乘法
        all_scores = torch.cat([positive_scores.unsqueeze(0), negative_scores], dim=0)
        # 计算 softmax 概率
        probs = F.softmax(all_scores, dim=0)  # (neg_num + 1, batch_size, n_agents)
        # 提取正样本的概率
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
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]  # reward (32,x,1)
        # actions (32,x,5,1)
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()  # (32,x,1)
        mask = batch["filled"][:, :-1].float()  # (32,x,1)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]  # (32,x+1,5,11)
        actions_onehot = batch["actions_onehot"][:, :-1]  # (32,x,5,11)
        obs_t = batch["obs"][:, :-1]  # (32,x,5,80)
        adj_t = batch["adj"][:, :-1]  # (32,x,5,5)
        obs = batch["obs"][:, 1:]  # (32,x,5,80)
        adj = batch["adj"][:, 1:]  # (32,x,5,5)

        # （0）准备相关数据
        neg_samples_num = 3
        negative_obs, negative_adj = self.generate_negative_samples(obs, adj, neg_samples_num)
        negative_obs = negative_obs.reshape(neg_samples_num, -1, self.args.n_agents, self.args.obs_shape)  # (32x,5,80)
        negative_adj = negative_adj.reshape(neg_samples_num, -1, self.args.n_agents, self.args.n_agents)  # (32x,5,5)

        obs_t = obs_t.reshape(-1, self.args.n_agents, self.args.obs_shape)  # (32x,5,80)
        adj_t = adj_t.reshape(-1, self.args.n_agents, self.args.n_agents)  # (32x,5,5)
        obs = obs.reshape(-1, self.args.n_agents, self.args.obs_shape)  # (32x,5,80)
        adj = adj.reshape(-1, self.args.n_agents, self.args.n_agents)  # (32x,5,5)

        # （0）准备相关数据
        anchor_obs = obs
        anchor_adj = adj
        # （0.1）数据变换
        positive_obs = obs_t
        positive_adj = adj_t

        if t_env > self.args.rnd_start_train:

            # （1）输入经过RND_target得到输出【原】
            # RND目标网络 -> 得到目标网络特征
            target_feature, _ = self.target_rnd_model(obs.cuda(), adj.cuda())  # (32x,5,64)

            # （2）输入经过RND_pred得到输出【原】
            # RND预测网络 -> 得到预测网络特征
            predictor_feature, _ = self.predictor_rnd_model(obs.cuda(), adj.cuda())
            # RND预测网络 -> 得到下一时刻的特征
            predictor_feature_t, _ = self.predictor_rnd_model(obs_t.cuda(), adj_t.cuda())  # (32x,5,64)

            # (3) 计算loss_RND【原】
            # MSE损失
            forward_loss = ((target_feature - predictor_feature) ** 2).mean(-1)

            # RND的损失表达式
            if self.args.model_type == 0:
                kl_loss = th.log((predictor_feature_t + 0.0001) / (predictor_feature + 0.0001)).mean(-1)
                loss_rnd = (forward_loss.sum() + self.args.kl_lambda * kl_loss.sum()) / mask.sum()

            if self.args.model_type == 1:
                loss_rnd = forward_loss.sum() / mask.sum()
            ############

            ##############
            if t_env - self.last_entropy_log_t >= 50000:
                self.last_entropy_log_t = t_env

                try:
                    # 1. 临时切换到评估模式
                    self.predictor_rnd_model.eval()

                    # 2. 在不计算梯度的情况下进行
                    with th.no_grad():
                        # 3. 直接使用当前训练批次的 obs 和 adj
                        #    确保它们的形状是 (batch_size, n_agents, ...)
                        #    根据您的代码, obs 和 adj 此时已经是正确的形状

                        # 4. 前向传播并获取注意力权重
                        #    确保您的 RNDNetwork 和 AttModel 支持 return_attention=True
                        _, attention_weights = self.predictor_rnd_model(obs.cuda(), adj.cuda(), return_attention=True)
                        _, attention_weights2 = self.predictor_rnd_model(obs_t.cuda(), adj_t.cuda(), return_attention=True)

                        # 5. 计算平均熵
                        epsilon = 1e-8
                        p_log_p = attention_weights * th.log(attention_weights + epsilon)
                        p_log_p2 = attention_weights2 * th.log(attention_weights2 + epsilon)
                        entropy_per_agent = -th.sum(p_log_p, dim=-1)
                        entropy_per_agent2 = -th.sum(p_log_p2, dim=-1)
                        mean_entropy = th.mean(entropy_per_agent).item()
                        mean_entropy2 = th.mean(entropy_per_agent2).item()
                        # 6. 记录到日志
                        self.logger.log_stat("attention_entropy", mean_entropy, t_env)
                        self.logger.log_stat("attention_entropy2", mean_entropy2, t_env)

                except Exception as e:
                    print(f"Error calculating attention entropy at step {t_env}: {e}")
                finally:
                    # 7. 确保模型最终切回训练模式，无论是否出错
                    self.predictor_rnd_model.train()

            # 内在奖励
            intrinsic_rewards = (self.args.intrinsic_scale * forward_loss).sum(-1)  # (32x)
            intrinsic_rewards = intrinsic_rewards.reshape(rewards.shape[0], rewards.shape[1], rewards.shape[2])  # (32,x,1)
            total_rewards = intrinsic_rewards.clone()  # (32,x,1)
            total_rewards += rewards.cuda()  # it (32,x,1)

            # Calculate estimated Q-Values
            mac_out = []
            mac.init_hidden(batch.batch_size)  # ???
            for t in range(batch.max_seq_length):  # max_seq_length=72
                agent_outs = mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # (32,83,5,11) # Concat over time

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
            # (32,82,5) # Remove the last dim

            x_mac_out = mac_out.clone().detach()  # (32,83,5,11)
            x_mac_out[avail_actions == 0] = -9999999
            max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)  # (32,83,5) (32,83,5)

            max_action_index = max_action_index.detach().unsqueeze(3)  # (32,83,5,1)
            is_max_action = (max_action_index == actions).int().float()  # (32,83,5,1)

            if show_demo:
                q_i_data = chosen_action_qvals.detach().cuda().numpy()
                q_data = (max_action_qvals - chosen_action_qvals).detach().cuda().numpy()

            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)  # (32,5,11)
                target_mac_out.append(target_agent_outs)
            # target_mac_out=(83)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

            # Mask out unavailable actions
            target_mac_out[avail_actions[:, 1:] == 0] = -9999999

            # Max over target Q-Values
            if self.args.double_q:  # ???
                # Get actions that maximise live Q (for double q-learning)
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
                target_max_qvals = target_mac_out.max(dim=3)[0]
                target_next_actions = cur_max_actions.detach()

                cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)).cuda()
                cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
            else:
                # Calculate the Q-Values necessary for the target
                target_mac_out = []
                self.target_mac.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    target_agent_outs = self.target_mac.forward(batch, t=t)
                    target_mac_out.append(target_agent_outs)
                # We don't need the first timesteps Q-Value estimate for calculating targets
                target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
                target_max_qvals = target_mac_out.max(dim=3)[0]

            # Mix
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

            # Calculate 1-step Q-Learning targets
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

            # Td-error
            td_error = (chosen_action_qvals - targets.detach())  # (32,82,1)

            mask = mask.expand_as(td_error)  # (32,82,1)

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask
            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
            masked_hit_prob = th.mean(is_max_action, dim=2) * mask
            hit_prob = masked_hit_prob.sum() / mask.sum()

            # RL Optimise
            optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
            optimiser.step()

            # (4) 冻结RND_pred中的注意力模块
            if self.args.model_type == 1:
                for param in self.predictor_rnd_model.att.parameters():
                    param.requires_grad = False

            # (5) 反向传播更新RND_pred参数【原】
            self.optimiser_rnd.zero_grad()
            loss_rnd.backward()
            self.optimiser_rnd.step()

        if t_env < self.args.atc_pretrain:
            # (6) 正样本、负样本经过ATC_target得到embeddings
            # 输入到RND的数据（32, 100, 5, 80）
            # ATC-锚点数据（32, 100, 5, 80）-> (32*100, 5, 64)
            # atc-正样本（32, 100，5, 80）-> (32*100, 5, 64)
            # atc-负样本（31, 32, 100, 5, 80）-> (31, 32*100, 5, 64)
            if self.args.model_type == 1:
                positive_embeddings = self.target_atc_model(positive_obs.cuda(), positive_adj.cuda())
                a, b, c = positive_embeddings.shape
                negative_embeddings = torch.empty((0, a, b, c), dtype=torch.float, device=positive_embeddings.device)
                for i in range(neg_samples_num):
                    single_neg_obs = negative_obs[i].cuda()  # 取出负样本数据块
                    single_neg_adj = negative_adj[i].cuda()  # 取出对应的邻接矩阵数据块
                    # 将负样本块输入 target_atc_model
                    single_neg_embeddings = self.target_atc_model(single_neg_obs, single_neg_adj)
                    # 将输出添加到 negative_embeddings
                    negative_embeddings = torch.cat((negative_embeddings, single_neg_embeddings.unsqueeze(0)), dim=0)

                # (7) 锚点样本经过ATC_pred得到输出
                anchor_embeddings = self.pred_atc_model(anchor_obs.cuda(), anchor_adj.cuda())

                # (8) 计算loss_ATC
                loss_ATC = self.atc_loss(anchor_embeddings, positive_embeddings, negative_embeddings) / mask.sum()

                # (9) 更新ATC_pred
                self.optimiser_atc.zero_grad()
                loss_ATC.backward()
                self.optimiser_atc.step()

                if t_env > self.args.atc_decay:
                    self.args.tau = self.args.tau * self.args.decay_coefficient
                #     self.args.tau_tgt = self.args.tau_tgt * self.args.decay_coefficient
                #     self.args.tau_pred = self.args.tau_tgt * self.args.decay_coefficient
                # if self.args.tau < 0.000001:
                #     self.args.tau = 0
                #     self.args.tau_tgt = 0
                #     self.args.tau_pred = 0

                # (10) 软更新ATC_target
                # self.args.tau = 0.001
                # 把多少参数给ATC_target
                with torch.no_grad():
                    for param_target, param_pred in zip(self.target_atc_model.parameters(), self.pred_atc_model.parameters()):
                        param_target.data.copy_(self.args.tau * param_pred.data + (1.0 - self.args.tau) * param_target.data)

                # (11) 将ATC_target中注意力模块（attC）的参数赋值给RND_target中的注意力模块（attA）
                # self.args.tau_tgt = 0.01
                # 把多少ATC_target参数给RND_target
                if t_env - self.t2 >= 100000:
                    self.t2 = t_env

                    with torch.no_grad():
                        for param_RND, param_ATC in zip(self.target_rnd_model.att.parameters(), self.target_atc_model.att.parameters()):
                            param_RND.data.copy_(self.args.tau_tgt * param_ATC.data + (1.0 - self.args.tau_tgt) * param_RND.data)

                # (12) 将ATC_pred中注意力模块（attD）的参数赋值给RND_pred中的注意力模块（attB）
                # self.args.tau_pred = 0.8
                # 把多少ATC_pred参数给RND_pred
                with torch.no_grad():
                    for param_RND, param_ATC in zip(self.predictor_rnd_model.att.parameters(), self.pred_atc_model.att.parameters()):
                        param_RND.data.copy_(self.args.tau_pred * param_ATC.data + (1.0 - self.args.tau_pred) * param_RND.data)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # (13) 把loss加上
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
