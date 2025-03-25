import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math


def build_net(layer_shape, activation, output_activation):
    '''Build net with for loop'''
    layers = []
    for j in range(len(layer_shape) - 1):
        act = activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


class Actor_BS(nn.Module):
    def __init__(self, state_dim, action_dim_BS, hid_shape, h_acti=nn.ReLU, o_acti=nn.ReLU):
        super(Actor_BS, self).__init__()

        layers = [2*state_dim] + list(hid_shape)
        self.a_net = build_net(layers, h_acti, o_acti)
        self.loc_layer_BS = nn.Linear(3, state_dim)
        self.other_layer_BS = nn.Linear(state_dim - 3, state_dim)

        self.mu_layer = nn.Linear(hid_shape[-1], action_dim_BS)
        self.log_std_layer = nn.Linear(hid_shape[-1], action_dim_BS)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, deterministic=False, with_logprob=True):
        '''Network with Enforcing Action Bounds'''
        loc_BS = F.leaky_relu(self.loc_layer_BS(state[:, :3]))
        other_BS = (self.other_layer_BS(state[:, 3:]))
        net_out = torch.cat([loc_BS, other_BS], dim=1)
        net_out = self.a_net(net_out)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  # 总感觉这里clamp不利于学习
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        if deterministic:
            u = mu
        else:
            u = dist.rsample()  # '''reparameterization trick of Gaussian'''#
        a = torch.tanh(u)

        if with_logprob:
            # get probability density of logp_pi_a from probability density of u, which is given by the original paper.
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)

            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(dim=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(dim=1, keepdim=True)
        else:
            logp_pi_a = None

        return a, logp_pi_a

class BS_Critic(nn.Module):
    def __init__(self, state_dim, action_dim_BS, hid_shape):
        super(BS_Critic, self).__init__()
        layers = [2*state_dim+action_dim_BS] + list(hid_shape) + [1]

        self.loc_layer_BS = nn.Linear(3, state_dim)
        self.other_layer_BS = nn.Linear(state_dim - 3, state_dim)
        # self.BS_action = nn.Linear(action_dim_BS, 2*action_dim_BS)
        self.Q_BS = build_net(layers, nn.ReLU, nn.Identity)

        self.loc_layer_BS2 = nn.Linear(3, state_dim)
        self.other_layer_BS2 = nn.Linear(state_dim - 3, state_dim)
        # self.BS_action2 = nn.Linear(action_dim_BS, 2*action_dim_BS)
        self.Q_BS2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, action):
        loc_BS = F.leaky_relu(self.loc_layer_BS(state[:, :3]))
        other_BS = (self.other_layer_BS(state[:, 3:]))
        # BS_action = F.leaky_relu(self.BS_action(action[:, -self.action_dim_BS:]))
        input_BS = torch.cat([loc_BS, other_BS, action], 1)
        q1 = self.Q_BS(input_BS)

        loc_BS2 = F.leaky_relu(self.loc_layer_BS2(state[:, :3]))
        other_BS2 = (self.other_layer_BS2(state[:, 3:]))
        # BS_action2 = F.leaky_relu(self.BS_action2(action[:, -self.action_dim_BS:]))
        input_BS2 = torch.cat([loc_BS2, other_BS2, action], 1)
        q2 = self.Q_BS2(input_BS2)
        return q1, q2

class Actor_ARISUAV(nn.Module):
    def __init__(self, state_dim, action_dim_ARISUAV, hid_shape, h_acti=nn.ReLU, o_acti=nn.ReLU):
        super(Actor_ARISUAV, self).__init__()

        layers = [2*state_dim] + list(hid_shape)
        # detach_dim = int(hid_shape[0] / 2 / state_dim)
        self.a_net = build_net(layers, h_acti, o_acti)
        self.loc_layer_ARISUAV = nn.Linear(3, state_dim)
        self.other_layer_ARISUAV = nn.Linear(state_dim - 3, state_dim)

        self.mu_layer = nn.Linear(hid_shape[-1], action_dim_ARISUAV)
        self.log_std_layer = nn.Linear(hid_shape[-1], action_dim_ARISUAV)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, deterministic=False, with_logprob=True):
        '''Network with Enforcing Action Bounds'''
        loc_ARISUAV = F.leaky_relu(self.loc_layer_ARISUAV(state[:, :3]))
        other_ARISUAV = (self.other_layer_ARISUAV(state[:, 3:]))
        net_out = torch.cat([loc_ARISUAV, other_ARISUAV], dim=1)
        net_out = self.a_net(net_out)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  # 总感觉这里clamp不利于学习
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        if deterministic:
            u = mu
        else:
            u = dist.rsample()  # '''reparameterization trick of Gaussian'''#
        a = torch.tanh(u)

        if with_logprob:
            # get probability density of logp_pi_a from probability density of u, which is given by the original paper.
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)

            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(dim=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(dim=1, keepdim=True)
        else:
            logp_pi_a = None
        return a, logp_pi_a


class ARISUAV_Critic(nn.Module):
    def __init__(self, state_dim, action_dim_ARISUAV, hid_shape):
        super(ARISUAV_Critic, self).__init__()
        # self.action_dim_ARISUAV = action_dim_ARISUAV
        # self.action1 = nn.Linear(action_dim_ARISUAV, action_dim_ARISUAV)
        layers = [2 * state_dim + action_dim_ARISUAV] + list(hid_shape) + [1]
        self.loc_layer_ARISUAV = nn.Linear(3, state_dim)
        self.other_layer_ARISUAV = nn.Linear(state_dim - 3, state_dim)
        self.Q_ARISUAV = build_net(layers, nn.ReLU, nn.Identity)

        # self.action2 = nn.Linear(action_dim_ARISUAV, action_dim_ARISUAV)
        self.loc_layer_ARISUAV2 = nn.Linear(3, state_dim)
        self.other_layer_ARISUAV2 = nn.Linear(state_dim - 3, state_dim)
        self.Q_ARISUAV2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, action):
        # action = whiten(action)
        loc_ARISUAV = F.leaky_relu(self.loc_layer_ARISUAV(state[:, :3]))
        other_ARISUAV = (self.other_layer_ARISUAV(state[:, 3:]))
        # action1 = (self.action1(action))
        input_ARISUAV = torch.cat([loc_ARISUAV, other_ARISUAV, action], 1)
        q1 = self.Q_ARISUAV(input_ARISUAV)

        loc_ARISUAV2 = F.leaky_relu(self.loc_layer_ARISUAV2(state[:, :3]))
        other_ARISUAV2 = (self.other_layer_ARISUAV2(state[:, 3:]))
        # action2 = (self.action2(action))
        input_ARISUAV2 = torch.cat([loc_ARISUAV2, other_ARISUAV2, action], 1)
        q2 = self.Q_ARISUAV2(input_ARISUAV2)
        return q1, q2

class QMIX(nn.Module):
    def __init__(self, state_dim):
        super(QMIX, self).__init__()
        self.hyper_w11 = nn.Linear(state_dim, 2*state_dim)
        self.hyper_w12 = nn.Linear(2*state_dim, 4*state_dim)
        self.hyper_w13 = nn.Linear(4*state_dim, 8*state_dim)
        self.hyper_w14 = nn.Linear(8*state_dim, 2)
        self.hyper_b11 = nn.Linear(state_dim, 2*state_dim)
        self.hyper_b12 = nn.Linear(2*state_dim, 4*state_dim)
        self.hyper_b13 = nn.Linear(4*state_dim, 8*state_dim)
        self.hyper_b14 = nn.Linear(8*state_dim, 1)

        # self.hyper_w21 = nn.Linear(state_dim, 2*state_dim)
        # self.hyper_w22 = nn.Linear(2*state_dim, 4*state_dim)
        # self.hyper_w23 = nn.Linear(4*state_dim, 8*state_dim)
        # self.hyper_w24 = nn.Linear(8 * state_dim, 2)
        # self.hyper_b21 = nn.Linear(state_dim, 2*state_dim)
        # self.hyper_b22 = nn.Linear(2*state_dim, 4*state_dim)
        # self.hyper_b23 = nn.Linear(4*state_dim, 8*state_dim)
        # self.hyper_b24 = nn.Linear(8 * state_dim, 1)

    def forward(self, state, q_list, H=torch.zeros(130, 2)):
        H = H.to(q_list.device)
        q_list = q_list - H
        w = F.relu(self.hyper_w11(state))
        w = F.relu(self.hyper_w12(w))
        w = F.relu(self.hyper_w13(w))
        w = torch.abs(self.hyper_w14(w))
        b = F.relu(self.hyper_b11(state))
        b = F.relu(self.hyper_b12(b))
        b = F.relu(self.hyper_b13(b))
        b = (self.hyper_b14(b))
        q_total1 = ((q_list*w).sum(dim=1)).unsqueeze(1) + b
        # q_total1 = torch.mm(q_list, w.transpose(0, 1)) + b
        # print("1",q_total1)
        # q_total1 = q_total1.mean(dim=1, keepdim=True)
        # print("2", q_total1)
        # print("ql",q_list)
        # print("w",w)
        # print("sdf", ((q_list*w).sum(dim=1)).unsqueeze(1))
        # print("b",b)
        # print("q_total1",q_total1)

        # w2 = F.relu(self.hyper_w21(state))
        # w2 = F.relu(self.hyper_w22(w2))
        # w2 = F.relu(self.hyper_w23(w2))
        # w2 = torch.abs(self.hyper_w24(w2))
        # b2 = F.relu(self.hyper_b21(state))
        # b2 = F.relu(self.hyper_b22(b2))
        # b2 = F.relu(self.hyper_b23(b2))
        # b2 = (self.hyper_b24(b2))
        # q_total2 = ((q_list*w2).sum(dim=1)).unsqueeze(1) + b2
        # q_total2 = torch.mm(q_list,w2.transpose(0,1)) + b2
        # q_total2 = q_total2.mean(dim=1, keepdim=True)
        # print("q_total2", q_total2)
        # print("tt",torch.min(q_total1, q_total2))
        return q_total1#torch.min(q_total1, q_total2)


class SAC_Agent(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            action_dim_ARISUAV,
            tau,
            gamma=0.99,
            hid_shape=(256, 256),
            a_lr=3e-4,
            c_lr=3e-4,
            alpha_lr=3e-4,
            batch_size=256,
            alpha=0.2,
            adaptive_alpha=True,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        self.device = device

        self.actor_BS = Actor_BS(state_dim, action_dim-action_dim_ARISUAV, hid_shape).to(device)
        self.actor_optimizer_BS = torch.optim.Adam(self.actor_BS.parameters(), lr=a_lr)

        self.BS_critic = BS_Critic(state_dim, action_dim-action_dim_ARISUAV,  hid_shape).to(device)
        # self.BS_critic_optimizer = torch.optim.Adam(self.BS_critic.parameters(), lr=c_lr)
        self.BS_critic_target = copy.deepcopy(self.BS_critic)

        self.actor_ARISUAV = Actor_ARISUAV(state_dim, action_dim_ARISUAV,  hid_shape).to(device)
        self.actor_optimizer_ARISUAV = torch.optim.Adam(self.actor_ARISUAV.parameters(), lr=a_lr)

        self.ARISUAV_critic = ARISUAV_Critic(state_dim, action_dim_ARISUAV,  hid_shape).to(device)
        # self.ARISUAV_critic_optimizer = torch.optim.Adam(self.ARISUAV_critic.parameters(), lr=c_lr)
        self.ARISUAV_critic_target = copy.deepcopy(self.ARISUAV_critic)

        self.QMIX = QMIX(state_dim).to(device)
        # self.QMIX_optimizer = torch.optim.Adam(self.QMIX.parameters(), lr=c_lr)
        self.QMIX_target = copy.deepcopy(self.QMIX)

        self.q_parameters = list(self.QMIX.parameters()) + list(self.BS_critic.parameters()) + list(self.ARISUAV_critic.parameters())
        self.q_parameters_optimizer = torch.optim.Adam(self.q_parameters, lr=c_lr)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.BS_critic_target.parameters():
            p.requires_grad = False
        for p in self.ARISUAV_critic_target.parameters():
            p.requires_grad = False

        self.action_dim = action_dim
        self.action_dim_ARISUAV = action_dim_ARISUAV
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.alpha_ARISUAV = alpha
        self.alpha_BS = alpha
        self.adaptive_alpha = adaptive_alpha
        if adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy_ARISUAV = torch.tensor(-action_dim_ARISUAV, dtype=float, requires_grad=True, device=device)
            # We learn log_alpha instead of alpha to ensure exp(log_alpha)=alpha>0
            self.log_alpha_ARISUAV = torch.tensor(np.log(alpha), dtype=float, requires_grad=True, device=device)
            self.alpha_optim_ARISUAV = torch.optim.Adam([self.log_alpha_ARISUAV], lr=alpha_lr)

            self.target_entropy_BS = torch.tensor(action_dim_ARISUAV-action_dim, dtype=float, requires_grad=True, device=device)
            # We learn log_alpha instead of alpha to ensure exp(log_alpha)=alpha>0
            self.log_alpha_BS = torch.tensor(np.log(alpha), dtype=float, requires_grad=True, device=device)
            self.alpha_optim_BS = torch.optim.Adam([self.log_alpha_BS], lr=alpha_lr)

    def select_action(self, state, deterministic=False, with_logprob=False):
        # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            a2, _ = self.actor_BS(state, deterministic, with_logprob)
            a1, _ = self.actor_ARISUAV(state, deterministic, with_logprob)
            a = self.cat_tensor(a1, a2)
        return a.cpu().numpy().flatten()

    def cat_tensor(self, item1, item2):
        return torch.cat((item1, item2), dim=1)

    def train(self, replay_buffer):
        s, a, r, s_prime, dead_mask = replay_buffer.sample(self.batch_size)

        #----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_prime1, log_pi_a_prime1 = self.actor_ARISUAV(s_prime)
            a_prime2, log_pi_a_prime2 = self.actor_BS(s_prime)
            q1_a, q2_a = self.ARISUAV_critic_target(s_prime, a_prime1)
            target_Q1_ARISUAV = torch.min(q1_a, q2_a)
            q1_b, q2_b = self.BS_critic_target(s_prime, a_prime2)
            target_Q1_BS = torch.min(q1_b, q2_b)
            q_list = self.cat_tensor(target_Q1_ARISUAV, target_Q1_BS)
            H = self.cat_tensor(self.alpha_ARISUAV * log_pi_a_prime1, self.alpha_BS * log_pi_a_prime2)
            target_Q = self.QMIX_target(s_prime, q_list, H)
            target_Q = r + (1 - dead_mask) * self.gamma * target_Q
            # print("t",target_Q)

        # Get current Q estimates
        q1_ca, q2_ca = self.ARISUAV_critic(s, a[:, :self.action_dim_ARISUAV])
        current_Q2_ARISUAV = torch.min(q1_ca, q2_ca)
        q1_cb, q2_cb = self.BS_critic(s, a[:, self.action_dim_ARISUAV:])
        current_Q2_BS = torch.min(q1_cb, q2_cb)
        current_Q = self.QMIX(s, self.cat_tensor(current_Q2_ARISUAV, current_Q2_BS))
        loss_QMIX = F.mse_loss(current_Q, target_Q)
        self.q_parameters_optimizer.zero_grad()
        loss_QMIX.backward()
        self.q_parameters_optimizer.step()

        #----------------------------- ↓↓↓↓↓ Update ARISUAV Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for params in self.ARISUAV_critic.parameters():
            params.requires_grad = False

        a1, log_pi_a1 = self.actor_ARISUAV(s)
        current_Q1, current_Q2 = self.ARISUAV_critic(s, a1)
        Q = torch.min(current_Q1, current_Q2)
        Q_list_com1 = self.cat_tensor(Q, torch.zeros(self.batch_size, 1).to(Q.device))
        Q = self.QMIX(s, Q_list_com1)
        a_loss1 = (self.alpha_ARISUAV * log_pi_a1 - Q).mean()
        self.actor_optimizer_ARISUAV.zero_grad()
        a_loss1.backward()
        self.actor_optimizer_ARISUAV.step()

        for params in self.ARISUAV_critic.parameters():
            params.requires_grad = True

        #----------------------------- ↓↓↓↓↓ Update BS Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for params in self.BS_critic.parameters():
            params.requires_grad = False

        a2, log_pi_a2 = self.actor_BS(s)
        current_Q1, current_Q2 = self.BS_critic(s, a2)
        Q = torch.min(current_Q1, current_Q2)
        Q_list_com = self.cat_tensor(torch.zeros(self.batch_size, 1).to(Q.device), Q)
        Q = self.QMIX(s, Q_list_com)
        a_loss2 = (self.alpha_BS * log_pi_a2 - Q).mean()
        self.actor_optimizer_BS.zero_grad()
        a_loss2.backward()
        self.actor_optimizer_BS.step()

        for params in self.BS_critic.parameters():
            params.requires_grad = True
        #----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha:
            # we optimize log_alpha instead of aplha, which is aimed to force alpha = exp(log_alpha)> 0
            # if we optimize aplpha directly, alpha might be < 0, which will lead to minimun entropy.

            alpha_loss1 = -(self.log_alpha_ARISUAV * (log_pi_a1 + self.target_entropy_ARISUAV).detach()).mean()
            self.alpha_optim_ARISUAV.zero_grad()
            alpha_loss1.backward()
            self.alpha_optim_ARISUAV.step()
            self.alpha_ARISUAV = self.log_alpha_ARISUAV.exp()

            alpha_loss2 = -(self.log_alpha_BS * (log_pi_a2 + self.target_entropy_BS).detach()).mean()
            self.alpha_optim_BS.zero_grad()
            alpha_loss2.backward()
            self.alpha_optim_BS.step()
            self.alpha_BS = self.log_alpha_BS.exp()

        #----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.ARISUAV_critic.parameters(), self.ARISUAV_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.BS_critic.parameters(), self.BS_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.QMIX.parameters(), self.QMIX_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    # def save(self, episode):
    #     torch.save(self.actor_ARISUAV.state_dict(), "./model/sac_actor{}.pth".format(episode))
    #     torch.save(self.actor_BS.state_dict(), "./model/sac_actor{}.pth".format(episode))
    #     torch.save(self.q_critic.state_dict(), "./model/sac_q_critic{}.pth".format(episode))
    #
    # def load(self, episode):
    #     self.actor_ARISUAV.load_state_dict(torch.load("./model/sac_actor{}.pth".format(episode)))
    #     self.actor_BS.load_state_dict(torch.load("./model/sac_actor{}.pth".format(episode)))
    #     self.q_critic.load_state_dict(torch.load("./model/sac_q_critic{}.pth".format(episode)))