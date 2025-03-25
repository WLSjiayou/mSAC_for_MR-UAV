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

class Actor_UAV(nn.Module):
    def __init__(self, state_dim_UAV, action_dim_UAV, hid_shape, h_acti=nn.ReLU, o_acti=nn.ReLU):
        super(Actor_UAV, self).__init__()
        layers = [state_dim_UAV] + list(hid_shape)
        self.a_net = build_net(layers, h_acti, o_acti)

        self.mu_layer = nn.Linear(hid_shape[-1], action_dim_UAV)
        self.log_std_layer = nn.Linear(hid_shape[-1], action_dim_UAV)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, deterministic=False, with_logprob=True):
        '''Network with Enforcing Action Bounds'''
        net_out = self.a_net(state)
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

class UAV_Critic(nn.Module):
    def __init__(self, state_dim_UAV, action_dim_UAV, hid_shape):
        super(UAV_Critic, self).__init__()
        layers = [state_dim_UAV+action_dim_UAV] + list(hid_shape) + [1]
        self.Q_BS = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_BS2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, action):
        input_BS = torch.cat([state, action], 1)
        q1 = self.Q_BS(input_BS)

        input_BS2 = torch.cat([state, action], 1)
        q2 = self.Q_BS2(input_BS2)
        return q1, q2

class Actor_BS(nn.Module):
    def __init__(self, state_dim_BS, action_dim_BS, hid_shape, h_acti=nn.ReLU, o_acti=nn.ReLU):
        super(Actor_BS, self).__init__()

        layers = [state_dim_BS] + list(hid_shape)
        self.a_net = build_net(layers, h_acti, o_acti)

        self.mu_layer = nn.Linear(hid_shape[-1], action_dim_BS)
        self.log_std_layer = nn.Linear(hid_shape[-1], action_dim_BS)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, deterministic=False, with_logprob=True):
        '''Network with Enforcing Action Bounds'''

        net_out = self.a_net(state)
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
    def __init__(self, state_dim_BS, action_dim_BS, hid_shape):
        super(BS_Critic, self).__init__()
        layers = [state_dim_BS+action_dim_BS] + list(hid_shape) + [1]
        self.Q_BS = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_BS2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, action):
        input_BS = torch.cat([state, action], 1)
        q1 = self.Q_BS(input_BS)
        input_BS2 = torch.cat([state, action], 1)
        q2 = self.Q_BS2(input_BS2)
        return q1, q2

class Actor_RIS(nn.Module):
    def __init__(self, state_dim_RIS, action_dim_RIS, hid_shape, h_acti=nn.ReLU, o_acti=nn.ReLU):
        super(Actor_RIS, self).__init__()

        layers = [state_dim_RIS] + list(hid_shape)
        self.a_net = build_net(layers, h_acti, o_acti)

        self.mu_layer = nn.Linear(hid_shape[-1], action_dim_RIS)
        self.log_std_layer = nn.Linear(hid_shape[-1], action_dim_RIS)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, deterministic=False, with_logprob=True):
        '''Network with Enforcing Action Bounds'''
        net_out = self.a_net(state)
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


class RIS_Critic(nn.Module):
    def __init__(self, state_dim_RIS, action_dim_RIS, hid_shape):
        super(RIS_Critic, self).__init__()
        layers = [state_dim_RIS + action_dim_RIS] + list(hid_shape) + [1]
        self.Q_RIS = build_net(layers, nn.ReLU, nn.Identity)
        # self.action2 = nn.Linear(action_dim_RIS, action_dim_RIS)
        self.Q_RIS2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, action):
        input_RIS = torch.cat([state, action], 1)
        q1 = self.Q_RIS(input_RIS)

        input_RIS2 = torch.cat([state, action], 1)
        q2 = self.Q_RIS2(input_RIS2)
        return q1, q2

class QMIX(nn.Module):
    def __init__(self, state_dim):
        super(QMIX, self).__init__()
        self.hyper_w11 = nn.Linear(state_dim, 2*state_dim)
        self.hyper_w12 = nn.Linear(2*state_dim, 4*state_dim)
        self.hyper_w13 = nn.Linear(4*state_dim, 8*state_dim)
        self.hyper_w14 = nn.Linear(8*state_dim, 3)
        self.hyper_b11 = nn.Linear(state_dim, 2*state_dim)
        self.hyper_b12 = nn.Linear(2*state_dim, 4*state_dim)
        self.hyper_b13 = nn.Linear(4*state_dim, 8*state_dim)
        self.hyper_b14 = nn.Linear(8*state_dim, 1)

    def forward(self, state, q_list, H=torch.zeros(130, 3)):
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

        return q_total1


class SAC_Agent(object):
    def __init__(
            self,
            state_dim,
            # state_dim_RIS,
            # state_dim_BS,
            action_dim_UAV,
            action_dim,
            action_dim_RIS,
            action_dim_BS,
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
        self.actor_UAV = Actor_UAV(state_dim, action_dim_UAV, hid_shape).to(device)
        self.actor_optimizer_UAV = torch.optim.Adam(self.actor_UAV.parameters(), lr=a_lr)

        self.UAV_critic = UAV_Critic(state_dim, action_dim_UAV,  hid_shape).to(device)
        self.UAV_critic_target = copy.deepcopy(self.UAV_critic)
        
        self.actor_BS = Actor_BS(state_dim, action_dim_BS, hid_shape).to(device)
        self.actor_optimizer_BS = torch.optim.Adam(self.actor_BS.parameters(), lr=a_lr)

        self.BS_critic = BS_Critic(state_dim, action_dim_BS,  hid_shape).to(device)
        # self.BS_critic_optimizer = torch.optim.Adam(self.BS_critic.parameters(), lr=c_lr)
        self.BS_critic_target = copy.deepcopy(self.BS_critic)

        self.actor_RIS = Actor_RIS(state_dim, action_dim_RIS,  hid_shape).to(device)
        self.actor_optimizer_RIS = torch.optim.Adam(self.actor_RIS.parameters(), lr=a_lr)

        self.RIS_critic = RIS_Critic(state_dim, action_dim_RIS,  hid_shape).to(device)
        # self.RIS_critic_optimizer = torch.optim.Adam(self.RIS_critic.parameters(), lr=c_lr)
        self.RIS_critic_target = copy.deepcopy(self.RIS_critic)

        self.QMIX = QMIX(state_dim).to(device)
        self.QMIX_target = copy.deepcopy(self.QMIX)

        self.q_parameters = list(self.QMIX.parameters()) + list(self.BS_critic.parameters()) + list(self.RIS_critic.parameters()) + list(self.UAV_critic.parameters())
        self.q_parameters_optimizer = torch.optim.Adam(self.q_parameters, lr=c_lr)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.BS_critic_target.parameters():
            p.requires_grad = False
        for p in self.RIS_critic_target.parameters():
            p.requires_grad = False
        for p in self.UAV_critic.parameters():
            p.requires_grad = False

        self.state_dim = state_dim
        # self.state_dim_RIS = state_dim_RIS
        # # print("s",self.state_dim_RIS)
        # self.state_dim_BS = state_dim_BS
        self.action_dim = action_dim
        self.action_dim_UAV = action_dim_UAV
        self.action_dim_RIS = action_dim_RIS
        self.action_dim_BS = action_dim_BS
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.alpha_UAV = alpha
        self.alpha_RIS = alpha
        self.alpha_BS = alpha
        self.adaptive_alpha = adaptive_alpha
        if adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy_RIS = torch.tensor(-action_dim_RIS, dtype=float, requires_grad=True, device=device)
            # We learn log_alpha instead of alpha to ensure exp(log_alpha)=alpha>0
            self.log_alpha_RIS = torch.tensor(np.log(alpha), dtype=float, requires_grad=True, device=device)
            self.alpha_optim_RIS = torch.optim.Adam([self.log_alpha_RIS], lr=alpha_lr)

            self.target_entropy_BS = torch.tensor(-action_dim_BS, dtype=float, requires_grad=True, device=device)
            # We learn log_alpha instead of alpha to ensure exp(log_alpha)=alpha>0
            self.log_alpha_BS = torch.tensor(np.log(alpha), dtype=float, requires_grad=True, device=device)
            self.alpha_optim_BS = torch.optim.Adam([self.log_alpha_BS], lr=alpha_lr)

            self.target_entropy_UAV = torch.tensor(-action_dim_UAV, dtype=float, requires_grad=True, device=device)
            # We learn log_alpha instead of alpha to ensure exp(log_alpha)=alpha>0
            self.log_alpha_UAV = torch.tensor(np.log(alpha), dtype=float, requires_grad=True, device=device)
            self.alpha_optim_UAV = torch.optim.Adam([self.log_alpha_UAV], lr=alpha_lr)

    def select_action(self, state, deterministic=False, with_logprob=False):
        # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            a3, _ = self.actor_BS(state, deterministic, with_logprob)
            a2, _ = self.actor_RIS(state, deterministic, with_logprob)
            a1, _ = self.actor_UAV(state, deterministic, with_logprob)
            a = self.cat_tensor(a1, a2, a3)
        return a.cpu().numpy().flatten()

    def cat_tensor(self, item1, item2, item3):
        return torch.cat((item1, item2, item3), dim=1)

    def train(self, replay_buffer):
        s, a, r, s_prime, dead_mask = replay_buffer.sample(self.batch_size)
        a_UAV = a[:, :self.action_dim_UAV]
        a_RIS = a[:, self.action_dim_UAV:self.action_dim_UAV+self.action_dim_RIS]
        a_BS = a[:, -self.action_dim_BS:]
        #----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_prime1, log_pi_a_prime1 = self.actor_UAV(s_prime)
            a_prime2, log_pi_a_prime2 = self.actor_RIS(s_prime)
            a_prime3, log_pi_a_prime3 = self.actor_BS(s_prime)
            q1_u, q2_u = self.UAV_critic_target(s_prime, a_prime1)
            target_Q1_UAV = torch.min(q1_u, q2_u)
            q1_r, q2_r = self.RIS_critic_target(s_prime, a_prime2)
            target_Q1_RIS = torch.min(q1_r, q2_r)
            q1_b, q2_b = self.BS_critic_target(s_prime, a_prime3)
            target_Q1_BS = torch.min(q1_b, q2_b)
            q_list = self.cat_tensor(target_Q1_UAV, target_Q1_RIS, target_Q1_BS)
            H = self.cat_tensor(self.alpha_UAV*log_pi_a_prime1, self.alpha_RIS * log_pi_a_prime2, self.alpha_BS * log_pi_a_prime3)
            target_Q = self.QMIX_target(s_prime, q_list, H)
            target_Q = r + (1 - dead_mask) * self.gamma * target_Q
            # print("t",target_Q)

        # Get current Q estimates
        q1_cu, q2_cu = self.UAV_critic(s, a_UAV)
        current_Q2_UAV = torch.min(q1_cu, q2_cu)
        q1_cr, q2_cr = self.RIS_critic(s, a_RIS)
        current_Q2_RIS = torch.min(q1_cr, q2_cr)
        q1_cb, q2_cb = self.BS_critic(s, a_BS)
        current_Q2_BS = torch.min(q1_cb, q2_cb)
        current_Q = self.QMIX(s, self.cat_tensor(current_Q2_UAV, current_Q2_RIS, current_Q2_BS))
        loss_QMIX = F.mse_loss(current_Q, target_Q)
        self.q_parameters_optimizer.zero_grad()
        loss_QMIX.backward()
        self.q_parameters_optimizer.step()
        #----------------------------- ↓↓↓↓↓ Update UAV Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for params in self.UAV_critic.parameters():
            params.requires_grad = False

        a1, log_pi_a1 = self.actor_UAV(s)
        current_Q1, current_Q2 = self.UAV_critic(s, a1)
        Q = torch.min(current_Q1, current_Q2)
        Q_list_com1 = self.cat_tensor(Q, torch.zeros(self.batch_size, 1).to(Q.device), torch.zeros(self.batch_size, 1).to(Q.device))
        Q = self.QMIX(s, Q_list_com1)
        a_loss1 = (self.alpha_UAV * log_pi_a1 - Q).mean()
        self.actor_optimizer_UAV.zero_grad()
        a_loss1.backward()
        self.actor_optimizer_UAV.step()

        for params in self.UAV_critic.parameters():
            params.requires_grad = True

        #----------------------------- ↓↓↓↓↓ Update RIS Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for params in self.RIS_critic.parameters():
            params.requires_grad = False

        a2, log_pi_a2 = self.actor_RIS(s)
        current_Q1, current_Q2 = self.RIS_critic(s, a2)
        Q = torch.min(current_Q1, current_Q2)
        Q_list_com1 = self.cat_tensor(torch.zeros(self.batch_size, 1).to(Q.device), Q, torch.zeros(self.batch_size, 1).to(Q.device))
        Q = self.QMIX(s, Q_list_com1)
        a_loss1 = (self.alpha_RIS * log_pi_a2 - Q).mean()
        self.actor_optimizer_RIS.zero_grad()
        a_loss1.backward()
        self.actor_optimizer_RIS.step()

        for params in self.RIS_critic.parameters():
            params.requires_grad = True

        #----------------------------- ↓↓↓↓↓ Update BS Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for params in self.BS_critic.parameters():
            params.requires_grad = False

        a3, log_pi_a3 = self.actor_BS(s)
        current_Q1, current_Q2 = self.BS_critic(s, a3)
        Q = torch.min(current_Q1, current_Q2)
        Q_list_com = self.cat_tensor(torch.zeros(self.batch_size, 1).to(Q.device), torch.zeros(self.batch_size, 1).to(Q.device), Q)
        Q = self.QMIX(s, Q_list_com)
        a_loss2 = (self.alpha_BS * log_pi_a3 - Q).mean()
        self.actor_optimizer_BS.zero_grad()
        a_loss2.backward()
        self.actor_optimizer_BS.step()

        for params in self.BS_critic.parameters():
            params.requires_grad = True
        #----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha:
            # we optimize log_alpha instead of aplha, which is aimed to force alpha = exp(log_alpha)> 0
            # if we optimize aplpha directly, alpha might be < 0, which will lead to minimun entropy.
            alpha_loss1 = -(self.log_alpha_UAV * (log_pi_a1 + self.target_entropy_UAV).detach()).mean()
            self.alpha_optim_UAV.zero_grad()
            alpha_loss1.backward()
            self.alpha_optim_UAV.step()
            self.alpha_UAV = self.log_alpha_UAV.exp()

            alpha_loss2 = -(self.log_alpha_RIS * (log_pi_a2 + self.target_entropy_RIS).detach()).mean()
            self.alpha_optim_RIS.zero_grad()
            alpha_loss2.backward()
            self.alpha_optim_RIS.step()
            self.alpha_RIS = self.log_alpha_RIS.exp()

            alpha_loss3 = -(self.log_alpha_BS * (log_pi_a3 + self.target_entropy_BS).detach()).mean()
            self.alpha_optim_BS.zero_grad()
            alpha_loss3.backward()
            self.alpha_optim_BS.step()
            self.alpha_BS = self.log_alpha_BS.exp()

        #----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.UAV_critic.parameters(), self.UAV_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.RIS_critic.parameters(), self.RIS_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.BS_critic.parameters(), self.BS_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.QMIX.parameters(), self.QMIX_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
