# python3.11
# -*- coding: utf-8 -*-
# @Time: 2023 12 04 19:10
# @Author:Jie


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from avoid_missile.task import AvoidMissileTask
from avoid_missile.environment import NoTacviewSimEnv3D, SimEnv3D
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
GAMMA = 0.99


# ------------------------------------ #
# Actor
# ------------------------------------ #

class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.tanh(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_actions]
        x = F.softmax(x, dim=1)  # [b,n_actions]-->[b,n_actions]
        return x


# ------------------------------------ #
# Critic
# ------------------------------------ #

class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens_1, n_hiddens_2):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens_1)
        self.fc2 = nn.Linear(n_hiddens_1, n_hiddens_2)
        self.out = nn.Linear(n_hiddens_2, 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):  # [b,n_states]-->[b,1]
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        actions_value = self.out(x)
        return actions_value


# ------------------------------------ #
# Actor-Critic
# ------------------------------------ #

class ActorCritic:
    def __init__(self, n_states, n_hiddens_1, n_hiddens_2, n_actions,
                 actor_lr, critic_lr, gamma):
        self.gamma = gamma

        self.actor = PolicyNet(n_states, n_hiddens_1, n_actions)
        self.critic = ValueNet(n_states, n_hiddens_1, n_hiddens_2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        # next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        # dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        td_value = self.critic(states)
        td_target = rewards
        td_delta = td_target - td_value

        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(F.mse_loss(td_value, td_target))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def cal_q(self, state: list) -> Tensor:
        with torch.no_grad():
            actions_value = self.critic(torch.FloatTensor(state))
        return actions_value


class CalReward:
    Lambda = 0.8

    def __init__(self) -> None:
        self.s_a_r_next_s: list = []
        self.transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
        }

    def reset(self) -> None:
        self.s_a_r_next_s = []
        self.transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
        }

    def store_s_a_r_next_s(self, s_a_r_next_s: list) -> None:
        self.s_a_r_next_s.append(s_a_r_next_s)

    def update_store_reward(self, a2c: ActorCritic, end_reward: float) -> dict:
        G = end_reward
        min_distance_index = len(self.s_a_r_next_s)
        for i in range(1, min_distance_index, 1):
            G = GAMMA * G
            self.s_a_r_next_s[min_distance_index - i][2] = self.Lambda ** (i - 1) * G
            for j in range(1, i, 1):
                self.s_a_r_next_s[min_distance_index - i][2] = self.s_a_r_next_s[min_distance_index - i][2] + \
                                                               (1 - self.Lambda) * self.Lambda ** (
                                                                       j - 1) * GAMMA ** j * a2c.cal_q(
                    self.s_a_r_next_s[min_distance_index - i + j][0])
            self.transition_dict['states'].append(self.s_a_r_next_s[min_distance_index - i][0])
            self.transition_dict['actions'].append(self.s_a_r_next_s[min_distance_index - i][1])
            self.transition_dict['next_states'].append(self.s_a_r_next_s[min_distance_index - i][3])
            self.transition_dict['rewards'].append(self.s_a_r_next_s[min_distance_index - i][2])
            self.transition_dict['dones'].append(0)
        return self.transition_dict


if __name__ == '__main__':

    # ----------------------------------------- #
    # base args set
    # ----------------------------------------- #
    action_list = [1, 2, 3, 4, 5, 12, 13]
    cal_reward = CalReward()
    env = NoTacviewSimEnv3D(AvoidMissileTask, agent_interaction_freq=1)
    # ----------------------------------------- #
    # model args
    # ----------------------------------------- #
    episode = 1500
    gamma = 0.99
    actor_lr = 1e-3
    critic_lr = 1e-3
    n_hiddens_1 = 100
    n_hiddens_2 = 50
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    state_dim = 12
    action_dim = 7

    # ----------------------------------------- #
    # model build
    # ----------------------------------------- #

    a2c = ActorCritic(n_states=state_dim,
                      n_hiddens_1=n_hiddens_1,
                      n_hiddens_2=n_hiddens_2,
                      n_actions=action_dim,
                      actor_lr=actor_lr,
                      critic_lr=critic_lr,
                      gamma=gamma)

    # ----------------------------------------- #
    # model training
    # ----------------------------------------- #
    for i in range(episode):
        print('<<<<<<<<Episode: %s' % i)
        state = env.reset()[0]
        cal_reward.reset()
        while True:
            # action choose and env step
            action_number = a2c.take_action(state)
            next_state, reward, done, info = env.step(action_list[action_number])

            # store experience
            cal_reward.store_s_a_r_next_s([state, action_number, reward, next_state])
            # next_state
            state = next_state

            # episode end and cal lambda return
            if done:
                if info:
                    writer.add_scalar('Reward', reward, i)
                transition_dict = cal_reward.update_store_reward(a2c, reward)
                print('episode%s---reward_sum: %s' % (i, round(reward, 2)))
                break

        # training
        a2c.update(transition_dict)
        # breakpoint saving
        if i % 2000 == 0:
            # save net args
            torch.save({
                'actor_state_dict': a2c.actor.state_dict(),
                'critic_state_dict': a2c.critic.state_dict(),
                'actor_optimizer_state_dict': a2c.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': a2c.critic_optimizer.state_dict()
            }, "./6-checkpoint/2-a2c_checkpoint/ckpt_{}.pth".format(i))
    env.close()
    torch.save(a2c, './4-net_param/2-a2c_net_params_{}.pth'.format(episode))
