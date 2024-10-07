# python3.6
# -*- coding: utf-8 -*-
# @Time: 2023 10 24 16:53
# @Author:Jie


import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from avoid_missile.task import AvoidMissileTask
from avoid_missile.environment import NoTacviewSimEnv3D, SimEnv3D
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

BATCH_SIZE = 32
LR = 0.00001
EPSILON = 0.1  # greedy policy
GAMMA = 0.99  # reward discount
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 10000
N_ACTIONS = 7
N_STATES = 12


class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc2 = nn.Linear(50, 50, bias=False)
        self.out = nn.Linear(50, N_ACTIONS)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions_value = self.out(x)
        return actions_value


class DQN(object):

    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.record_loss = []

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        if self.memory_counter == MEMORY_CAPACITY:
            np.savetxt('./5-observe_data/{}.csv'.format(str(self.memory_counter)), self.memory, delimiter=',')

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        # q_next = self.target_net(b_s_).detach()
        q_target = b_r
        loss = self.loss_func(q_eval, q_target)

        self.record_loss.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def cal_q(self, state: list):
        mean_q_value = self.target_net(torch.FloatTensor(state)).detach().mean(0)
        return mean_q_value


class CalReward:
    Lambda = 0.8

    def __init__(self) -> None:
        self.s_a_r_next_s: list = []

    def reset(self) -> None:
        self.s_a_r_next_s = []

    def store_s_a_r_next_s(self, s_a_r_next_s: list) -> None:
        self.s_a_r_next_s.append(s_a_r_next_s)

    def update_store_reward(self, dqn: DQN, end_reward: float) -> None:
        G = end_reward
        min_distance_index = len(self.s_a_r_next_s)
        for i in range(1, min_distance_index, 1):
            G = GAMMA * G
            self.s_a_r_next_s[min_distance_index - i][2] = self.Lambda ** (i-1) * G
            for j in range(1, i, 1):
                self.s_a_r_next_s[min_distance_index - i][2] = self.s_a_r_next_s[min_distance_index - i][2] + \
                                                               (1 - self.Lambda) * self.Lambda ** (j - 1) * GAMMA ** j * dqn.cal_q(self.s_a_r_next_s[min_distance_index - i + j][0])
            dqn.store_transition(self.s_a_r_next_s[min_distance_index - i][0], self.s_a_r_next_s[min_distance_index - i][1],
                                 self.s_a_r_next_s[min_distance_index - i][2], self.s_a_r_next_s[min_distance_index - i][3])


if __name__ == "__main__":
    # try:
    #     dqn = torch.load('./4-net_param/dqn_net_params_1000.pth')
    # except:
    #     dqn = DQN()
    #     print('Not file be found!')
    dqn = DQN()
    cal_reward = CalReward()
    action_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    episode = 10000
    env = NoTacviewSimEnv3D(AvoidMissileTask, agent_interaction_freq=1)
    for i in range(episode):  # 400 episodes
        print('<<<<<<<<<Episode: %s' % i)
        state = env.reset()[0]
        cal_reward.reset()
        while True:
            # action choose and env step
            # action_number = np.random.randint(7)
            action_number = dqn.choose_action(state)
            next_state, reward, done, info = env.step(action_list[action_number])
            cal_reward.store_s_a_r_next_s([state, action_number, reward, next_state])  # TODO: improve position 1
            # dqn.store_transition(state, action_number, reward, next_state)
            state = next_state

            # training
            if dqn.memory_counter >= MEMORY_CAPACITY:
                if dqn.memory_counter == MEMORY_CAPACITY:
                    print("start training!")
                    time.sleep(5)
                    EPSILON = 0.7
                dqn.learn()

            # episode end and state store
            if done:
                if info:
                    writer.add_scalar('Reward', reward, i)
                cal_reward.update_store_reward(dqn, reward)
                print('episode%s---reward_sum: %s' % (i, round(reward, 2)))
                break

        # breakpoint saving
        if i % 2000 == 0:
            torch.save(dqn, "./6-checkpoint/ckpt_{}.pth".format(i))
    env.close()
    torch.save(dqn, './4-net_param/dqn_net_params_{}.pth'.format(episode))
