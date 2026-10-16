"""
Our work was supported by the following docs:
 - https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym
from collections import deque
import os
import pandas as pd
import matplotlib.pyplot as plt
from test_script import *
from utilities import select_action, optimize_model, ReplayMemory
from test_script import QNetwork
from itertools import count

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
def train_model(strategy, device=device):
    env = gym.make('CartPole-v1')
    np.random.seed(2048) #Used in order for all strategies to be trained on same pseudo-random lengths
    #training hyperparameters
    num_episodes = 500
    batch_size = 128
    gamma = 0.99
    tau = 0.005
    lr = 3e-4
    epsilons= {
    "fixed": {"epsilon" : 0.8},
    "decay_per_episode": {"epsilon" : 0.8, 
                       "epsilon_low" : 0.05,
                       "decay_episodes": 250}
    }
    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    policy_net = QNetwork(n_observations, n_actions).to(device)
    target_net = QNetwork(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    memory = ReplayMemory(10000)

    for i_episode in range(num_episodes):
        pole_length = np.random.uniform(0.4, 1.8)
        env.unwrapped.length = pole_length

        #set epsilon based on 2 epsilon-greedy strategies
        #baseline 
        if strategy == "fixed":
            epsilon = epsilons[strategy]["epsilon"]
        #epsilon which decays with each step of episode
        elif strategy == "decay_per_episode" and i_episode == 0:
            epsilon = epsilons[strategy]["epsilon"]
            epsilon_low = epsilons[strategy]["epsilon_low"]
            epsilon_decay = (epsilon - epsilon_low)/epsilons[strategy]["decay_episodes"] #reach the low epsilon around specified episode

        # Initialize the environment and get its state
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        if i_episode != 0 and strategy == "decay_per_episode":
            if epsilon >= epsilon_low:
                epsilon = max(epsilon_low, epsilon-epsilon_decay)
        for t in count():
            #action gap method
            if strategy == "action_gap":
                with torch.no_grad():
                    qs = policy_net(state)
                    sorted_q, _ = torch.sort(qs, descending=True)
                    best_q = sorted_q[0][0]
                    second_best_q = sorted_q[0][1]
                    action_gap = (best_q - second_best_q).item()

                    limit = 0.1
                    
                    if action_gap < limit:
                        action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
                    else:
                        action = qs.max(1)[1].view(1, 1)
            else:
                action = select_action(state, epsilon, policy_net, env, device)

            observation, reward, terminated, _, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated # ignore truncation during training

            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(device, memory, policy_net, target_net, optimizer, gamma, batch_size)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
            target_net.load_state_dict(target_net_state_dict)
            
            if done:
                break
    #save model after training
    torch.save(policy_net.state_dict(), f'weights/{strategy}.pth')
    print('Complete')

if __name__ == "__main__":
    train_model("fixed", device)
    train_model("decay_per_episode", device)
    train_model("action_gap", device)