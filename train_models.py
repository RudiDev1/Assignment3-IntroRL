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

def train_model(strategy, device):
    env = gym.make('CartPole-v1')

    #training hyperparameters
    num_episodes = 50
    batch_size = 128
    gamma = 0.99
    tau = 0.005
    lr = 3e-4

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

        #set epsilon based on 3 independent methods
        if strategy == "baseline":
            #baseline
            epsilon = 0.5
        elif strategy == "VDBE-Boltzmann":
            #VDBE-Boltzmann
            epsilon = 0.5
        elif strategy == "third":
            #pole-length based
            epsilon = 0.5

        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = select_action(state, epsilon, policy_net, env, device)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

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
                #what do we do when it is done?
                break
    #save model after training
    torch.save(policy_net.state_dict(), f'weights/{strategy}.pth')
    print('Complete')

if __name__ == "__main__":
    train_model("baseline", device)