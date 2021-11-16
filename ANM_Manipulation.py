import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from anm_utils_39 import ANM39_Easy
from advertorch.attacks import GradientSignAttack
from stable_baselines3 import PPO, A2C
import copy
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
from MPC_Perfect import MPCAgentPerfect


time_steps=100000
clip_min=0.0,
clip_max=255.0
eps=0.05
num_states=18-1
num_actions=6
num_attacks=10
epsilon=1.0




# Parallel environments
#env = make_vec_env("CartPole-v1", n_envs=4)
device = torch.device("cpu")
envs = ANM39_Easy()
obs = envs.reset()
envs2 = ANM39_Easy()
obs = envs2.reset()

model = PPO("MlpPolicy", envs, verbose=1)
#model.learn(total_timesteps=time_steps)
model = PPO.load("39bus_result/ANM_39_A2C.zip")


agent = MPCAgentPerfect(envs.simulator, envs.action_space, envs.gamma,
                            safety_margin=0.96, planning_steps=10)



obs = envs.reset()
reward_RL_vec=np.zeros((200,1),dtype=float)
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = envs.step(action)
    print("Rewards RL Policy", rewards)
    print("Obs", obs)
    reward_RL_vec[i] = rewards
    # print("Observation", obs)

print("Total rewards", np.sum(reward_RL_vec))
print("Mean rewards", np.sum(reward_RL_vec)/200.0)
print("Variance", np.sqrt(np.var(reward_RL_vec)))



reward_MPC_vec=np.zeros((200,1),dtype=float)
for i in range(200):
        a = agent.act(envs2)
        obs, r, done, _ = envs2.step(a)
        #print("Observations", obs)
        print("Rewards MPC", r)
        reward_MPC_vec[i]=r

print("Total rewards", np.sum(reward_MPC_vec))
print("Mean rewards", np.sum(reward_MPC_vec)/200.0)
print("MPC Variance", np.sqrt(np.var(reward_MPC_vec)))

'''reward_adv_vec=np.zeros((200,1),dtype=float)
reward_adv_MPC=np.zeros((200,1),dtype=float)
obs = envs.reset()
for i in range(200):
    action0, _states = model.predict(obs)
    #print("Action 0", action0)
    obs_adv=np.copy(obs)
    state_diff=0.0
    attack_vec=np.zeros_like(obs)

    while state_diff<=epsilon:
        attk_diff = 0.0

        for j in range(num_states):
            obs_plus = np.copy(obs_adv)
            obs_minus = np.copy(obs_adv)
            obs_plus[j]+=eps
            obs_minus[j]-=eps
            action_plus, _states = model.predict(obs_plus)
            action_minus, _states = model.predict(obs_minus)
            #print("A1", action_plus)
            #print("A2", action_minus)
            current_diff_plus=np.linalg.norm(action_plus-action0, 2)
            current_diff_minus=np.linalg.norm(action_minus-action0, 2)
            current_diff=np.maximum(current_diff_minus, current_diff_plus)
            if current_diff>attk_diff:
                attk_diff=current_diff
                attk_index=j
                if current_diff_plus>current_diff_minus:
                    attack_vec[j]=1
                else:
                    attack_vec[j]=-1
        #print("Attack index", attk_index)
        obs_adv[attk_index]+=attack_vec[attk_index]*eps
        state_diff=np.linalg.norm(obs-obs_adv,2)
    #print("Final attack state", obs_adv)
    action_adv, _states = model.predict(obs_adv)
    #print("Original state", obs)
    #print("Perturbed state", obs_adv)
    print("Final adversarial actions", action_adv)

    print("Action original", action0)
    obs, rewards, dones, info = envs.step(action_adv)
    print("Rewards", rewards)
    reward_adv_vec[i]=rewards

    a = agent.act(envs2)
    obs, r, done, _ = envs2.step(a)
    #print("Observation", obs)
    print("Rewards MPC", r)
    reward_adv_MPC[i] = r

print("Total rewards", np.sum(reward_adv_vec))
print("Mean rewards", np.sum(reward_adv_vec)/200.0)
print("Variance", np.sqrt(np.var(reward_adv_vec)))

print("Total rewards MPC Attack", np.sum(reward_adv_MPC))
print("Mean rewards MPC Attack", np.sum(reward_adv_MPC)/200.0)
print("Variance MPC Attack", np.sqrt(np.var(reward_adv_MPC)))

    #a = agent.act(envs)
    #obs, r, done, _ = envs2.step(a)
    # print("Observations", obs)
    #print("Rewards MPC", r)
    #reward_MPC.append(r)



#Test on clean data, MPC perfect agent'''




##########Random Attack

reward_adv_vec=np.zeros((200,1),dtype=float)
reward_adv_MPC=np.zeros((200,1),dtype=float)
obs = envs.reset()
for i in range(200):
    action0, _states = model.predict(obs)
    #print("Action 0", action0)
    obs_adv=np.copy(obs)
    state_diff=0.0

    attack_vec=np.random.uniform(-0.03, 0.03,size=(np.shape(obs)))
    obs_adv+=attack_vec
    while np.linalg.norm(obs_adv-obs, 2)<=epsilon:
        #print("Here")
        obs_adv+=0.1*attack_vec
    #print("HEre2")

    #print("Final attack state", obs_adv)
    action_adv, _states = model.predict(obs_adv)
    #print("Original state", obs)
    #print("Perturbed state", obs_adv)
    print("Final adversarial actions", action_adv)

    print("Action original", action0)
    obs, rewards, dones, info = envs.step(action_adv)
    print("Rewards", rewards)
    reward_adv_vec[i]=rewards

    a = agent.act(envs2)
    obs, r, done, _ = envs2.step(a)
    #print("Observation", obs)
    print("Rewards MPC", r)
    reward_adv_MPC[i] = r

print("Total rewards", np.sum(reward_adv_vec))
print("Mean rewards", np.sum(reward_adv_vec)/200.0)
print("Variance", np.sqrt(np.var(reward_adv_vec)))

print("Total rewards MPC Attack", np.sum(reward_adv_MPC))
print("Mean rewards MPC Attack", np.sum(reward_adv_MPC)/200.0)
print("Variance MPC Attack", np.sqrt(np.var(reward_adv_MPC)))






