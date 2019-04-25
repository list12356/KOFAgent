from env import SFEnvironmentDummy
from policy import Policy
from model import CNNBase
import torch
import random
from storage import RolloutStorage
from ppo import PPO
import numpy as np

num_steps = 512
num_process = 1
num_mini_batch = 4
device = torch.device("cuda:0")
env = SFEnvironmentDummy(device)


frames = env.start()

epoch = 0
episode_rewards = []
episode = 1

while(True):

    print_episode = False
    for step in range(num_steps):
        move_action = random.randint(0, 8)
        attack_action = random.randint(0, 9)
        frames, reward, done, info = env.step(torch.Tensor([[move_action, attack_action]]))
        if 'r' in info.keys():
            print_episode = True
            episode_rewards.append(info['r'])
            episode += 1


    if print_episode:
        print("Epoch {}, loss: {:.3f}, reward mean/min/max: {:.3f}/{:.3f}/{:.3f}".format(
                episode, 0, np.mean(episode_rewards), np.min(episode_rewards), np.max(episode_rewards)))
    
    epoch = epoch + 1

