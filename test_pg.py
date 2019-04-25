
from envs_ref import make_vec_envs
from policy import Policy
from model import CNNBase, CNNSimpleBase, CNNAtariBase
import torch
import random
from storage import RolloutStorage
from pg import PG
import numpy as np
import os

num_process = 8
num_steps = 128
num_mini_batch = 4
device = torch.device('cuda:0')
env = make_vec_envs('PongNoFrameskip-v4', 1, num_process,
                        0.95, './tmp/atari', device, False)

policy = Policy(env.observation_space.shape, env.action_space, base=CNNAtariBase, base_kwargs={'input_channel': 4})
policy.to(device)


algorithm = PG(policy, epoch=4, num_mini_batch=num_mini_batch)
rollouts = RolloutStorage(num_steps, num_process, env.observation_space.shape, env.action_space,
                            policy.recurrent_hidden_state_size)

frames = env.reset()

rollouts.obs[0].copy_(frames)
rollouts.to(device)
histories = {}
epoch = 0
episode_rewards = []

while(True):

    for step in range(num_steps):
        with torch.no_grad():
            value, action, action_log_probs, rnn_hxs = policy.step(rollouts.obs[step])
        
        frames, reward, done, infos = env.step(action)
        
        masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] for d in done])
        rollouts.insert(frames, rnn_hxs, action,
                                action_log_probs, value, reward, masks, bad_masks)
        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
    
    with torch.no_grad():
        next_value = policy.get_value(
            rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
            rollouts.masks[-1]).detach()
    
    rollouts.compute_returns(next_value, False, 0.99,
                                 0.95, False)
    
    loss = algorithm.update(rollouts)

    rollouts.after_update()

    if len(episode_rewards) > 0:
        print("Epoch {}, loss: {:.3f}, reward mean/min/max: {:.3f}/{:.3f}/{:.3f}".format(
            epoch, loss, np.mean(episode_rewards), np.min(episode_rewards), np.max(episode_rewards)))
    
    epoch = epoch + 1





