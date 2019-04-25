from env import SFEnvironmentDummy
from policy import Policy
from model import CNNBase, CNNSimpleBase
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
policy = Policy(env.observation_space.shape, env.action_space, base=CNNSimpleBase)
policy.to(device)


algorithm = PPO(policy, clip_param =0.2, ppo_epoch=4, num_mini_batch=num_mini_batch,\
    value_loss_coef=0.5, entropy_coef=0.01, lr=2.5e-4, max_grad_norm=0.5, eps=1e-5)
rollouts = RolloutStorage(num_steps, num_process, env.observation_space.shape, env.action_space,
                            policy.recurrent_hidden_state_size)

frames = env.start()

rollouts.obs[0].copy_(frames)
rollouts.to(device)
epoch = 0
episode = 0
episode_rewards = []
running_rewards = None


while(True):

    print_epoch = False
    for step in range(num_steps):
        with torch.no_grad():
            value, action, action_log_probs, rnn_hxs = policy.step(rollouts.obs[step])
        
        frames, reward, done, info = env.step(action)
        
        # masks = torch.FloatTensor(
        #             [[0.0] if done_ else [1.0] for done_ in done])
        masks = torch.FloatTensor(
                    [[1.0]for d in done])
        bad_masks = torch.FloatTensor(
            [[0.0] for d in done])
        rollouts.insert(frames, rnn_hxs, action,
                                action_log_probs, value, reward, masks, bad_masks)
        if 'r' in info.keys():
            episode += 1
            episode_rewards.append(info['r'])
            if running_rewards != None:
                running_rewards = info['r'] * 0.01 + running_rewards
            else:
                running_rewards = info['r']
            print_epoch = True
    
    with torch.no_grad():
        next_value = policy.get_value(
            rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
            rollouts.masks[-1]).detach()
    
    rollouts.compute_returns(next_value, False, 0.99,
                                 0.95, False)
    
    value_loss_epoch, action_loss_epoch, dist_entropy_epoch = algorithm.update(rollouts)

    rollouts.after_update()

    if print_epoch:
        print("Episode {}, loss: {:.5f}, reward mean/min/max: {:.3f}/{:.3f}/{:.3f},  running: {:.3f}".format(
                episode, action_loss_epoch, np.mean(episode_rewards), np.min(episode_rewards), np.max(episode_rewards), running_rewards))
    
    epoch = epoch + 1

