from env import SFEnvironmentDummy
from policy import Policy
from model import CNNBase
import torch
import random
from storage import RolloutStorage
from pg import PG
import numpy as np

device = torch.device("cuda:0")
env = SFEnvironmentDummy(device)
policy = Policy(env.observation_space.shape, env.action_space, base=CNNSimpleBase)
policy.to(device)


algorithm = PG(policy, epoch=4, num_mini_batch=8)
num_steps = 1024
rollouts = RolloutStorage(num_steps, 1, env.observation_space.shape, env.action_space,
                            policy.recurrent_hidden_state_size)

frames = env.start()

rollouts.obs[0].copy_(frames)
rollouts.to(device)
histories = {}
epoch = 0
histories['reward'] = []

while(True):

    for step in range(num_steps):
        with torch.no_grad():
            value, action, action_log_probs, rnn_hxs = policy.step(rollouts.obs[step])
        
        frames, reward, done, info = env.step(action)
        
        masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] for d in done])
        rollouts.insert(frames, rnn_hxs, action,
                                action_log_probs, value, reward, masks, bad_masks)
        if 'r' in info.keys():
            histories['reward'].append(info['r'])
    
    with torch.no_grad():
        next_value = policy.get_value(
            rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
            rollouts.masks[-1]).detach()
    
    rollouts.compute_returns(next_value, False, 0.99,
                                 0.95, False)
    
    loss = algorithm.update(rollouts)

    rollouts.after_update()

    if len(histories['reward']) > 0:
        print("Epoch {}, loss: {:.3f}, episode reward {:.3f}, history mean: {:.3f}".format(
            epoch, loss, histories['reward'][-1], np.mean(histories['reward'])))
    
    epoch = epoch + 1

