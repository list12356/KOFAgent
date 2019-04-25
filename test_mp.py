from env import make_vec_envs
from policy import Policy
from model import CNNSimpleBase
import torch
import random
from storage import RolloutStorage

num_steps = 16
num_process = 4
device = torch.device("cuda:0")

env = make_vec_envs(num_process, device)
policy = Policy(env.observation_space.shape, env.action_space, base=CNNSimpleBase)
policy.to(device)

rollouts = RolloutStorage(num_steps, num_process, env.observation_space.shape, env.action_space,
                            policy.recurrent_hidden_state_size)

frames = env.reset()

rollouts.obs[0].copy_(frames)
rollouts.to(device)
step = 0


while(True):

    for step in range(num_steps):
        with torch.no_grad():
            value, action, action_log_probs, rnn_hxs = policy.step(rollouts.obs[step])
        
        
        frames, reward, done, _ = env.step(action)
        
        masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] for d in done])
        rollouts.insert(frames, rnn_hxs, action,
                                action_log_probs, value, reward, masks, bad_masks)
    
    with torch.no_grad():
        next_value = policy.get_value(
            rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
            rollouts.masks[-1]).detach()
    
    rollouts.compute_returns(next_value, False, 0.99,
                                 0.95, False)
    
    rollouts.after_update()