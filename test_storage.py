from env import SFEnvironmentDummy
from policy import Policy
from policy import CNNBase
import torch
import random
from storage import RolloutStorage

device = torch.device("cuda:0")
env = SFEnvironmentDummy(device)
policy = Policy(env.observation_space.shape, env.action_space, base=CNNBase)
policy.to(device)

num_steps = 16
rollouts = RolloutStorage(num_steps, 1, env.observation_space.shape, env.action_space,
                            policy.recurrent_hidden_state_size)

frames = env.start()

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