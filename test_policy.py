from env import SFEnvironmentDummy
from policy import Policy
from policy import CNNBase
import torch
import random


device = torch.device("cuda:0")
env = SFEnvironmentDummy(device)
policy = Policy(env.observation_space.shape, env.action_space, base=CNNBase)
policy.to(device)

num_steps = 16
frames = env.start()


while(True):

    for step in range(num_steps):
        with torch.no_grad():
            value, action, action_log_probs, rnn_hxs = policy.step(frames)
        
        frames, reward, done, _ = env.step(action)
        masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
        policy.evaluate_actions(frames, rnn_hxs, masks, action)
