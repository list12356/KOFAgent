from distributions import Categorical, DiagGaussian, Bernoulli
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import init

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            num_outputs = action_space.nvec.tolist()
            self.dist = [Categorical(self.base.output_size, num_output) for num_output in num_outputs] 
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size


    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if isinstance(self.dist, list):
           [d.to(*args, **kwargs) for d in self.dist]

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def step(self, inputs, rnn_hxs=None, masks=None, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        # for multi discrete
        if isinstance(self.dist, list):
            dist = [d(actor_features) for d in self.dist]
            if deterministic:
                action = [d.mode() for d in dist]
            else:
                action = [d.sample() for d in dist]

            action_log_probs = torch.cat([dist[i].log_probs(action[i]) \
                for i in range(len(dist))], dim=1)
            action_log_probs = torch.sum(action_log_probs, dim=1, keepdim=True)
            action = torch.cat(action, dim=1)
            dist_entropy = torch.stack([d.entropy().mean() for d in dist]).mean()
        else:
            dist = self.dist(actor_features)
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

            action_log_probs = dist.log_probs(action)
            dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs
    
    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        
        if isinstance(self.dist, list):
            dist = [d(actor_features) for d in self.dist]
            action_log_probs = torch.cat([dist[i].log_probs(action[:, i]) \
                for i in range(len(dist))], dim=1)
            action_log_probs = torch.sum(action_log_probs, dim=1, keepdim=True)
            dist_entropy = torch.stack([d.entropy().mean() for d in dist]).mean()
        else:
            dist = self.dist(actor_features)
            action_log_probs = dist.log_probs(action)
            dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs