import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PG():
    def __init__(self,
                 actor_critic,
                 epoch=1,
                 num_mini_batch=1,
                 lr=7e-4,
                 eps=1e-5,
                 max_grad_norm=None):

        self.actor_critic = actor_critic
        self.epoch = epoch
        self.num_mini_batch = num_mini_batch

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        for e in range(self.epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

            loss_out = None
            for sample in data_generator:
                frames_batch, rhx, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(frames_batch, rhx, masks_batch, actions_batch)

                # Calculate loss
                loss = -torch.sum(torch.mul(action_log_probs, return_batch))
                
                # Update network weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_out = loss.item()

        return loss_out
