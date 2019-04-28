from kof97.vec_env import KOFEnvironmentDummy
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
use_meta = False
device = torch.device("cuda:0")
env = KOFEnvironmentDummy(device, frames_per_step=3, throttle=True)
policy = Policy(env.observation_space.shape, env.action_space, base=CNNSimpleBase)
policy.to(device)

# import pdb; pdb.set_trace()
policy = torch.load("./saved_models/kof_ppo/700.policy")

algorithm = PPO(policy, clip_param =0.2, ppo_epoch=4, num_mini_batch=num_mini_batch,\
    value_loss_coef=0.5, entropy_coef=0.01, lr=2.5e-4, max_grad_norm=0.5, eps=1e-5)
rollouts = RolloutStorage(num_steps, num_process, env.observation_space.shape, env.action_space,
                            policy.recurrent_hidden_state_size)

frames, infos = env.reset()
power = torch.FloatTensor(
    [[1.0 if x + 1 <= info["powerP1"] else 0.0 for x in range(3)] for info in infos]
)
position = torch.FloatTensor(
    [[info['positionP1'], info['positionP2']] for info in infos]
)


rollouts.obs[0].copy_(frames)
rollouts.power[0].copy_(power)
rollouts.position[0].copy_(position)
rollouts.to(device)
epoch = 0
episode = 0
episode_rewards = []
episode_stages = []
running_rewards = None


# logger = open("./logs/kof_ppo_2.log", 'w+')

while(True):

    print_epoch = False
    for step in range(num_steps):
        with torch.no_grad():
            value, action, action_log_probs, rnn_hxs = policy.step(rollouts.obs[step], rollouts.power[step], rollouts.position[step])
        
        frames, reward, done, infos = env.step(action)
        masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])

        bad_masks = torch.FloatTensor(
            [[0.0] for d in done])
        
        power = torch.FloatTensor(
            [[1.0 if x + 1 <= info["powerP1"] else 0.0 for x in range(3)] for info in infos]
        )
        position = torch.FloatTensor(
            [[info['positionP1'], info['positionP2']] for info in infos]
        )

        rollouts.insert(frames, rnn_hxs, action, action_log_probs, \
                    value, reward, masks, bad_masks, power, position)
                    
        for info in infos:
            if 'r' in info.keys():
                episode += 1
                episode_rewards.append(info['r'])
                episode_stages.append(info['stage'])
                if running_rewards != None:
                    running_rewards = info['r'] * 0.01 + running_rewards
                else:
                    running_rewards = info['r']
                print_epoch = True
                break
    
    with torch.no_grad():
        next_value = policy.get_value(
            rollouts.obs[-1], rollouts.power[-1], rollouts.position[-1], \
            rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]).detach()
    
    rollouts.compute_returns(next_value, False, 0.99,
                                 0.95, False)
    
    # value_loss_epoch, action_loss_epoch, dist_entropy_epoch = algorithm.update(rollouts)

    rollouts.after_update()

    if print_epoch:
        logs = "Epoch: {}, loss: {:.5f}, reward mean/min/max: {:.3f}/{:.3f}/{:.3f}, current/running: {:.3f}/{:.3f}, stage mean/min/max: {:.3f}/{:.3f}/{:.3f}, current: {:.3f}"\
            .format(epoch, 0, np.mean(episode_rewards), np.min(episode_rewards)\
            , np.max(episode_rewards), episode_rewards[-1], running_rewards, \
            np.mean(episode_stages), np.min(episode_stages), np.max(episode_stages), episode_stages[-1])
        print(logs)
        # logger.write(logs + '\n')
        # logger.flush()
    
    # if epoch % 100 == 0:
    #     torch.save(policy, "./saved_models/kof_ppo/" + str(epoch) + ".policy")
    #     torch.save(policy.base, "./saved_models/kof_ppo/" + str(epoch) + ".base")

    epoch = epoch + 1

