from kof97.vec_env import make_vec_envs, KOFEnvironmentDummy
from policy import Policy
from model import CNNSimpleBase
import torch
import random
from storage import RolloutStorage

num_steps = 16
num_process = 1
device = torch.device("cuda:0")

# env = make_vec_envs(num_process, device)
env = KOFEnvironmentDummy(device)
policy = Policy(env.observation_space.shape, env.action_space, base=CNNSimpleBase)
policy.to(device)

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
step = 0
epoch = 0
episode = 0
episode_rewards = []
episode_stages = []
running_rewards = None

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

        # import pdb; pdb.set_trace()
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
            rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
            rollouts.masks[-1]).detach()
    
    rollouts.compute_returns(next_value, False, 0.99,
                                 0.95, False)
    
    rollouts.after_update()

    if print_epoch:
        print("Episode {}, loss: {:.5f}, reward mean/min/max: {:.3f}/{:.3f}/{:.3f}, \
            current/running: {:.3f}/{:.3f}\n\t\t stage mean/min/max: {:.3f}/{:.3f}/{:.3f}\
            current: {:.3f}".format(\
            episode, 0, np.mean(episode_rewards), np.min(episode_rewards)\
            , np.max(episode_rewards), episode_rewards[-1], prunning_rewards, \
            np.mean(episode_stages), np.min(episode_stages), np.max(episode_stages), episode_stages[-1]))