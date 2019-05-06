from kof97.vec_env import make_vec_envs, KOFEnvironmentDummy
from model.policy import Policy
from model.base import CNNSimpleBase
from utils.storage import RolloutStorage
from utils.cmd_utils import cmd_arg_parser
from utils.utils import add_noise

import torch
import random
import numpy as np
import sys
import os

def main(args):
    parser = cmd_arg_parser()
    args = parser.parse_args(args)

    num_steps = args.num_steps
    num_process = args.num_process
    
    device = torch.device("cuda:0")

    save_path = './saved_models/' + args.exp_name + '/'
    log_dir = './logs/' + args.exp_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if num_process > 1:
        env = make_vec_envs(num_process, device, frames_per_step=args.frames_per_step, \
            monitor= args.monitor, render=args.render, stack_frame=args.stack_frame)
    else:
        env = KOFEnvironmentDummy(device, frames_per_step=args.frames_per_step, frame_ratio=args.frame_ratio, \
            monitor=args.monitor, throttle=args.throttle, stack_frame=args.stack_frame)
    policy = Policy(env.observation_space.shape, env.action_space, base=CNNSimpleBase)
    policy.to(device)

    rollouts = RolloutStorage(num_steps, num_process, env.observation_space.shape, env.action_space,
                                policy.recurrent_hidden_state_size)

    frames = env.reset()
    power = torch.FloatTensor(
        [[0.0 for x in range(3)] for _ in range(num_process)]
    )
    position = torch.FloatTensor(
        [[0.608, 0.928] for _ in range(num_process)]
    )

    rollouts.obs[0].copy_(frames)
    rollouts.power[0].copy_(power)
    rollouts.position[0].copy_(position)
    rollouts.to(device)
    epoch = 0
    episode = 0
    fps = 0
    episode_rewards = []
    episode_stages = []
    running_rewards = None
    force_explore = False
    # import pdb; pdb.set_trace()

    if args.restore:
        if args.restore_cnn:
            policy.base = torch.load(args.restore_path + str(args.restore) + '.base')
        else:
            policy = torch.load(args.restore_path + str(args.restore) + '.policy')
            # algorithm.optimizer = torch.load(args.restore_path + str(args.restore) + '.optim')
            epoch = args.restore

    logger = open(log_dir + "/kof_ppo.log", 'a+')

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
            if force_explore:
                frames, power, position = add_noise(frames, power, position, args.explore_scale)

            rollouts.insert(frames, rnn_hxs, action, action_log_probs, \
                            value, reward, masks, bad_masks, power, position)

            for info in infos:
                if 'game_done' in info.keys():
                    episode += 1
                    fps = info['fps']
                    print_epoch = True

                    episode_reward = 0
                    episode_stage = 0
                    for info in infos:
                        episode_reward += info['episode_reward']
                        episode_stage += info['stage']
                    
                    episode_rewards.append(episode_reward/num_process)
                    episode_stages.append(episode_stage/num_process)
                    
                    # if episode_stage/num_process < 1.0 and args.explore:
                    #     force_explore = True
                    # else:
                    #     force_explore = False

                    if running_rewards != None:
                        running_rewards = episode_reward/num_process * 0.01 + running_rewards*0.99
                    else:
                        running_rewards = episode_reward/num_process
                    break
        
        with torch.no_grad():
            next_value = policy.get_value(
                rollouts.obs[-1], rollouts.power[-1], rollouts.position[-1], \
                rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]).detach()
        
        rollouts.compute_returns(next_value, True, 0.99,
                                    0.95, False)
        
        rollouts.after_update()

        if print_epoch:
            logs = "Epoch: {}, fps: {:.3f}, loss: {:.5f}, reward mean/min/max: {:.3f}/{:.3f}/{:.3f}, current/running: {:.3f}/{:.3f}, stage mean/min/max: {:.3f}/{:.3f}/{:.3f}, current: {:.3f}"\
                .format(epoch, fps, 0, np.mean(episode_rewards), np.min(episode_rewards)\
                , np.max(episode_rewards), episode_rewards[-1], running_rewards, \
                np.mean(episode_stages), np.min(episode_stages), np.max(episode_stages), episode_stages[-1])
            print(logs)
            logger.write(logs + '\n')
            logger.flush()

        epoch = epoch + 1

if __name__ == "__main__":
    main(sys.argv[1:])