
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env import ShmemVecEnv
from utils.utils import resize_image
from kof97.env import Environment
from utils.utils import transform_frames

import torch
import numpy as np
import time


class KOFEnvironmentDummy():
    def __init__(self, device, frames_per_step=3, frame_ratio=3, throttle=False, monitor=False, stack_frame=False):
        roms_path = "./roms/"  # Replace this with the path to your ROMs
        self.venv = Environment("env1", roms_path, frame_ratio=frame_ratio, frames_per_step=frames_per_step, difficulty=3, throttle=throttle)
        self.observation_space = Box(0, 255, shape=[3, 190, 300]) # use stack here
        self.action_space = Discrete(38)
        self.device = device
        self.episode_reward = 0
        self.monitor = monitor
        self.monitor_file = None
        if self.monitor:
            self.monitor_file = open('./tmp/kof_env/env1.log', 'a+')
        self.stage = 0
        self.fraps = 0
        self.elapsed_time = 0
        self.stack_frame = stack_frame

    def reset(self):
        self.episode_reward = 0
        frames, info = self.venv.reset()
        return transform_frames(frames, self.device, self.stack_frame)

    def step(self, actions):

        start = time.time()
        frames, reward, round_done, stage_done, game_done, info = self.venv.step(int(actions))
        end = time.time()

        frames = transform_frames(frames, self.device, self.stack_frame)
        rewards = torch.FloatTensor([reward['P1']]).unsqueeze(dim=0) # for later parallel, the current num_process = 1
        self.fraps += info['fraps']
        self.elapsed_time += end - start
        self.episode_reward += float(reward['P1'])
        info['episode_reward'] = self.episode_reward
        if self.monitor:
            self.monitor_file.write(str(reward['P1']) + '\t' + str(int(actions)) + '\n')

        if stage_done:
            self.stage += 1
        if game_done:
            # import pdb; pdb.set_trace()
            info['game_done'] = True
            info['stage'] = self.stage
            info['fps'] = self.fraps / self.elapsed_time
            self.elapsed_time = 0
            self.fraps =0
            self.episode_reward = 0
            self.stage = 0
            if self.monitor:
                self.monitor_file.flush()
            self.reset()

        return frames, rewards, [round_done or game_done or stage_done], [info]

    def close(self):
        self.venv.close()

class KOFEnvironment():
    def __init__(self, env_id, device, frames_per_step=3, frame_ratio=3, monitor=False, render=True):
        roms_path = "./roms/"  # Replace this with the path to your ROMs
        self.venv = Environment(env_id, roms_path, frame_ratio=frame_ratio, frames_per_step=frames_per_step, difficulty=3, render=render, throttle=False)
        self.observation_space = Box(0, 255, shape=[3, 224, 320, 3]) 
        self.action_space = Discrete(38)
        self.device = device
        self.episode_reward = 0
        self.stage = 0
        self.monitor = monitor
        self.monitor_file = None
        self.env_id = env_id
        if self.monitor:
            self.monitor_file = open('./tmp/kof_env/' + env_id + '.log', 'a+')

    def reset(self):
        self.episode_reward = 0
        self.stage = 0
        frames, info = self.venv.reset()
        return frames

    def step(self, actions):
        # assert(actions.shape[0], 2)
        frames, reward, round_done, stage_done, game_done, info = self.venv.step(int(actions))
        
        self.episode_reward += float(reward['P1'])
        info['episode_reward'] = self.episode_reward
        info['stage'] = self.stage

        if self.monitor:
            self.monitor_file.write(str(reward['P1']) + '\t' + str(int(actions)) + '\n')
        
        if stage_done:
            self.stage += 1
        if game_done:
            info['game_done'] = True
            self.stage = 0
            if self.monitor:
                self.monitor_file.flush()

        return frames, reward, round_done or game_done or stage_done, info
    
    def close(self):
        self.venv.close()


class KOFEnvironmentShmem(VecEnvWrapper):
    def __init__(self, venv, device, stack_frame=False):
        """Return only every `skip`-th frame"""
        super(KOFEnvironmentShmem, self).__init__(venv)
        self.observation_space = Box(0, 255, shape=[3, 190, 300])
        self.device = device
        self.fraps = 0
        self.elapsed_time = 0
        self.start = 0
        self.end = 0
        self.stack_frame = stack_frame

    def reset(self):
        obs = self.venv.reset()
        obs = [transform_frames(ob, self.device, self.stack_frame) for ob in obs]
        obs = torch.cat(obs, dim=0)
        return obs

    def step_async(self, actions):
        self.start = time.time()
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, infos = self.venv.step_wait()
        obs = [transform_frames(ob, self.device, self.stack_frame) for ob in obs]
        obs = torch.cat(obs, dim=0)
        reward = torch.FloatTensor([[r['P1'] for r in reward]]).t()
        self.end = time.time()
        self.elapsed_time += self.end - self.start

        #hard reset all the environment if any one has stopped
        for info in infos:
            self.fraps += info["fraps"]
            if 'game_done' in info.keys():
                self.reset()
                info['fps'] = self.fraps / self.elapsed_time
                self.fraps = 0
                self.elapsed_time = 0
                break

        return obs, reward, done, infos


def make_env(env_id, device, frames_per_step=3, frame_ratio=3, monitor=False, render=True):
    def _thunk():
        env = KOFEnvironment(env_id, device, frames_per_step=frames_per_step, frame_ratio=frame_ratio, monitor=monitor, render=render)

        return env

    return _thunk


def make_vec_envs(num_processes, device, frames_per_step=3, frame_ratio=3, monitor=False, render=True, stack_frame=False):
    envs = [
        make_env("env" + str(i), device, frames_per_step, frame_ratio, monitor, render)
        for i in range(num_processes - 1)
    ]
    envs.append(make_env("env" + str(num_processes), device, frames_per_step, frame_ratio, monitor, True))

    envs = ShmemVecEnv(envs, context='fork')
    envs = KOFEnvironmentShmem(envs, device, stack_frame=stack_frame)

    return envs