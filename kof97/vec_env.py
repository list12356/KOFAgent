
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env import ShmemVecEnv
from utils import resize_image
from kof97.env import Environment
import torch
import numpy as np


# def transform_frames(frames, device):
#     import pdb; pdb.set_trace()
#     # process the frames
#     x = []
#     for frame in frames:
#         frame = frame[32:214, 12:372]  # crop
#         frame = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]  # greyscale
#         frame = frame[::3, ::3]  # downsample
#         frame = frame / 255
#         frame = frame - frame.mean()
#         x.append(torch.FloatTensor(frame.reshape(1, 61, 120)).to(device))
#     return torch.stack(x, dim=1)


def transform_frames(frames, device):
    # only get the first frame
    
    frame = frames[-1]
    frame = frame[20:210, 9:312]  # crop
    frame = resize_image(frame.astype(np.uint8), (125, 75))
    frame = frame.transpose(2, 0, 1)
    return torch.FloatTensor(frame).to(device).unsqueeze(0)

class KOFEnvironmentDummy():
    def __init__(self, device, frames_per_step=3, throttle=False):
        roms_path = "./roms/"  # Replace this with the path to your ROMs
        self.venv = Environment("env1", roms_path, frame_ratio=3, frames_per_step=frames_per_step, difficulty=3, throttle=throttle)
        self.observation_space = Box(0, 255, shape=[3, 75, 125]) # use stack here
        self.action_space = Discrete(35)
        self.device = device
        self.epsiode_reward = 0
        self.monitor = open('./tmp/kof_env_2.log', 'w+')
        self.started = False
        self.stage = 0

    def reset(self):
        frames, info = self.venv.reset()
        return transform_frames(frames, self.device), [info]

    def step(self, actions):
        frames, reward, round_done, stage_done, game_done, info = self.venv.step(int(actions))
        

        if round_done:
            frames, info = self.venv.reset()
            if stage_done:
                self.stage += 1
            if game_done:
                info['r'] = self.epsiode_reward
                self.monitor.flush()
                info['stage'] = self.stage
                self.epsiode_reward = 0
                self.stage = 0

        frames = transform_frames(frames, self.device)
        rewards = torch.FloatTensor([reward['P1']]).unsqueeze(dim=0) # for later parallel, the current num_process = 1
        self.epsiode_reward += float(reward['P1'])
        self.monitor.write(str(reward['P1']) + '\t' + str(int(actions)) + '\n')

        return frames, rewards, [round_done or game_done or stage_done], [info]

    def close(self):
        self.venv.close()

class KOFEnvironment():
    def __init__(self, env_id, device):
        roms_path = "./roms/"  # Replace this with the path to your ROMs
        self.venv = Environment(env_id, roms_path, frame_ratio=2, frames_per_step=3, difficulty=3)
        self.observation_space = Box(0, 255, shape=[3, 224, 320, 3]) 
        self.action_space = Discrete(35)
        self.device = device
        self.epsiode_reward = 0
        self.started = False
        self.stage = 0

    def reset(self):
        if not self.started:
            self.started = True
            return self.venv.start()
        return self.venv.reset()

    def step(self, actions):
        # assert(actions.shape[0], 2)
        frames, reward, round_done, stage_done, game_done, info = self.venv.step(int(actions))
        
        if round_done:
            frames = self.venv.reset()
            if stage_done:
                self.stage += 1
            if game_done:
                info['r'] = self.epsiode_reward
                info['stage'] = self.stage
                self.epsiode_reward = 0
                self.stage = 0

        self.epsiode_reward += float(reward['P1'])

        return frames, reward, round_done or game_done or stage_done, info
    
    def close(self):
        self.venv.close()


class KOFEnvironmentShmem(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(KOFEnvironmentShmem, self).__init__(venv)
        self.observation_space = Box(0, 255, shape=[3, 75, 125])
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = [transform_frames(ob, self.device) for ob in obs]
        obs = torch.cat(obs, dim=0)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = [transform_frames(ob, self.device) for ob in obs]
        obs = torch.cat(obs, dim=0)
        reward = torch.FloatTensor([[r['P1'] for r in reward]]).t()

        #hard reset all the environment if any one has stopped

        return obs, reward, done, info


def make_env(env_id, device):
    def _thunk():
        env = KOFEnvironment(env_id, device)

        return env

    return _thunk


def make_vec_envs(num_processes,
                  device,
                  num_frame_stack=None):
    envs = [
        make_env("env" + str(i), device)
        for i in range(num_processes)
    ]

    envs = ShmemVecEnv(envs, context='fork')
    envs = KOFEnvironmentShmem(envs, device)

    return envs