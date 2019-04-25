
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.box import Box
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env import ShmemVecEnv
import torch


def transform_frames(frames, device):
    # process the frames
    x = []
    for frame in frames:
        frame = frame[32:214, 12:372]  # crop
        frame = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]  # greyscale
        frame = frame[::3, ::3]  # downsample
        frame = frame / 255
        frame = frame - frame.mean()
        x.append(torch.FloatTensor(frame.reshape(1, 61, 120)).to(device))
    return torch.stack(x, dim=1)

class SFEnvironmentDummy():
    def __init__(self, device):
        from MAMEToolkit.sf_environment import Environment
        roms_path = "./roms/"  # Replace this with the path to your ROMs
        self.venv = Environment("env1", roms_path, frame_ratio=2, frames_per_step=3, difficulty=3)
        self.observation_space = Box(0, 255, shape=[3, 61, 120]) # use stack here
        self.action_space = MultiDiscrete([8, 9])
        self.device = device
        self.epsiode_reward = 0
        self.started = False
    
    def start(self):
        return transform_frames(self.venv.start(), self.device)

    def reset(self):
        if not self.started:
            self.started = True
            return transform_frames(self.venv.start(), self.device)
        return transform_frames(self.venv.reset(), self.device)

    def step(self, actions):
        assert(actions.shape[1], 2)
        frames, reward, round_done, stage_done, game_done = self.venv.step(int(actions[0, 0]), int(actions[0, 1]))
        
        info = {}
        if round_done:
            frames = self.venv.reset()
            if game_done:
                info['r'] = self.epsiode_reward
                self.epsiode_reward = 0

        frames = transform_frames(frames, self.device)
        rewards = torch.FloatTensor([reward['P1']]).unsqueeze(dim=0) # for later parallel, the current num_process = 1
        self.epsiode_reward += float(reward['P1'])

        return frames, rewards, [round_done or game_done or stage_done], info

    def close(self):
        self.venv.close()

class SFEnvironment():
    def __init__(self, env_id, device):
        from MAMEToolkit.sf_environment import Environment
        roms_path = "./roms/"  # Replace this with the path to your ROMs
        self.venv = Environment(env_id, roms_path, frame_ratio=2, frames_per_step=3, difficulty=3)
        self.observation_space = Box(0, 255, shape=[3, 224, 384, 3]) # use stack here
        self.action_space = MultiDiscrete([8, 9])
        self.device = device
        self.epsiode_reward = 0
        self.started = False

    def reset(self):
        if not self.started:
            self.started = True
            return self.venv.start()
        return self.venv.reset()

    def step(self, actions):
        # assert(actions.shape[0], 2)
        frames, reward, round_done, stage_done, game_done = self.venv.step(int(actions[0]), int(actions[1]))
        
        info = {}
        if round_done:
            # frames = self.venv.reset()
            if game_done:
                info['r'] = self.epsiode_reward
                self.epsiode_reward = 0

        self.epsiode_reward += float(reward['P1'])

        return frames, reward, round_done or game_done or stage_done, info
    
    def close(self):
        self.venv.close()


class SFEnvironmentShmem(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(SFEnvironmentShmem, self).__init__(venv)
        self.observation_space = Box(0, 255, shape=[3, 61, 120])
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

        return obs, reward, done, info


def make_env(env_id, device):
    def _thunk():
        env = SFEnvironment(env_id, device)

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
    envs = SFEnvironmentShmem(envs, device)

    return envs