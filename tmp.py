from env import SFEnvironmentDummy
from env import SFEnvironmentShmem
from env import SFEnvironment
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv

import torch

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

    # observation_space = Box(0, 255, shape=[3, 61, 120])
    envs = ShmemVecEnv(envs, context='fork')

    envs2 = SFEnvironmentShmem(envs, device)
    import pdb; pdb.set_trace()
    envs2.reset()
    return envs


device = torch.device("cuda:0")
make_vec_envs(2, device)