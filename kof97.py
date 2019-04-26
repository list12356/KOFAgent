
# from MAMEToolkit.emulator.Emulator import Emulator
# from MAMEToolkit.emulator.Address import Address

# def setup_memory_addresses():
#     return {
#         "stage": Address('0x10A7F1', 'u16'),
#         "start": Address('0x1081E2', 'u16'),
#         "winsP1": Address('0x10A85B', 'u16'),
#         "winsP2": Address('0x10A841', 'u16'),
#         "healthP1": Address('0x108239', 'u16'),
#         "healthP2": Address('0x108439', 'u16')
#     }

# Emulator("env_01", "./roms", "kof97", setup_memory_addresses(), frame_ratio=3, render=True, throttle=True, debug=False)

from kof97.env import Environment

env = Environment("env_01", roms_path="./roms", throttle=True)
env.reset()

import numpy as np

while(True):
    # action = np.random.randint(17)
    action = np.random.randint(33)
    frame, reward, round_done, stage_done, game_done, info = env.step(action)
    # import pdb; pdb.set_trace()
    print('{}, {}'.format(reward['P1'], reward['P2']))
    info = {}
    if round_done:
        frames = env.reset()
        