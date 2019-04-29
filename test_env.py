from kof97.env import Environment

env = Environment("env_01", roms_path="./roms", throttle=False)
env.reset()

import numpy as np
import time

elapsed = 0
fraps = 0
stages = 0

while(True):
    # action = np.random.randint(17)
    action = np.random.randint(35)
    start = time.time()
    frame, reward, round_done, stage_done, game_done, info = env.step(action)
    end = time.time()
    elapsed += end - start
    fraps += info["fraps"]
    # print('reward_P1: {}, reward_P2: {}'.format(reward['P1'], reward['P2']))
    info = {}
    if stage_done:
        stages += 1
        print("stages: " + str(stages))
    if game_done:
        print("Fps: " + str(fraps/elapsed))
        elapsed = 0
        fraps = 0
        env.reset()