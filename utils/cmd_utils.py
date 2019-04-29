
import argparse

def cmd_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default='CNNSimple-Stack-Explore')
    parser.add_argument("--num_steps", default=128, type=int)
    parser.add_argument("--num_process", default=4, type=int)
    parser.add_argument("--ppo_epoch", default=4, type=int)
    parser.add_argument("--num_mini_batch", default=4, type=int)
    parser.add_argument("--frames_per_step", default=3, type=int)
    parser.add_argument("--stack_frame", default=False)
    parser.add_argument("--render", default=True)
    parser.add_argument("--throttle", default=False)
    parser.add_argument("--monitor", default=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--restore_path", default='./saved_models/CNNSimple/1800.base')
    return parser