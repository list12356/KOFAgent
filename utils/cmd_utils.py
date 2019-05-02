
import argparse

def cmd_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default='CNNSimple-Stack-Explore')
    parser.add_argument("--num_steps", default=128, type=int)
    parser.add_argument("--num_process", default=4, type=int)
    parser.add_argument("--ppo_epoch", default=4, type=int)
    parser.add_argument("--num_mini_batch", default=4, type=int)
    parser.add_argument("--frames_per_step", default=3, type=int)
    parser.add_argument("--stack_frame", default=0, type=int)
    parser.add_argument("--render", default=1, type=int)
    parser.add_argument("--throttle", default=0, type=int)
    parser.add_argument("--monitor", default=1, type=int)
    parser.add_argument("--explore", default=1, type=int)
    parser.add_argument("--explore_scale", default=0.2, type=float)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--restore", default=0, type=int)
    parser.add_argument("--restore_cnn", default=0, type=int)
    parser.add_argument("--lr", default=2.5e-4, type=float)
    parser.add_argument("--use_linear_lr_decay", default=1, type=int)
    parser.add_argument("--restore_path", default='./saved_models/CNNSimple/')
    return parser