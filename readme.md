This repository is for CS838 project. Training an AI player for King of Fighters 97 arcade game. Inspired while I play game with my friend at home and by https://github.com/M-J-Murray/SFAgents and using the https://github.com/M-J-Murray/MAMEToolkit tool kit.

##Requirement

install the openai baselines https://github.com/M-J-Murray/MAMEToolkit

put the altered MAME emulator into /usr/game folder or put it anywhere that is in your $PATH environment variable

install the MAME toolkit from https://github.com/M-J-Murray/MAMEToolkit

example usage:

python main.py --exp_name=myexp --num_process=4

more options can be found in utils/cmd_utils.py

##ATTENTION

don't distribute the roms under this folder, it is illegal.
