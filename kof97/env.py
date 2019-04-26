from MAMEToolkit.emulator.Address import Address
from kof97.steps import *
from kof97.actions import *
from kof97.Emulator import Emulator

# Returns the list of memory addresses required to train on Street Fighter
def setup_memory_addresses():
    return {
        "round": Address('0x10A7F1', 'u8'),
        "stage": Address('0x10A798', 'u8'),
        "start": Address('0x1081E2', 'u8'),
        "winsP1": Address('0x10A85B', 'u8'),
        "winsP2": Address('0x10A84A', 'u8'),
        "healthP1": Address('0x108239', 'u8'),
        "healthP2": Address('0x108439', 'u8'),
        "positionP1": Address('0x108118', 'u16'),
        "positionP2": Address('0x108318', 'u16'),
        "powerP1": Address('0x1082E3', 'u8')
    }


# The Street Fighter specific interface for training an agent against the game
class Environment(object):

    # env_id - the unique identifier of the emulator environment, used to create fifo pipes
    # difficulty - the difficult to be used in story mode gameplay
    # frame_ratio, frames_per_step - see Emulator class
    # render, throttle, debug - see Console class
    def __init__(self, env_id, roms_path, difficulty=3, frame_ratio=3, frames_per_step=3, render=True, throttle=False, debug=False):
        self.difficulty = difficulty
        self.frame_ratio = frame_ratio
        self.frames_per_step = frames_per_step
        self.throttle = throttle
        self.emu = Emulator(env_id, roms_path, "kof97", setup_memory_addresses(), frame_ratio=frame_ratio, render=render, throttle=throttle, debug=debug)
        self.started = False
        self.expected_health = {"P1": 0, "P2": 0}
        self.expected_wins = {"P1": 0, "P2": 0}
        self.round_done = False
        self.stage_done = False
        self.game_done = False
        self.round = 0
        self.stage = 0

    # Runs a set of action steps over a series of time steps
    # Used for transitioning the emulator through non-learnable gameplay, aka. title screens, character selects
    def run_steps(self, steps):
        for step in steps:
            for i in range(step["wait"]):
                self.emu.step([])
            self.emu.step([action.value for action in step["actions"]])

    # Must be called first after creating this class
    # Sends actions to the game until the learnable gameplay starts
    # Returns the first few frames of gameplay
    def start(self):
        if self.throttle:
            for i in range(int(250/self.frame_ratio)):
                self.emu.step([])
        
        # self.run_steps(set_difficulty(self.frame_ratio, self.difficulty))
        self.run_steps(start_game(self.frame_ratio))
        self.started = True
        return self.wait_for_fight_start()

    # Observes the game and waits for the fight to start
    def wait_for_fight_start(self):
        data = self.emu.step([])
        while data["start"] == 0x08:
            data = self.emu.step([])
        frames = []
        for i in range(self.frames_per_step):
            data = self.emu.step([])
            frames.append(data['frame'])
        info = {}
        info["stage"] = self.stage
        info["positionP1"] = data["positionP1"]
        info["positionP2"] = data["positionP2"]
        info["powerP1"] = data["powerP1"]
        return frames, info
    
    # wait until the next stage
    def wait_for_stage(self):
        data = self.emu.step([])
        while data["start"] != 0x08:
            data = self.emu.step([])

    def reset(self):
        if not self.started:
            return self.start()
        if self.game_done:
            return self.new_game()
        elif self.stage_done:
            return self.next_stage()
        elif self.round_done:
            return self.next_round()
        else:
            raise EnvironmentError("Reset called while gameplay still running")
            # do hard reset
            # return self.new_game()

    # To be called when a round finishes
    # Performs the necessary steps to take the agent to the next round of gameplay
    def next_round(self):
        self.round_done = False
        self.wait_for_stage()
        # import pdb; pdb.set_trace()
        self.expected_health = {"P1": 0x67, "P2": 0x67}
        return self.wait_for_fight_start()

    # To be called when a game finishes
    # Performs the necessary steps to take the agent(s) to the next game and resets the necessary book keeping variables
    def next_stage(self):
        self.run_steps(next_stage(self.frame_ratio))
        self.wait_for_stage()
        self.expected_health = {"P1": 0x67, "P2": 0x67}
        self.expected_wins = {"P1": 0, "P2": 0}
        self.round_done = False
        self.stage_done = False
        self.round = 0
        self.stage += 1
        return self.wait_for_fight_start()

    def new_game(self):
        # self.wait_for_stage()
        self.run_steps(new_game(self.frame_ratio))
        self.run_steps(start_game(self.frame_ratio))
        self.expected_health = {"P1": 0x67, "P2": 0x67}
        self.expected_wins = {"P1": 0, "P2": 0}
        self.round_done = False
        self.stage_done = False
        self.game_done = False
        self.round = 0
        self.stage = 0
        return self.wait_for_fight_start()

    # Checks whether the round or game has finished
    def check_done(self, data):
        if data["round"] != self.round:
            self.round_done = True
            self.round = data["round"]
            if data["winsP1"] == 2:
                self.stage_done = True
            if data["winsP2"] == 2:
                self.game_done = True

    # Steps the emulator along by the requested amount of frames required for the agent to provide actions
    def step(self, action):
        if self.started:
            if not self.round_done and not self.stage_done and not self.game_done:
                info = {}
                data = {}
                frames = []
                if action < 18:
                    actions = normal_actions[action]
                    data = self.emu.step([action.value for action in actions])
                    while len(frames) < self.frames_per_step:
                        data = self.emu.step([action.value for action in actions])
                        frames.append(data["frame"])
                else:
                    steps = step_actions[action - 18]
                    for action in steps:
                        data = self.emu.step([action.value for action in step_dict[action]])
                        if len(frames) < self.frames_per_step:
                            frames.append(data["frame"])

                self.check_done(data)
                p1_diff = (self.expected_health["P1"] - data["healthP1"])
                p2_diff = (self.expected_health["P2"] - data["healthP2"])
                self.expected_health = {"P1": data["healthP1"], "P2": data["healthP2"]}

                rewards = {
                    "P1": (p2_diff-p1_diff),
                    "P2": (p1_diff-p2_diff)
                }
                # ad-hoc, since the reward may be hacked
                if self.round_done:
                    rewards = {"P1": 0, "P2": 0}

                if abs(p2_diff-p1_diff) > 100:
                    rewards = {"P1": 0, "P2": 0}

                data["rewards"] = rewards
                data["frame"] = frames
                info["stage"] = self.stage
                info["positionP1"] = data["positionP1"]
                info["positionP2"] = data["positionP2"]
                info["powerP1"] = data["powerP1"]
                return data["frame"], data["rewards"], self.round_done, self.stage_done, self.game_done, info
            else:
                # if self.round_done or self.stage_done:
                raise EnvironmentError("Attempted to step while characters are not fighting")
        else:
            raise EnvironmentError("Start must be called before stepping")

    # Safely closes emulator
    def close(self):
        self.emu.close()
