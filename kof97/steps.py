from kof97.actions import Actions

# A = Agent
# C = Computer
# H = Human
# An enurable class used to specify the set of action steps required to perform different predefined tasks
# E.g. changing the story mode difficulty, or starting a new game in single player story mode
# def set_difficulty(frame_ratio, difficulty):
#     steps = [
#         {"wait": 0, "actions": [Actions.SERVICE]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_JPUNCH]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_JPUNCH]}]
#     if (difficulty % 8) < 3:
#         steps += [{"wait": int(10/frame_ratio), "actions": [Actions.P1_LEFT]} for i in range(3-(difficulty % 8))]
#     else:
#         steps += [{"wait": int(10/frame_ratio), "actions": [Actions.P1_RIGHT]} for i in range((difficulty % 8)-3)]
#     steps += [
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_JPUNCH]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_JPUNCH]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_JPUNCH]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
#         {"wait": int(10/frame_ratio), "actions": [Actions.P1_JPUNCH]}]
#     return steps


def start_game(frame_ratio):
    return [
        {"wait": 1, "actions": [Actions.COIN_P1]},
        {"wait": 1, "actions": [Actions.COIN_P1]},
        {"wait": 1, "actions": [Actions.COIN_P1]},
        {"wait": 1, "actions": [Actions.COIN_P1]},
        {"wait": 1, "actions": [Actions.COIN_P1]},
        {"wait": 1, "actions": [Actions.COIN_P1]},
        {"wait": int(60/frame_ratio), "actions": [Actions.P1_START]},
        {"wait": int(60/frame_ratio), "actions": [Actions.P1_BUTTON1]},
        {"wait": int(180/frame_ratio), "actions": [Actions.P1_BUTTON1]},
        {"wait": int(120/frame_ratio), "actions": [Actions.P1_RIGHT]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_RIGHT]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_RIGHT]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_BUTTON1]},
        {"wait": int(180/frame_ratio), "actions": [Actions.P1_BUTTON1]},
        {"wait": int(60/frame_ratio), "actions": [Actions.P1_BUTTON1]}]


def next_stage(frame_ratio):
    return [{"wait": int(360/frame_ratio), "actions": [Actions.P1_BUTTON1]},
            {"wait": int(360/frame_ratio), "actions": [Actions.P1_BUTTON1]},
            {"wait": int(360/frame_ratio), "actions": [Actions.P1_BUTTON1]},
            {"wait": int(360/frame_ratio), "actions": [Actions.P1_BUTTON1]},
            {"wait": int(60/frame_ratio), "actions": [Actions.P1_BUTTON1]}
        ]

def new_game(frame_ratio):
    return [
            {"wait": int(60/frame_ratio), "actions": [Actions.SERVICE]},
            {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
            {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
            {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
            {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
            {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
            {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
            {"wait": int(10/frame_ratio), "actions": [Actions.P1_BUTTON1]},
            {"wait": int(360/frame_ratio), "actions": [Actions.COIN_P1]}
    ]
