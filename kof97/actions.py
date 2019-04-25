from enum import Enum

class Action(object):

    def __init__(self, port, field):
        self.port = port
        self.field = field

    def get_lua_string(self):
        return 'iop.ports["' + self.port + '"].fields["' + self.field + '"]'


class SpecialAction(object):

    def __init__(self, port, field, press):
        self.port = port
        self.field = field
        self.press = press

    def get_lua_string(self):
        return 'iop.ports["' + self.port + '"].fields["' + self.field + '"]:set_value(' + str(self.press) +')'


# An enumerable class used to specify which actions can be used to interact with a game
# Specifies the Lua engine port and field names required for performing an action
class Actions(Enum):
    # Starting
    SERVICE =   Action(':TEST', 'Service Mode')

    COIN_P1 =   Action(':AUDIO/COIN', 'Coin 1')
    COIN_P2 =   Action(':AUDIO/COIN', 'Coin 2')

    P1_START =  Action(':edge:joy:START', '1 Player Start')
    P2_START =  Action(':edge:joy:START', '2 Players Start')

    # Movement
    P1_UP =     Action(':edge:joy:JOY1', 'P1 Up')
    P1_DOWN =   Action(':edge:joy:JOY1', 'P1 Down')
    P1_LEFT =   Action(':edge:joy:JOY1', 'P1 Left')
    P1_RIGHT =  Action(':edge:joy:JOY1', 'P1 Right')

    P2_UP =     Action(':edge:joy:JOY2', 'P2 Up')
    P2_DOWN =   Action(':edge:joy:JOY2', 'P2 Down')
    P2_LEFT =   Action(':edge:joy:JOY2', 'P2 Left')
    P2_RIGHT =  Action(':edge:joy:JOY2', 'P2 Right')

    # Fighting
    P1_BUTTON1 = Action(':edge:joy:JOY1', 'P1 Button 1')
    P1_BUTTON2 = Action(':edge:joy:JOY1', 'P1 Button 2')
    P1_BUTTON3 = Action(':edge:joy:JOY1', 'P1 Button 3')
    P1_BUTTON4 = Action(':edge:joy:JOY1', 'P1 Button 4')


    P2_BUTTON1 = Action(':edge:joy:JOY2', 'P2 Button 1')
    P2_BUTTON2 = Action(':edge:joy:JOY2', 'P2 Button 2')
    P2_BUTTON3 = Action(':edge:joy:JOY2', 'P2 Button 3')
    P2_BUTTON4 = Action(':edge:joy:JOY2', 'P2 Button 4')


class SActions(Enum):

    # Movement
    D0 =   SpecialAction(':edge:joy:JOY1', 'P1 Down', 0)
    D1 =   SpecialAction(':edge:joy:JOY1', 'P1 Down', 1)
    L0 =   SpecialAction(':edge:joy:JOY1', 'P1 Left', 0)
    L1 =   SpecialAction(':edge:joy:JOY1', 'P1 Left', 1)
    R0 =  SpecialAction(':edge:joy:JOY1', 'P1 Right', 0)
    R1 =  SpecialAction(':edge:joy:JOY1', 'P1 Right', 1)
    A1 =  SpecialAction(':edge:joy:JOY1', 'P1 Button 1', 1)
    A0 =  SpecialAction(':edge:joy:JOY1', 'P1 Button 1', 0)
    B1 =  SpecialAction(':edge:joy:JOY1', 'P1 Button 2', 1)
    B0 =  SpecialAction(':edge:joy:JOY1', 'P1 Button 2', 0)
    C1 =  SpecialAction(':edge:joy:JOY1', 'P1 Button 3', 1)
    C0 =  SpecialAction(':edge:joy:JOY1', 'P1 Button 3', 0)
    DD1 =  SpecialAction(':edge:joy:JOY1', 'P1 Button 4', 1)
    DD0 =  SpecialAction(':edge:joy:JOY1', 'P1 Button 4', 0)


normal_actions = [

[Actions.P1_BUTTON1, Actions.P1_BUTTON2],
[Actions.P1_DOWN],
[Actions.P1_DOWN, Actions.P1_RIGHT],
[Actions.P1_UP],
[Actions.P1_UP, Actions.P1_LEFT],
[Actions.P1_UP, Actions.P1_RIGHT],
[Actions.P1_LEFT],
[Actions.P1_RIGHT],
[Actions.P1_BUTTON1],
[Actions.P1_BUTTON2],
[Actions.P1_BUTTON3],
[Actions.P1_BUTTON4],
[Actions.P1_DOWN, Actions.P1_BUTTON1],
[Actions.P1_DOWN, Actions.P1_BUTTON2],
[Actions.P1_DOWN, Actions.P1_BUTTON3],
[Actions.P1_DOWN, Actions.P1_BUTTON4],
]

step_dict = {
    1: [Actions.P1_DOWN, Actions.P1_LEFT],
    2: [Actions.P1_DOWN],
    3: [Actions.P1_DOWN, Actions.P1_RIGHT],
    4: [Actions.P1_LEFT],
    6: [Actions.P1_RIGHT],
    'A': [Actions.P1_BUTTON1],
    'B': [Actions.P1_BUTTON2],
    'C': [Actions.P1_BUTTON3],
    'D': [Actions.P1_BUTTON4],
}

step_actions= [
    [SActions.D1, SActions.L1, SActions.D0, SActions.L0, SActions.C1, SActions.C0],
]

step_actions_old= [
    [2, 1, 4, 'C'],
]