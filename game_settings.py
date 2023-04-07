from enum import Enum


class GameStatus(Enum):
    DRAW_BOUNDARIES = 0
    USER_CONTROLS = 1
    AI_TRAIN = 2


class GameSettings:
    # const
    WINDOW_HEIGHT = 600
    WINDOW_WIDTH = 800

    BOUNDARY_COLOR = (180, 180, 0, 255)
    MOUSE_CLICK_HIT_BOX = 15

    def __init__(self, game_status=GameStatus.DRAW_BOUNDARIES):
        self.GAME_STATUS = game_status
