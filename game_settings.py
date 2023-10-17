from enum import Enum


class GameStatus(Enum):
    DRAW_BOUNDARIES = 0
    USER_CONTROLS = 1
    AI_TRAIN = 2


class GameSettings:
    # const
    WINDOW_HEIGHT = 600
    WINDOW_WIDTH = 800
    RENDER_FPS = 60

    BOUNDARY_COLOR = (180, 180, 0, 255)
    SENSOR_COLOR = (50, 150, 50, 255)

    LINE_WIDTH = 2
    SENSOR_LENGTH = 50

    MOUSE_CLICK_HIT_BOX = 15
    CAR_HIT_BOX = 3

    def __init__(self, game_status=GameStatus.DRAW_BOUNDARIES):
        self.GAME_STATUS = game_status
