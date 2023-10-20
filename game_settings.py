from enum import Enum


class GameStatus(Enum):
    DRAW_BOUNDARIES = 0
    DRAW_GOALS = 1
    USER_CONTROLS = 2
    AI_TRAIN = 3

    def __str__(self):
        if self == GameStatus.DRAW_BOUNDARIES:
            return "Drawing Boundaries"
        elif self == GameStatus.DRAW_GOALS:
            return "Drawing Goals"
        elif self == GameStatus.USER_CONTROLS:
            return "User Controls"
        elif self == GameStatus.AI_TRAIN:
            return "AI Training"


class GameSettings:
    # const
    WINDOW_HEIGHT = 600
    WINDOW_WIDTH = 1000
    RENDER_FPS = 60

    FONT_SIZE = 10

    GAME_STATUS_LABEL_X = 20
    GAME_STATUS_LABEL_Y = 35
    DIST_LABELS_POSITION_X = WINDOW_WIDTH - 80
    DIST_LABELS_POSITION_Y = WINDOW_HEIGHT - 50
    GOAL_LABEL_POSITION_X = 20
    GOAL_LABEL_POSITION_Y = WINDOW_HEIGHT - 15

    BOUNDARY_COLOR = (180, 180, 0, 255)
    SENSOR_COLOR = (50, 150, 50, 255)
    GOAL_COLOR = (0, 0, 180, 255)

    LINE_WIDTH = 2
    SENSOR_LENGTH = 50
    INTERSECTION_POINT_SIZE = 4

    MOUSE_CLICK_HIT_BOX = 15
    CAR_HIT_BOX = 3

    def __init__(self, game_status=GameStatus.DRAW_BOUNDARIES):
        self.GAME_STATUS = game_status
