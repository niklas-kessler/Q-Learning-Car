from car import Car


class AICar(Car):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward = 0

    def _reset_keys(self):
        self.keys['up'] = False
        self.keys['right'] = False
        self.keys['down'] = False
        self.keys['left'] = False

    def action(self, action):
        self._reset_keys()
        if action == 0:
            self.keys['up'] = True
        elif action == 1:
            self.keys['right'] = True
            self.keys['up'] = True
        elif action == 2:
            self.keys['right'] = True
        elif action == 3:
            self.keys['right'] = True
            self.keys['down'] = True
        elif action == 4:
            self.keys['down'] = True
        elif action == 5:
            self.keys['left'] = True
            self.keys['down'] = True
        elif action == 6:
            self.keys['left'] = True
        elif action == 7:
            self.keys['left'] = True
            self.keys['up'] = True

    def decide(self):
        pass
