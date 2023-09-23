from car import Car


class AICAR(Car):

    def _reset_keys(self):
        self.keys['up'] = False
        self.keys['right'] = False
        self.keys['down'] = False
        self.keys['left'] = False

    def action_to_direction(self, action):
        self._reset_keys()
        if action == 0:
            self.keys['up'] = True
        elif action == 1:
            self.keys['right'] = True
            self.keys['up'] = True
        if action == 2:
            self.keys['right'] = True
        if action == 3:
            self.keys['right'] = True
            self.keys['down'] = True
        if action == 4:
            self.keys['down'] = True
        if action == 5:
            self.keys['left'] = True
            self.keys['down'] = True
        if action == 6:
            self.keys['left'] = True
        if action == 7:
            self.keys['left'] = True
            self.keys['up'] = True

    def decide(self):
        pass

    def game_over(self):
        pass
