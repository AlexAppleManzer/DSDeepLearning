import random

class Game:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.state = 0
        self.score = 0

        self.apple_pos = (random.randrange(width - 1), random.randrange(height - 1))
        self.snake_head = (int(width/2), int(height/2))
        self.snake_body = []

    def start(self):
        return self.getBoard()

    def getBoard(self):
        return (self.apple_pos, self.snake_head, self.snake_body)

    def getState(self):
        return self.state

    def getScore(self):
        return self.score

    def tick(self, direction):
        # 0 North
        # 1 East
        # 2 South
        # 3 West

        if self.state != 0:
            return self.getBoard()

        if len(self.snake_body) > 0:
            self.snake_body.pop()
            self.snake_body.insert(0, self.snake_head)
        
        if direction == 0:
            self.snake_head = (
                self.snake_head[0],
                self.snake_head[1] - 1
            )
        elif direction == 1:
            self.snake_head = (
                self.snake_head[0] + 1,
                self.snake_head[1]
            )
        elif direction == 2:
            self.snake_head = (
                self.snake_head[0],
                self.snake_head[1] + 1
            )
        elif direction == 3:
            self.snake_head = (
                self.snake_head[0] - 1,
                self.snake_head[1]
            )

        if self.snake_head in self.snake_body:
            self.state = -1
        elif 0 > self.snake_head[0] or self.snake_head[0] >= self.width:
            self.state = -1
        elif 0 > self.snake_head[1] or self.snake_head[1] >= self.height:
            self.state = -1

        self.score = len(self.snake_body)
        if self.snake_head == self.apple_pos:
            if len(self.snake_body) == 0:
                self.snake_body.append(self.snake_head)
            else:
                self.snake_body.append(self.snake_body[-1])

            self.apple_pos = (random.randrange(self.width - 1), random.randrange(self.height - 1))
        elif len(self.snake_body) >= self.width * self.height:
            self.state = 1



        board = self.getBoard()
        return board
