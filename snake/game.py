# import the pygame module, so you can use it
import pygame
import time

from logic import Game

width = 10
height = 10
 
def render_board(board, screen, assets):
    # board: Apple, Head, Body
    # assets: apple, head, body
    screen.fill((72, 49, 0))
    
    def multiply_tuple(t, scalar):
        return tuple(item * 100 for item in t)

    screen.blit(assets[0], multiply_tuple(board[0], 100))
    screen.blit(assets[1], multiply_tuple(board[1], 100))

    for segment in board[2]:
        screen.blit(assets[2], multiply_tuple(segment, 100))

def text_objects(text, font):
    textSurface = font.render(text, True, (0,0,0))
    return textSurface, textSurface.get_rect()

def display_message(text, screen):
    largeText = pygame.font.Font('freesansbold.ttf', 115)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = (width / 2 * 100, height / 2 * 100)
    screen.blit(TextSurf, TextRect)

# define a main function
def main():
     
    # initialize the pygame module
    pygame.init()
    # load and set the logo
    logo = pygame.image.load("assets/snek.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("snake")

    # load images for later
    head_image = pygame.image.load("assets/head.png")
    body_image = pygame.image.load("assets/body.png")
    apple_image = pygame.image.load("assets/apple.png")
    assets = (apple_image, head_image, body_image)
     
    # create a surface on screen that has the size of 240 x 180
    screen = pygame.display.set_mode((width * 100, height * 100))

    game = Game(width, height)
    board = game.start()

    render_board(board, screen, assets)
    pygame.display.flip()


    
    # define a variable to control the main loop
    running = True
    gameRunning = True
    direction = 0

    # main loop
    while running:
        # event handling, gets all event from the event queue
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    direction = 0
                if event.key == pygame.K_RIGHT:
                    direction =  1
                if event.key == pygame.K_DOWN:
                    direction = 2
                if event.key == pygame.K_LEFT:
                    direction = 3

        if gameRunning:
            board = game.tick(direction)
            if game.getState() == 0:
                render_board(board, screen, assets)
                pygame.display.flip()
                pygame.time.wait(150)
            if game.getState() == 1:
                display_message(f"You win! Score: {game.getScore()}", screen)
                pygame.display.flip()
                gameRunning = False
            if game.getState() == -1:
                display_message(f"You lose! Score: {game.getScore()}", screen)
                pygame.display.flip()
                gameRunning = False
     
# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()