import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

from logic import Game

from itertools import count
from collections import namedtuple
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        print(self.memory)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# class DQN(nn.Module):
#     def __init__(self, h, w, outputs):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
#         self.bn3 = nn.BatchNorm2d(32)

#         # Number of Linear input connections depends on output of conv2d layers
#         # and therefore the input image size, so compute it.
#         def conv2d_size_out(size, kernel_size = 5, stride = 2):
#             return (size - (kernel_size - 1) - 1) // stride  + 1
#         convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
#         convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
#         linear_input_size = convw * convh * 32
#         self.head = nn.Linear(linear_input_size, outputs)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         return self.head(x.view(x.size(0), -1))

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(w * h, w * h)
        self.layer2 = nn.Linear(w * h, w * h)
        self.layer3 = nn.Linear(w * h, outputs)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.log_softmax(self.layer3(x))
        return x

BATCH_SIZE = 4
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
width = 10
height = 10
n_actions = 4


policy_net = DQN(height, width, n_actions).to(device)
target_net = DQN(height, width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

episode_durations = []

def get_screen(game):
    def lower_dimension(thing):
        return thing[0] + thing[1] * width

    game_screen = game.getBoard()
    size = width*height
    screen = np.zeros(((size), (size), (size)))
    screen[0][lower_dimension(game_screen[0])] = 1
    screen[1][lower_dimension(game_screen[1])] = 1
    for item in game_screen[2]:
        screen[2][lower_dimension(item)] = 1 

    return torch.from_numpy(screen).unsqueeze(0).float().to(device)

# def select_action(state):
#     global steps_done
#     sample = random.random()
#     eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#         math.exp(-1. * steps_done / EPS_DECAY)
#     steps_done += 1
#     if sample > eps_threshold:
#         with torch.no_grad():
#             # t.max(1) will return largest column value of each row.
#             # second column on max result is index of where max element was
#             # found, so we pick action with the larger expected reward.
#             return policy_net(state).max(1)[1].view(1, 1)
#     else:
#         return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def select_action(state):
    return torch.argmax(policy_net.forward(state))


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward).squeeze()
    print(action_batch.shape)
    print(reward_batch.shape)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()


    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


import pygame
from game import render_board
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

    num_episodes = 300
    for i_episode in range(num_episodes):
        game = Game(width, height)
        last_screen = get_screen(game)
        current_screen = get_screen(game)
        for t in count():
            render_board(game.getBoard(), screen, assets)
            pygame.display.flip()
            # Select and perform an action
            action = select_action(current_screen)
            reward = game.getScore()
            done = game.getState != 0
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(game)

            # Store the transition in memory
            memory.push(last_screen, action, current_screen, reward)

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                print(f"score: {game.getScore()}")
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')

if __name__=="__main__":
    # call the main function
    main()