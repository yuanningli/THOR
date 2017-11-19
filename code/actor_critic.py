import argparse
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

import robosims
import cv2
import recog_stream
import json

# choose architecture
architecture = 'ResNet'
num_samples = 400
if architecture == 'ResNet':
    num_features = 2048
else:
    num_features = 4096

# initialize environment
env = robosims.controller.ChallengeController(
    unity_path='thor-201705011400-OSXIntel64.app/Contents/MacOS/thor-201705011400-OSXIntel64',
    x_display="0.0" # this parameter is ignored on OSX, but you must set this to the appropriate display on Linux
)
env.start()
recog_net = recog_stream.RecogNet(architecture)

# parse input
parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['action', 'value'])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4096, 1024)
        self.affine2 = nn.Linear(1024, 256)
        self.action_head = nn.Linear(256, 8)
        self.value_head = nn.Linear(256, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x1 = F.relu(self.affine1(x))
        x2 = F.relu(self.affine2(x1))
        action_scores = self.action_head(x2)
        state_values = self.value_head(x2)
        return F.softmax(action_scores), state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs, state_value = model(Variable(state))
    action = probs.multinomial()
    model.saved_actions.append(SavedAction(action, state_value))
    return action.data


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    value_loss = 0
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for (action, value), r in zip(saved_actions, rewards):
        reward = r - value.data[0,0]
        action.reinforce(reward)
        value_loss += F.smooth_l1_loss(value, Variable(torch.Tensor([r])))
    optimizer.zero_grad()
    final_nodes = [value_loss] + list(map(lambda p: p.action, saved_actions))
    gradients = [torch.ones(1)] + [None] * len(saved_actions)
    autograd.backward(final_nodes, gradients)
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def get_target_feature(target_name, recog_model):
    target_image = cv2.imread("thor-challenge-targets/" + target_name['targetImage'])
    target_image = cv2.resize(target_image, (300, 300))
    return recog_model.feat_extract(target_image).squeeze()


def get_state_feature(current_event, recog_model, target_feat):
    img = current_event.frame
    img_feat = recog_model.feat_extract(img).squeeze()
    return torch.cat((img_feat, target_feat), 0).data.numpy()


action_sets = ['MoveLeft', 'MoveRight', 'MoveAhead', 'MoveBack', 'LookUp', 'LookDown', 'RotateRight', 'RotateLeft']
running_reward = 10

with open("thor-challenge-targets/targets-train.json") as f:
    current_targets = json.loads(f.read())

    for target in current_targets:
        # initialize
        env.initialize_target(target)
        # convert target image
        target_feature = get_target_feature(target, recog_net)
        event = env.step(action=dict(action='MoveAhead'))

        for i_episode in count(1):
            env.initialize_target(target)
            state = get_state_feature(event, recog_net, target_feature)
            for t in range(10000):  # Don't infinite loop while learning
                action = select_action(state)
                event = env.step(action=dict(action=action_sets[int(action[0, 0])]))
                state = get_state_feature(event, recog_net, target_feature)
                done = env.target_found()
                if not done:
                    reward = -1
                else:
                    reward = 100
                model.rewards.append(reward)
                if done:
                    break

            running_reward = running_reward * 0.99 + t * 0.01
            finish_episode()
            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                    i_episode, t, running_reward))
            if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, t))
                break
