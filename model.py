# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = int(np.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1))
    w = int(np.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1))
    return h, w


class ResNetBlock(nn.Module):
    """Resnet Block for retrieving features from board states"""

    def __init__(self, num_channels):
        super(ResNetBlock, self).__init__()
        self.conv_block_relu = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=num_channels),
            nn.ReLU(),
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=num_channels),
        )

    def forward(self, x):
        res = x
        ret = self.conv_block_relu(x)
        ret = self.conv_block(ret)
        ret += res
        ret = F.relu(ret)
        return ret


class Net_Res(nn.Module):
    """Alpha-Zero Net with ResBlocks and with padding=3 to enhance edge case performance"""

    def __init__(self, board_size, feature_channel, num_res_blocks):
        super(Net_Res, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(4, feature_channel, kernel_size=3, padding=3),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU()
        )

        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResNetBlock(feature_channel))
        self.res_blocks = nn.Sequential(*res_blocks)

        convh, convw = calc_conv2d_output((board_size, board_size), 3, 1, 3)
        conv_out = convh * convw

        self.policy_head = nn.Sequential(
            nn.Conv2d(feature_channel, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * conv_out, board_size ** 2),
            nn.LogSoftmax(dim=1),
        )

        self.value_logits_head = nn.Sequential(
            nn.Conv2d(feature_channel, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * conv_out, 64),
            nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.res_blocks(x)
        policy_logits = self.policy_head(x)
        val_logits = self.value_logits_head(x)
        val = self.value_head(val_logits)
        return policy_logits, val_logits, val


class PAM(nn.Module):
    def __init__(self, input_size, in_channel):
        super(PAM, self).__init__()
        self.input_size = input_size
        self.avgPoolx = nn.AvgPool2d(kernel_size=(input_size, 3), padding=(0, 1), stride=1)
        self.avgPooly = nn.AvgPool2d(kernel_size=(3, input_size), padding=(1, 0), stride=1)

        self.fc = nn.Linear(2 * input_size, 2 * input_size)
        self.convx = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1)
        self.convy = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ret = x
        Sx = self.avgPoolx(x)
        Sy = self.avgPooly(x).transpose(3, 2)

        Sxy = torch.cat((Sx, Sy), dim=-1)
        Sxy = self.fc(Sxy)
        Sx = Sxy[:, :, :, :self.input_size].clone()
        Sy = Sxy[:, :, :, self.input_size:].clone()

        Sx = self.convx(Sx).transpose(3, 2).view(-1, self.input_size, 1)
        Sy = self.convy(Sy).view(-1, 1, self.input_size)
        Xa = torch.bmm(Sx, Sy).view(x.size(0), x.size(1), self.input_size, self.input_size)
        Xa = self.sigmoid(Xa)
        ret = ret * Xa
        return ret


class Res_PAM(nn.Module):
    def __init__(self, input_size, in_channel):
        super(Res_PAM, self).__init__()
        self.res_pam = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            PAM(input_size, in_channel)
        )

    def forward(self, x):
        ret = x
        x_ = self.res_pam(x)
        ret = ret + x_
        return ret


class Net_Gomoku(nn.Module):
    """Alpha-Zero Net with ResBlocks and with padding=3 to enhance edge case performance"""

    def __init__(self, board_size, feature_channel):
        super(Net_Gomoku, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(4, feature_channel, kernel_size=3, padding=3),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU()
        )

        convh, convw = calc_conv2d_output((board_size, board_size), 3, 1, 3)
        conv_out = convh * convw

        self.block = nn.Sequential(
            ResNetBlock(feature_channel),
            Res_PAM(convh, feature_channel),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(feature_channel, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * conv_out, board_size ** 2),
            nn.LogSoftmax(dim=1),
        )

        self.value_logits_head = nn.Sequential(
            nn.Conv2d(feature_channel, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * conv_out, 64),
            nn.ReLU(),
            nn.LogSoftmax(dim=1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.block(x)
        policy_logits = self.policy_head(x)
        val_logits = self.value_logits_head(x)
        val = self.value_head(val_logits)
        return policy_logits, val_logits, val


class Net_Pure(nn.Module):
    """Alpha-Zero Net with only convolution layer"""

    def __init__(self, board_size):
        super(Net_Pure, self).__init__()
        padding = 1

        self.conv_block = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=padding),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * board_size ** 2, board_size ** 2),
            nn.LogSoftmax(dim=1)
        )

        self.value_logits_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size ** 2, 64),
            nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_block(x)
        policy_logits = self.policy_head(x)
        val_logits = self.value_logits_head(x)
        val = self.value_head(val_logits)
        return policy_logits, val_logits, val


class PolicyValueNet:
    """policy-value network """

    def __init__(self, board_size, device, checkpoint=None,
                 model_type="pure", feature_channel=256, num_res=1, exp_name="train"):
        self.board_size = board_size
        self.exp_name = exp_name
        if model_type == "res":
            self.net = Net_Res(board_size, feature_channel, num_res).cuda()
        elif model_type == "pure":
            self.net = Net_Pure(board_size).cuda()
        else:
            self.net = Net_Gomoku(board_size, feature_channel).cuda()
        self.device = device
        self.net.to(device)
        self.model_type = model_type
        if checkpoint:
            params = torch.load(checkpoint)
            self.net.load_state_dict(params['net'])

        # if checkpoint:
        #     params = torch.load(checkpoint)
        #     self.net.load_state_dict(params['net'])
        #     self.optimizer.load_state_dict(params['optim'])

    def eval_state(self, state_batch):
        state_batch = np.array(state_batch)
        state_batch = Variable(torch.FloatTensor(state_batch).to(self.device))
        policy_logits, value_logits, value = self.net(state_batch)
        return policy_logits, value_logits, value

    def policy_value_fn(self, board):
        legal_positions = np.array(board.availables, dtype=int)
        current_state = torch.FloatTensor(np.ascontiguousarray(board.current_state().reshape(
            -1, 4, self.board_size, self.board_size))).to(self.device)
        current_state = Variable(current_state)
        policy_logits, _, value = self.net(current_state)
        act_probs = np.exp(policy_logits.data.cpu().numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    # def train_step(self, state_batch, mcts_probs, winner_batch, weight_batch, lr):
    #     state_batch = Variable(torch.FloatTensor(np.array(state_batch)).cuda())
    #     mcts_probs = Variable(torch.FloatTensor(np.array(mcts_probs)).cuda())
    #     winner_batch = Variable(torch.FloatTensor(np.array(winner_batch)).cuda())
    #     weight_batch = torch.FloatTensor(np.array(weight_batch)).cuda()

    #     self.optimizer.zero_grad()
    #     set_learning_rate(self.optimizer, lr)
    #     policy_logits, value_logits, value = self.net(state_batch)
    #     value_loss = F.mse_loss(value.view(-1), winner_batch, reduction="none")
    #     policy_loss = -torch.sum(mcts_probs * policy_logits, dim=1)
    #     loss = value_loss + policy_loss

    #     if self.exp_name != "train":
    #         averageValues = []
    #         averageFeatures = []
    #         value_ = value.data.cpu().numpy()
    #         value_logits_ = value_logits.data.cpu().numpy()
    #         for i in range(len(value_)):
    #             fIndex = np.max([0, i - 2])
    #             lIndex = np.min([len(value_), i + 3])
    #             averageValues.append(np.mean(value_[fIndex: lIndex]))
    #             averageFeatures.append(np.mean(value_logits_[fIndex: lIndex]), axis=0)
    #         if "PC" in self.exp_name:
    #             averageValues = torch.Tensor(np.array(averageValues)).cuda()
    #             pc_loss = F.mse_loss(value.view(-1), averageValues.view(-1), reduction='none')
    #             loss += self.beta * pc_loss
    #         if "FC" in self.exp_name:
    #             averageFeatures = torch.Tensor(np.array(averageFeatures)).cuda()
    #             fc_loss = F.mse_loss(value_logits, averageFeatures, reduction='none')
    #             loss += self.lamb * fc_loss

    #     loss *= weight_batch
    #     loss = loss.mean()
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss.data
