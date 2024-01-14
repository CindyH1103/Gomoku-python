# -*- coding: utf-8 -*-
import random
import numpy as np
import os
from collections import defaultdict, deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from game import Board, Game
from mcts_alphaZero import MCTSPlayer as MCTS_AZ
from model import PolicyValueNet
import datetime
from config import *
import random
import sys
import ray
from ray import tune
from sumTree import SumTree
from mcts_pure import MCTSPlayer as MCTS_Pure
from ray.air import Checkpoint, session
import math


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def redirect_log_file(log_root="./log", exp_name="train", model_type="pure"):
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    t = str(datetime.datetime.now())

    filename = t[2:][:-7] + " || " + exp_name + "_" + model_type
    filename += ".txt"
    out_file = os.path.join(log_root, filename)
    print("Redirect log to: ", out_file, flush=True)
    sys.stdout = open(out_file, 'a')
    sys.stderr = open(out_file, 'a')
    print("Start time:", t, flush=True)


def set_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PriorityReplayBuffer(object):
    """ The class represents prioritized experience replay buffer.

    The class has functions: store samples, pick samples with
    probability in proportion to sample's priority, update
    each sample's priority, reset alpha.

    see https://arxiv.org/pdf/1511.05952.pdf .

    """

    def __init__(self, memory_size, batch_size, alpha):
        """ Prioritized experience replay buffer initialization.

        Parameters
        ----------
        memory_size : int
            sample size to be stored
        batch_size : int
            batch size to be selected by `select` method
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        """
        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.alpha = alpha

    def add(self, data, priority):
        """ Add new sample.

        Parameters
        ----------
        data : object
            new sample
        priority : float
            sample's priority
        """
        self.tree.add(data, priority ** self.alpha)

    def select(self, beta):
        """ The method return samples randomly.

        Parameters
        ----------
        beta : float

        Returns
        -------
        out :
            list of samples
        weights:
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """

        if self.tree.filled_size() < self.batch_size:
            return None, None, None

        out = []
        indices = []
        weights = []
        priorities = []
        for _ in range(self.batch_size):
            r = random.random()
            while (r == 0):
                r = random.random()
            data, priority, index = self.tree.find(r)

            while not data:
                r = random.random()
                data, priority, index = self.tree.find(r)
                print(self.tree.data)

            priorities.append(priority)
            weights.append((1. / self.memory_size / priority) ** beta if priority > 1e-16 else 0)
            indices.append(index)
            out.append(data)
            # self.priority_update([index], [0])  # To avoid duplicating

        self.priority_update(indices, priorities)  # Revert priorities

        # print('the type of weights is {}, and the type of weights[0] is{} and weights[0] = {}'.format(type(weights), type(weights[0]), weights[0]))
        try:
            # Normalize for stability
            # max_value = max(tensor.item() for tensor in weights)
            max_value = max(weights)

            # 对列表中的每个 Tensor 元素进行归一化处理
            for i in range(len(weights)):
                # weights[i] = weights[i].cpu().detach().numpy() / max_value
                weights[i] /= max_value
        except Exception:
            print('the type of weights is {}, and the type of weights[0] is{} and weights[0] = {}'.format(type(weights), type(weights[0]), weights[0]))
            print('exception occurred')
            # print(Exception)
            print(Exception)
        # print(out, weights)
        return out, weights

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p ** self.alpha)

    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.

        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i) ** -old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)

    def __len__(self):
        return self.tree.filled_size()


# ---------------------------------------------------------------------------------------------------------------------


class TrainPipeline:
    def __init__(self, lr_init, ckpt, device=torch.device('cuda'), pv_ratio=1, pc_ratio=2.0, fc_ratio=1.0):
        # params of the board and the game
        self.board = Board(width=board_size, height=board_size, n_in_row=n_in_row)
        self.game = Game(self.board, device=device)
        # training params
        self.learn_rate = lr_init
        if not priority_replay:
            self.data_buffer = deque(maxlen=buffer_size)
        else:
            print("priority replay buffer is in use")
            self.data_buffer = PriorityReplayBuffer(buffer_size, batch_size, replay_alpha)
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = pure_mcts_playout_num
        self.device = device
        self.net = PolicyValueNet(board_size,
                                  model_type=model_type,
                                  exp_name=exp_name,
                                  feature_channel=feature_channel,
                                  num_res=num_res,
                                  device=self.device)
        self.optimizer = optim.Adam(self.net.net.parameters(), weight_decay=1e-4)
        self.mcts_player = MCTS_AZ(self.net.policy_value_fn,
                                   c_puct=c_puct,
                                   n_playout=n_playout,
                                   is_selfplay=1)
        self.pv_ratio = pv_ratio
        self.pc_ratio = pc_ratio
        self.fc_ratio = fc_ratio
        if ckpt:
            params = torch.load(ckpt)
            self.load_checkpoint(params)

    def load_checkpoint(self, params):
        self.net.net.load_state_dict(params['net'])
        self.optimizer.load_state_dict(params['optim'])

    def get_equi_data(self, play_data):
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(board_size, board_size)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            _, play_data = self.game.start_self_play(self.mcts_player,
                                                     temp=temp)
            play_data = list(play_data)[:]
            # if the game length is 9, always abandon, if above 20, always save, in the middle just a linear
            if np.random.rand() <= -3 / 44 * len(play_data) + 15 / 11:
                continue
            # augment the data
            play_data = self.get_equi_data(play_data)
            if not priority_replay:
                self.data_buffer.extend(play_data)
            else:
                states = [data[0] for data in play_data]
                _, _, values = self.net.eval_state(states)
                for i in range(len(play_data)):
                    priority = abs(values[i][0] - play_data[i][2]) + replay_e
                    priority_float = priority.item()
                    self.data_buffer.add(play_data[i], priority_float)
                    # self.data_buffer.add(play_data[i], priority)

    def policy_update(self):
        """update the policy-value net"""
        if not priority_replay:
            mini_batch = random.sample(self.data_buffer, batch_size)
            weight_batch = [1] * batch_size
            weight_batch = np.array(weight_batch)
        else:
            mini_batch, weight_batch = self.data_buffer.select(replay_beta)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, _, _ = self.net.eval_state(state_batch)
        old_probs = np.exp(old_probs.data.cpu().numpy())
        for _ in range(epochs):
            loss = self.train_step(state_batch,
                                   mcts_probs_batch,
                                   winner_batch,
                                   weight_batch)
            new_probs, _, _ = self.net.eval_state(state_batch)
            new_probs = np.exp(new_probs.data.cpu().numpy())
            kl_div = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl_div > kl_targ * 4:
                break
        # adaptively adjust the learning rate
        if kl_div > kl_targ * 2 and self.learn_rate > 1e-5:
            self.learn_rate /= 1.5
        elif kl_div < kl_targ / 2 and self.learn_rate < 1e-3:
            self.learn_rate *= 1.5

        # if priority_replay:
        #     self.data_buffer.priority_update(indices, )

        print("learning rate: {}, loss: {}".format(self.learn_rate, loss), flush=True)
        return loss

    def save_checkpoint(self, file):
        """ save model params to file """
        file += "_" + exp_name + "_" + model_type + ".model"
        torch.save({"net": self.net.net.state_dict(), "optim": self.optimizer.state_dict()}, file)

    def train(self, game_batch_num_):
        filename = "current_policy" + "_" + exp_name + "_" + model_type + ".model"
        filename_best = "best_policy" + "_" + exp_name + "_" + model_type + ".model"
        for i in range(game_batch_num_):
            self.collect_selfplay_data(play_batch_size)
            print("batch i:{}".format(i + 1), flush=True)
            if len(self.data_buffer) > batch_size:
                self.policy_update()
            # check the performance of the current model, and save the model params
            if (i + 1) % check_freq == 0:
                print("current self-play batch: {}".format(i + 1), flush=True)
                self.save_checkpoint('current_policy')
                prob = np.random.rand()
                if prob >= prob_compete_with_best:
                    win_ratio, _, _ = self.game.policy_evaluate(filename,
                                                                model_type=model_type,
                                                                feature_channel=feature_channel,
                                                                num_res=num_res,
                                                                exp_name=exp_name,
                                                                pure_mcts_playout_num=self.pure_mcts_playout_num)
                    if win_ratio > self.best_win_ratio:
                        print("New best policy from pure MCTS", flush=True)
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.save_checkpoint('best_policy')
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                else:
                    win_ratio = self.game.policy_compete(filename, filename_best,
                                                         model_type=model_type,
                                                         feature_channel=feature_channel,
                                                         num_res=num_res,
                                                         exp_name=exp_name,
                                                         pure_mcts_playout_num=self.pure_mcts_playout_num)
                    if win_ratio >= 0.8:
                        print("New best policy by beating the previous best", flush=True)
                        # update the best_policy
                        self.save_checkpoint('best_policy')

    def train_step(self, state_batch, mcts_probs, winner_batch, weight_batch):
        mcts_probs = Variable(torch.FloatTensor(np.array(mcts_probs)).to(self.device))
        winner_batch = Variable(torch.FloatTensor(np.array(winner_batch)).to(self.device))
        weight_batch = torch.FloatTensor(np.array(weight_batch)).to(self.device)

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, self.learn_rate)
        policy_logits, value_logits, value = self.net.eval_state(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch, reduction="none")
        policy_loss = - torch.sum(mcts_probs * policy_logits, dim=1)
        loss = self.pv_ratio * value_loss + policy_loss

        if exp_name != "train":
            averageValues = []
            averageFeatures = []
            value_ = value.data.cpu().numpy()
            value_logits_ = value_logits.data.cpu().numpy()
            for i in range(len(value_)):
                fIndex = np.max([0, i - 2])
                lIndex = np.min([len(value_), i + 3])
                averageValues.append(np.mean(value_[fIndex: lIndex]))
                averageFeatures.append(np.mean(value_logits_[fIndex: lIndex], axis=0))
            if "PC" in exp_name:
                averageValues = torch.Tensor(np.array(averageValues)).to(self.device)
                pc_loss = F.mse_loss(value.view(-1), averageValues.view(-1), reduction='none')
                loss += self.pc_ratio * pc_loss
            if "FC" in exp_name:
                averageFeatures = torch.Tensor(np.array(averageFeatures)).to(self.device)
                fc_loss = F.mse_loss(value_logits, averageFeatures, reduction='none')
                fc_loss = torch.mean(fc_loss, dim=1)
                loss += self.fc_ratio * fc_loss

        loss *= weight_batch
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        return loss.data

    def policy_evaluate_for_tune(self, n_games=10):
        current_mcts_player = MCTS_AZ(self.net.policy_value_fn,
                                      c_puct=c_puct,
                                      n_playout=n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner, _ = self.game.start_play(current_mcts_player,
                                             pure_mcts_player,
                                             start_player=i % 2,
                                             is_shown=0)
            win_cnt[winner] += 1

        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        return win_ratio


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    set_seed()
    redirect_log_file(exp_name=exp_name, model_type=model_type)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    training_pipeline = TrainPipeline(2e-3, init_model, device)
    training_pipeline.train(game_batch_num)
