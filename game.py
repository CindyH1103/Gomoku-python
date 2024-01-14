from __future__ import print_function

import random

import numpy as np
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from collections import defaultdict, deque
from model import PolicyValueNet
import time
import pickle
import pygame
import mydraw
import sys
import os
import torch

# 关于画图的一些数据
SIZE = 90  # 棋盘每个点之间的间隔
Line_Points = 9  # 棋盘每行/每列点数
Outer_Width = 60  # 棋盘外宽度
Border_Width = 12  # 边框宽度
Inside_Width = 12  # 边框跟实际的棋盘之间的间隔
Border_Length = SIZE * (Line_Points - 1) + Inside_Width * 2 + Border_Width  # 边框线的长度
Start_X = Start_Y = Outer_Width + int(Border_Width / 2) + Inside_Width  # 网格线起点（左上角）坐标
SCREEN_HEIGHT = SIZE * (Line_Points - 1) + Outer_Width * 2 + Border_Width + Inside_Width * 2  # 游戏屏幕的高
SCREEN_WIDTH = SCREEN_HEIGHT + 650  # 游戏屏幕的宽
Text_X = Start_X + Border_Length + 30  # 文字的坐标
Text_Y = Start_Y
Line_Space = 120  # 文字行间距

Stone_Radius = SIZE // 2 - 9  # 棋子半径
Stone_Radius2 = SIZE // 4  # 图例棋子半径
Checkerboard_Color = (0xE3, 0x92, 0x65)  # 棋盘颜色
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
RED_COLOR = (200, 30, 30)
BLUE_COLOR = (30, 30, 200)


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 9))
        self.height = int(kwargs.get('height', 9))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board
        self.pure_mcts_playout_num = 200
        self.device = kwargs.get('device')

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1, human=0):
        if human == 1:
            pos = list()
            """start a game between two players"""
            if start_player not in (0, 1):
                raise Exception('start_player should be either 0 (player1 first) '
                                'or 1 (player2 first)')
            self.board.init_board(start_player)
            p1, p2 = self.board.players
            player1.set_player_ind(p1)
            player2.set_player_ind(p2)
            players = {p1: player1, p2: player2}

            color = [(0, 0, 0), (255, 255, 255), (128, 128, 128)]
            pygame.init()
            my_font = pygame.font.SysFont("arial", 60)
            my_font_small = pygame.font.SysFont("arial", 40)
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            step_time = ""
            done = False
            clock = pygame.time.Clock()
            while not done:
                # 代替原来的while True，还是一次只执行一步

                # 不关闭窗口就一直有，但是需要判断game_end。若已结束，不再执行任何动作，只是不断刷新屏幕。
                # 每刷新一次就新获取一次状态
                # 一会儿挪到后面

                # 人的状态从鼠标获取
                # 需要传回去，if鼠标有动作且合法，调个game里的函数获取当前人类动作
                # 马上刷新屏幕状态显示上去
                # 此时PC根据人类动作落子
                # 还是存在poses里，这样保证不丢数据
                clock.tick(10)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                # 填充棋盘背景色
                screen.fill(Checkerboard_Color)
                # 画棋盘网格线外的边框
                pygame.draw.rect(screen, BLACK_COLOR, (Outer_Width, Outer_Width, Border_Length, Border_Length),
                                 Border_Width)
                # 画网格线
                for i in range(Line_Points):
                    pygame.draw.line(screen, BLACK_COLOR,
                                     (Start_Y, Start_Y + SIZE * i),
                                     (Start_Y + SIZE * (Line_Points - 1), Start_Y + SIZE * i),
                                     1)
                for j in range(Line_Points):
                    pygame.draw.line(screen, BLACK_COLOR,
                                     (Start_X + SIZE * j, Start_X),
                                     (Start_X + SIZE * j, Start_X + SIZE * (Line_Points - 1)),
                                     1)

                # 添加文字信息
                # 轮数
                text_round = my_font.render("Human vs PC", True, color[0])
                screen.blit(text_round, (Text_X, Text_Y))
                # 图例
                text_PCplayer = my_font_small.render("MCTS_AlphaZero", True, color[0])
                text_Ourplayer = my_font_small.render("Human Player", True, color[0])
                pygame.draw.circle(screen, color[0], [Text_X + Stone_Radius2, Text_Y + Line_Space], Stone_Radius2)
                pygame.draw.circle(screen, color[1], [Text_X + Stone_Radius2, Text_Y + Line_Space * 1.5], Stone_Radius2)
                screen.blit(text_PCplayer, (Text_X + Stone_Radius2 * 3, Text_Y + Line_Space - Stone_Radius2 - 5))
                screen.blit(text_Ourplayer, (Text_X + Stone_Radius2 * 3, Text_Y + Line_Space * 1.5 - Stone_Radius2 - 5))
                # 电脑下棋所用时间
                text_steptime = my_font_small.render("Time taken by AlphaZero: " + step_time, True, color[0])
                screen.blit(text_steptime, (Text_X, Text_Y + Line_Space * 2.5 - Stone_Radius2 - 5))
                # 画所有棋子
                for ele in pos:
                    [i, j] = ele[1]
                    x = Start_X + SIZE * i
                    y = Start_Y + SIZE * j
                    pygame.draw.circle(screen, color[ele[0] - 1], [x, y], Stone_Radius)

                text_end = my_font_small.render("Game end.", True, color[0])
                # 游戏结束
                # 不再执行游戏逻辑但是继续画图
                end, winner = self.board.game_end()
                if end:
                    if is_shown:
                        if winner != -1:
                            print("Game end. Winner is", players[winner])
                        else:
                            print("Game end. Tie")
                    screen.blit(text_end, (Text_X, Text_Y + Line_Space * 3))
                    if winner == -1:
                        text_winner = my_font_small.render("Tie.", True, color[0])
                    elif winner == 1:
                        text_winner = my_font_small.render("The winner is PC.", True, color[0])
                    else:
                        text_winner = my_font_small.render("The winner is human player.", True, color[0])
                    screen.blit(text_winner, (Text_X, Text_Y + Line_Space * 3.5))
                    pygame.display.flip()
                    continue
                    #  return winner, pos

                pygame.display.flip()

                current_player = self.board.get_current_player()
                player_in_turn = players[current_player]
                # print("Current player:", current_player, "player in turn:", player_in_turn)
                # 人类玩家等左键
                if current_player == 2:
                    key_pressed = pygame.mouse.get_pressed()
                    x, y = pygame.mouse.get_pos()
                    [i, j] = player_in_turn.find_pos(x, y)
                    if i >= 0 and i < Line_Points and j >= 0 and j < Line_Points:
                        x = Start_X + SIZE * i
                        y = Start_Y + SIZE * j
                        pygame.draw.circle(screen, color[1], [x, y], Stone_Radius)
                        pygame.display.flip()
                    if key_pressed[0]:
                        move = player_in_turn.get_action(self.board)
                        if move == -1 or move not in board.availables:
                            print("invalid move")
                            continue
                    else:
                        continue
                else:
                    move, step_time = player_in_turn.get_action(self.board)
                    # 电脑下棋时显示时间，人类下棋时不要显示
                pos.append((current_player, self.board.move_to_location(move)))
                self.board.do_move(move)

            pygame.quit()

        else:
            pos = list()
            """start a game between two players"""
            if start_player not in (0, 1):
                raise Exception('start_player should be either 0 (player1 first) '
                                'or 1 (player2 first)')
            self.board.init_board(start_player)
            p1, p2 = self.board.players
            player1.set_player_ind(p1)
            player2.set_player_ind(p2)
            players = {p1: player1, p2: player2}
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            while True:
                current_player = self.board.get_current_player()
                player_in_turn = players[current_player]
                move, _ = player_in_turn.get_action(self.board)
                pos.append((current_player, self.board.move_to_location(move)))
                self.board.do_move(move)
                if is_shown:
                    self.graphic(self.board, player1.player, player2.player)
                end, winner = self.board.game_end()
                if end:
                    if is_shown:
                        if winner != -1:
                            print("Game end. Winner is", players[winner])
                        else:
                            print("Game end. Tie")
                    return winner, pos

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players, moves = [], [], [], []
        while True:
            move, move_probs, _ = player.get_action(self.board,
                                                    temp=temp,
                                                    return_prob=1)
            # store the data
            moves.append(move)
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)

    def policy_evaluate(self, checkpoint='./best_policy_train_pure.model', n_games=10, model_type="pure",
                        feature_channel=256, num_res=1, exp_name="train", pure_mcts_playout_num=200):
        pvnet = PolicyValueNet(self.board.width, checkpoint=checkpoint, model_type=model_type,
                               feature_channel=feature_channel, num_res=num_res, exp_name=exp_name, device=self.device)
        current_mcts_player = MCTSPlayer(policy_value_fn=pvnet.policy_value_fn, c_puct=5,
                                         n_playout=pure_mcts_playout_num)
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        poses = list()
        winners = list()
        time_all = 0
        for i in range(n_games):
            start = time.time()
            winner, pos = self.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            end = time.time()
            time_all += end - start
            win_cnt[winner] += 1
            poses.append(pos)
            players = ["MCTS_Pure.", "Our Player.", "Tie"]
            winners.append(players[winner])

        # self.start_play(current_mcts_player, pure_mcts_player, start_player=1)
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        print("average time: {}".format(time_all / n_games))
        return win_ratio, poses, winners

    def human_play(self, checkpoint='./best_policy_train_res.model', model_type="res", feature_channel=256, num_res=5,
                   exp_name="train", pure_mcts_playout_num=200):
        pvnet = PolicyValueNet(self.board.width, checkpoint=checkpoint, model_type=model_type,
                               feature_channel=feature_channel, num_res=num_res, exp_name=exp_name, device=self.device)
        current_mcts_player = MCTSPlayer(policy_value_fn=pvnet.policy_value_fn, c_puct=5,
                                         n_playout=pure_mcts_playout_num)
        human_player = Human_Player()
        self.start_play(current_mcts_player, human_player, start_player=random.randint(0, 1), is_shown=0, human=1)

    def policy_compete(self, checkpoint1='./current_policy.model', checkpoint2='./best_policy.model',
                       n_games=10, model_type="pure", feature_channel=256, num_res=1, exp_name="train",
                       pure_mcts_playout_num=200):
        pvnet1 = PolicyValueNet(self.board.width, checkpoint=checkpoint1, model_type=model_type,
                                feature_channel=feature_channel, num_res=num_res, exp_name=exp_name, device=self.device)
        current_mcts_player = MCTSPlayer(policy_value_fn=pvnet1.policy_value_fn, c_puct=5,
                                         n_playout=pure_mcts_playout_num)
        pvnet2 = PolicyValueNet(self.board.width, checkpoint=checkpoint2, model_type=model_type,
                                feature_channel=feature_channel, num_res=num_res, exp_name=exp_name, device=self.device)
        best_mcts_player = MCTSPlayer(policy_value_fn=pvnet2.policy_value_fn, c_puct=5,
                                      n_playout=pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        time_all = 0
        for i in range(n_games):
            start = time.time()
            winner, _ = self.start_play(current_mcts_player,
                                        best_mcts_player,
                                        start_player=i % 2,
                                        is_shown=0)
            end = time.time()
            time_all += end - start
            win_cnt[winner] += 1

        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        print("average time: {}".format(time_all / n_games))
        return win_ratio

    def policy_compete1(self, checkpoint1='./current_policy.model', checkpoint2='./best_policy.model',
                        model_type1="pure", model_type2="res",
                        n_games=10, feature_channel1=256, feature_channel2=256, num_res=5, exp_name="train",
                        pure_mcts_playout_num=200):
        pvnet1 = PolicyValueNet(self.board.width, checkpoint=checkpoint1, model_type=model_type1,
                                feature_channel=feature_channel1, num_res=num_res, exp_name=exp_name, device=self.device)
        current_mcts_player = MCTSPlayer(policy_value_fn=pvnet1.policy_value_fn, c_puct=5,
                                         n_playout=pure_mcts_playout_num)
        pvnet2 = PolicyValueNet(self.board.width, checkpoint=checkpoint2, model_type=model_type2,
                                feature_channel=feature_channel2, num_res=num_res, exp_name=exp_name, device=self.device)
        best_mcts_player = MCTSPlayer(policy_value_fn=pvnet2.policy_value_fn, c_puct=5,
                                      n_playout=pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        time_all = 0
        for i in range(n_games):
            start = time.time()
            winner, _ = self.start_play(current_mcts_player,
                                        best_mcts_player,
                                        start_player=i % 2,
                                        is_shown=0)
            end = time.time()
            time_all += end - start
            win_cnt[winner] += 1

        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        print("average time: {}".format(time_all / n_games))
        return win_ratio


class Human_Player(object):
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def find_pos(self, x, y):
        x = (x - Start_X + SIZE // 2) // SIZE
        y = (y - Start_Y + SIZE // 2) // SIZE
        return [x, y]

    def get_action(self, board):
        # 等待鼠标点击，如果不点就卡着不动——卡着不动就好
        # location = input("Type in the next move in format: x, y, Please type in the move: ")
        # location = location.split(",")
        # location = [int(n) for n in location]
        x, y = pygame.mouse.get_pos()
        # print(x, y)
        location = self.find_pos(x, y)
        move = board.location_to_move(location)
        # print(move)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICE'] = '0'
    board_width = 9
    board_height = 9
    n_in_row = 5
    board = Board(width=board_width,
                  height=board_height,
                  n_in_row=n_in_row,
                  device=torch.device('cuda'))
    task = Game(board, device=torch.device('cuda'))
    # win_ratio = task.policy_compete1("best_policy_finetune_FC_pure.model", "best_policy_finetune_PC_pure.model", "pure", "pure")
    # print(win_ratio)

    # finetune PC
    # _, poses, _ = task.policy_evaluate(checkpoint='./best_policy_finetune_PC_gomokuNet.model', model_type='gomokuNet', feature_channel=128, num_res=5, pure_mcts_playout_num=1000, n_games=10)
    # print('gomokuNet_PC average steps: ', sum([len(pose) for pose in poses]) / 10)
    # _, poses, _ = task.policy_evaluate(checkpoint='./best_policy_finetune_PC_res.model', n_games=10, model_type="res", feature_channel=256, num_res=5, pure_mcts_playout_num=1000)
    # print('res_PC average steps: ', sum([len(pose) for pose in poses]) / 10)
    # _, poses, _ = task.policy_evaluate(checkpoint='./best_policy_finetune_PC_pure.model', n_games=10, model_type="pure", feature_channel=256, num_res=5, pure_mcts_playout_num=1000)
    # print('pure_PC average steps: ', sum([len(pose) for pose in poses]) / 10)
    #
    # # train
    # _, poses, _ = task.policy_evaluate(checkpoint='./best_policy_train_gomokuNet.model', model_type='gomokuNet', feature_channel=128, num_res=5, pure_mcts_playout_num=1000, n_games=10)
    # print('gomokuNet_train average steps: ', sum([len(pose) for pose in poses]) / 10)
    # _, poses, _ = task.policy_evaluate(checkpoint='./best_policy_train_res.model', n_games=10, model_type="res", feature_channel=256, num_res=5, pure_mcts_playout_num=1000)
    # print('res_train average steps: ', sum([len(pose) for pose in poses]) / 10)
    # _, poses, _ = task.policy_evaluate(checkpoint='./best_policy_train_pure.model', n_games=10,
    #                                                  model_type="pure", feature_channel=256, num_res=5,
    #                                                  pure_mcts_playout_num=1000)
    # print('pure_train average steps: ', sum([len(pose) for pose in poses]) / 10)

    # mode = int(input("Choose mode:    1. ai vs ai    2. human vs ai\n"))
    # if mode == 1:
    #     _, poses, winners = task.policy_evaluate(pure_mcts_playout_num=200, n_games=10)
    #     mydraw._draw(poses, winners, board_width)
    # elif mode == 2:
    #   task.human_play(pure_mcts_playout_num=200)
    # else:
    #     exit(-1)
