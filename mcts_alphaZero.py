# -*- coding: utf-8 -*-
"""
a modified MCTS for alphaZero based on "mcts_pure.py" provided
"""

import numpy as np
import copy
from operator import itemgetter
import time


def rollout_policy_fn(board):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # different from pure MCTS, we just direcly propagate without performing rollouts
        action_probs, leaf_value = self._policy(state)
        # Check for end of game
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # unlike in pure MCTS, instead of performing a roll out ,we just propagate the value along the path
            # if we encounter a terminal state, we propagate the actual reward
            # reference: https://web.stanford.edu/~surag/posts/alphazero.html
            if winner == state.get_current_player():
                # the player win the game
                leaf_value = 1
            elif winner == -1:
            #     # the case of tie
                leaf_value = 0
            else:
                # the player lose the game
                leaf_value = -1
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_and_probs(self, state, temp):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state
        AlphaZero's idea is that if a node is visited more often then it should
        be assigned a high probability.
        According to the paper, the prob -> visit_count^(1/N)

        Return: possible action and corresponding probability
        """
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        acts = list(self._root._children.keys())
        num_visits = np.array([child._n_visits for child in self._root._children.values()])
        probs = 1.0 / temp * np.log(num_visits + 1e-10)
        probs = np.exp(probs - np.max(probs))
        probs /= np.sum(probs)

        return acts, probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000, weight=0.75, alpha=0.3, is_selfplay=False):
        # guesses are that alpha should be set to 10/n where n is the average number of legal moves in the games
        # here all default values are from the papers
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self.alpha = alpha
        self.weight = weight
        self.is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        start = time.time()
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            overall_probs = np.zeros(board.width * board.height)
            moves, move_probs = self.mcts.get_move_and_probs(board, temp)
            if self.is_selfplay:
                move_probs_ = self.weight * move_probs + (1 - self.weight) * np.random.dirichlet(self.alpha*np.ones(len(moves)))
                move = np.random.choice(moves, p=move_probs_)
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(moves, p=move_probs)
                # the action used in mcts pure
                self.mcts.update_with_move(-1)
            overall_probs[list(moves)] = move_probs
            step_time = round(time.time() - start, 4)
            if return_prob:
                return move, overall_probs, str(step_time) + "s"
            else:
                return move, str(step_time) + "s"
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
