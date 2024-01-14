import sys
import os
import math


class SumTree(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.tree_level = math.ceil(math.log(max_size + 1, 2)) + 1
        self.tree_size = 2 ** self.tree_level - 1
        self.tree = [0 for _ in range(self.tree_size)]
        self.data = [None for _ in range(self.max_size)]
        self.size = 0
        self.cursor = 0

    def add(self, contents, value):
        index = self.cursor
        self.cursor = (self.cursor + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        self.data[index] = contents
        self.val_update(index, value)

    def get_val(self, index):
        tree_index = 2 ** (self.tree_level - 1) - 1 + index
        return self.tree[tree_index]

    def val_update(self, index, value):
        tree_index = 2 ** (self.tree_level - 1) - 1 + index
        diff = value - self.tree[tree_index]
        self.reconstruct(tree_index, diff)

    def reconstruct(self, tindex, diff):
        self.tree[tindex] += diff
        if not tindex == 0:
            tindex = int((tindex - 1) / 2)
            self.reconstruct(tindex, diff)

    def find(self, value, norm=True):
        # print('enter function find, value = {}'.format(value))
        if norm:
            value *= self.tree[0]
            # print('value={} and self.tree[0]={}'.format(value, self.tree[0]))
        result = self._find(value, 0)
        # return self._find(value, 0)
        # if (result[0] == None):
            # print('in find, value={}, the root node weight = {}'.format(value, self.tree[0]))
            # print('have found')
        # print('return function find')
        return result

    def _find(self, value, index):
        # 判断index对应的节点是否为叶节点
        # print('enter _find, value={}, index={}'.format(value, index))
        # print(type(value))
        if 2**(self.tree_level-1)-1 <= index:
            # print('leaf node')
            # print("data index: {}, tree index: {}".format(index-(2**(self.tree_level-1)-1), index))
            # if (index-(2**(self.tree_level-1)-1)) > (self.size-1):
                # print('index out of range, {} > {}'.format(index-(2**(self.tree_level-1)-1), self.size-1))
            # if (self.data[index-(2**(self.tree_level-1)-1)] == None):
                # print('data size: {}'.format(self.size))
                # print("data index: {}, tree index: {}".format(index-(2**(self.tree_level-1)-1), index))
            return self.data[index-(2**(self.tree_level-1)-1)], self.tree[index], index-(2**(self.tree_level-1)-1)

        # 这里是考虑左子节点
        left = self.tree[2*index+1]
        # print('left = {}'.format(left))

        if value <= left:
            # print('smaller than left')
            return self._find(value,2*index+1)
        else:
            # print('larger than left')
            return self._find(value-left,2*(index+1))

    def filled_size(self):
        return self.size

