#!env python
# -*- coding: utf-8 -*-
# Mancala solver for Python 2.7 and 3.5
import six
import sys
import argparse
import os
import copy
import numpy as np

class Board(object):
    BLACK    = 1 << 0
    WHITE    = 1 << 1
    HOLE     = 1 << 2
    KALAH    = 1 << 3
    def __init__(self, init_stones=4, size=6):
        # <-
        # |0|[1][2][3][4][5][6]| |  : white
        # | |[d][c][b][a][9][8]|7|  : black
        # ->
        self.size = size
        self.n_pockets = size * 2 + 2
        self.data = np.zeros(self.n_pockets, np.int32)

        self.black_holes = list(range(size + 2, self.n_pockets))
        self.white_holes = list(range(1, size + 1))
        self.black_kalah = size + 1
        self.white_kalah = 0
        self.black_holes_and_kalah = self.black_holes + [self.black_kalah]
        self.white_holes_and_kalah = self.white_holes + [self.white_kalah]

        self.mask = np.zeros(self.n_pockets, np.int32)
        self.mask[self.black_holes] = Board.BLACK | Board.HOLE
        self.mask[self.black_kalah] = Board.BLACK | Board.KALAH
        self.mask[self.white_holes] = Board.WHITE | Board.HOLE
        self.mask[self.white_kalah] = Board.WHITE | Board.KALAH

        self.opposite_holes = {}
        for i in range(1, size + 1):
            j = self.n_pockets - i
            self.opposite_holes[i] = j
            self.opposite_holes[j] = i

        self.next_hole = [(i + self.n_pockets - 1) % self.n_pockets for i in range(self.n_pockets)]
        self.prev_hole = [(i + 1) % self.n_pockets for i in range(self.n_pockets)]

        self.data[self.black_holes] = init_stones
        self.data[self.white_holes] = init_stones

    def finalize_game(self):
        for index in self.black_holes:
            self.data[self.black_kalah] += self.data[index]
            self.data[index] = 0
        for index in self.white_holes:
            self.data[self.white_kalah] += self.data[index]
            self.data[index] = 0

    def drop(self, from_index, n_stones):
        while n_stones > 0:
            self.data[from_index] += 1
            n_stones -= 1
            from_index = self.next_hole[from_index]
        last_drop_index = self.prev_hole[from_index]
        return last_drop_index

    def pick(self, index):
        if isinstance(index, Move):
            index = index.index
        n_stones = self.data[index]
        self.data[index] = 0
        last_drop_index = self.drop(index - 1, n_stones)
        return last_drop_index

    def moveable(self, index, color):
        if isinstance(index, Move):
            index = index.index
        if color == Color.black:
            return (index in self.black_holes) and self.data[index] > 0
        if color == Color.white:
            return (index in self.white_holes) and self.data[index] > 0
        return False

    def move(self, index, color, events = []):
        if isinstance(index, Move):
            index = index.index
        assert self.moveable(index, color)
        # pick
        last_drop_index = self.pick(index)
        # gather
        if self.data[last_drop_index] == 1:
            if color == Color.black and last_drop_index in self.black_holes:
                j = self.opposite_holes[last_drop_index]
                if self.data[j] > 0:
                    self.data[self.black_kalah] += self.data[j] + self.data[last_drop_index]
                    self.data[j] = 0
                    self.data[last_drop_index] = 0
                    events.append('gather')
            elif color == Color.white and last_drop_index in self.white_holes:
                j = self.opposite_holes[last_drop_index]
                if self.data[j] > 0:
                    self.data[self.white_kalah] += self.data[j] + self.data[last_drop_index]
                    self.data[j] = 0
                    self.data[last_drop_index] = 0
                    events.append('gather')
        # gain-turn
        if color == Color.black and last_drop_index == self.black_kalah:
            events.append('gain-turn')
            return Color.black
        if color == Color.white and last_drop_index == self.white_kalah:
            events.append('gain-turn')
            return Color.white
        # normal turn end
        return - color

    def pretty(self):
        return u'\n'.join([
            u'   ' + u''.join('#{:<2d} '.format(i) for i in range(self.size + 1)),
            u'W |{:2d}|'.format(self.data[0]) + u''.join('[{:2d}]'.format(i) for i in self.data[1:self.size + 1]) + u'|  |',
            u'B |  |' + u''.join('[{:2d}]'.format(i) for i in self.data[self.size + 2:][::-1]) + u'|{:2d}|'.format(self.data[self.size + 1]),
            u'      ' + u''.join('#{:<2d} '.format(i) for i in range(self.size * 2 + 2 - 1, self.size, -1)),
            ])

    def tentative_stones_balance(self):
        # (black stones, white stones)
        return self.data[self.black_holes_and_kalah].sum(), self.data[self.white_holes_and_kalah].sum()


def evaluate_board(board, color):
    if True:
        # only stones in the kahala.
        b = board.data[board.black_kalah]
        w = board.data[board.white_kalah]
    else:
        # all stones.
        b, w = board.tentative_stones_balance()

    if color == Color.black:
        return b - w
    else:
        return w - b


class Color(object):
    def __init__(self, value):
        self.side = value
    def __neg__(self):
        return Color([0, 2, 1][self.side])
    def __eq__(self, rhs):
        return self.side == rhs.side
    def __repr__(self):
        return ['X', 'B', 'W'][self.side]
Color.none = Color(0)
Color.black = Color(1)
Color.white = Color(2)

class Move(object):
    def __init__(self, index = -1):
        self.index = index
    def __repr__(self):
        return 'm{}'.format(self.index)

def generate_legal_moves(stack, color):
    board = stack[-1]
    if color == Color.black:
        moves = [Move(i) for i in board.black_holes if board.moveable(i, color)]
    elif color == Color.white:
        moves = [Move(i) for i in board.white_holes if board.moveable(i, color)]
    return moves

class SearchInfo(object):
    def __init__(self):
        self.nodes = 0

def search_dfs(stack, root_color, turn_color, max_depth, depth, search_info):
    # return PV := [(leaf status, color, move, score)]
    search_info.nodes += 1

    moves = generate_legal_moves(stack, turn_color)
    if len(moves) == 0:
        # no more move possible. game end.
        b = copy.deepcopy(stack[-1])
        b.finalize_game()
        score = evaluate_board(b, root_color)
        return [('NL', None, turn_color, score)]

    if depth == 0:
        # static evaluate
        score = evaluate_board(stack[-1], root_color)
        return [('LE', None, turn_color, score)]

    minimax_score = None
    minimax_node = None
    minimax_color = None
    minimax_pv = []
    for move in moves:
        # next depth
        next_stack = stack + [copy.deepcopy(stack[-1])]
        next_color = next_stack[-1].move(move, turn_color)
        pv = search_dfs(next_stack, root_color, next_color, max_depth, depth - 1, search_info)
        _, _, _, pv_score = pv[-1]
        if turn_color == root_color:
            do_update = minimax_score is None or pv_score > minimax_score
        else:
            do_update = minimax_score is None or pv_score < minimax_score
        if do_update:
            minimax_score = pv_score
            minimax_node = move
            minimax_color = next_color
            minimax_pv = pv

    minimax_pv.append(('NO', minimax_node, turn_color, minimax_score))
    return minimax_pv


def test():
    b = Board()
    print('-'*80)
    print(b.pretty())
    b.pick(10)
    print('-'*80)
    print(b.pretty())
    color = Color.black
    print(generate_legal_moves([b], color))
    print('-'*80)
    print(b.move(Move(4), Color.white))
    print(b.pretty())
    print('eval = ', evaluate_board(b, Color.black))
    print('-'*80)
    b.finalize_game()
    print('-'*80)
    print(b.pretty())

def test_search():
    stack = [Board()]
    info = SearchInfo()
    pv = search_dfs(stack, Color.black, Color.black, 4, 4, info)
    print(pv)

def test_play(args, white_is_human = False):
    stack = [Board()]
    color = Color.black
    max_depth = args.depth
    max_turn = 100
    for i in range(max_turn):
        print('-- # {:2d} --------------------------------'.format(i))
        print('')
        print(stack[-1].pretty())
        print('')
        b = stack[-1].data[stack[-1].black_kalah]
        w = stack[-1].data[stack[-1].white_kalah]
        allb, allw = stack[-1].tentative_stones_balance()
        print('BLACK = {}/{} WHITE = {}/{}'.format(b, allb, w, allw))

        moves = generate_legal_moves(stack, color)
        if len(moves) == 0:
            break

        if white_is_human and color == Color.white:
            print('Your turn ({}).'.format(color))
            moves = generate_legal_moves(stack, color)
            if len(moves) == 0:
                break
            index = -1
            while not stack[-1].moveable(index, color):
                ch = six.input('Enter move:').strip()
                try:
                    index = int(ch)
                except ValueError:
                    continue
            move = Move(index)
            print('move = {}'.format(move))
        else:
            print('TURN = {} thinking..'.format(color))
            info = SearchInfo()
            pv = search_dfs(stack, color, color, max_depth, max_depth, info)
            if len(pv) > 0:
                _, move, _, score = pv[-1]
                if move is None:
                    break
            print('move = {} score = {} (examined {} nodes)'.format(move, score, info.nodes))

        next_board = copy.deepcopy(stack[-1])
        events = []
        next_color = next_board.move(move, color, events)
        stack.append(next_board)
        if 'gain-turn' in events:
            print('{} gained another turn!!'.format(color))
        if 'gather' in events:
            print('{} gathered stones!!'.format(color))

        color = next_color

    print('='*80)
    print('Game end.')
    stack[-1].finalize_game()
    b, w = stack[-1].tentative_stones_balance()
    print(stack[-1].pretty())
    if b > w:
        print('Black won.')
    elif b < w:
        print('White won.')
    else:
        print('draw.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', dest='mode',
            action='store_const', const='test',
            help=u'set test mode')
    parser.add_argument('--search', dest='mode',
            action='store_const', const='search',
            help=u'set search mode')
    parser.add_argument('--self-play', dest='mode',
            action='store_const', const='self', default='human',
            help=u'set self play mode')
    parser.add_argument('--depth', type=int, default=4,
            help=u'max search depth')
    args = parser.parse_args()

    if args.mode == 'test':
        test()
    elif args.mode == 'search':
        test_search()
    elif args.mode == 'self':
        test_play(args)
    else:
        test_play(args, white_is_human=True)


