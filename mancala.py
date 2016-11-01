#!env python
# -*- coding: utf-8 -*-
# Mancala solver for Python 2.7 and 3.5
import six
import sys
import argparse
import os
import time
import numpy as np

class Board(object):
    BLACK    = 1 << 0
    WHITE    = 1 << 1
    HOLE     = 1 << 2
    KALAH    = 1 << 3
    def __init__(self, init_stones = 4, size = 6):
        # <-
        # |0|[1][2][3][4][5][6]| |  : white
        # | |[d][c][b][a][9][8]|7|  : black
        # ->
        self.size = size
        self.n_pockets = size * 2 + 2
        self.init_stones = init_stones

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

    def holes(self, color):
        return self.black_holes if color == Color.black else self.white_holes


class Position(object):
    u'''minimal information to represent the status of the game.'''
    def __init__(self, board, data = None):
        assert isinstance(board, Board)
        self.board = board
        if data is None:
            self.data = np.zeros(self.board.n_pockets, np.int32)
            self.data[self.board.black_holes] = self.board.init_stones
            self.data[self.board.white_holes] = self.board.init_stones
        else:
            self.data = data

    def clone(self):
        res = Position(self.board, np.array(self.data))
        return res

    def finalize_game(self):
        # collect all stones in the black(white) side and put them into their kalah.
        for index in self.board.black_holes:
            self.data[self.board.black_kalah] += self.data[index]
            self.data[index] = 0
        for index in self.board.white_holes:
            self.data[self.board.white_kalah] += self.data[index]
            self.data[index] = 0

    def _drop(self, from_index, n_stones):
        while n_stones > 0:
            self.data[from_index] += 1
            n_stones -= 1
            from_index = self.board.next_hole[from_index]
        last_drop_index = self.board.prev_hole[from_index]
        return last_drop_index

    def _pick(self, index):
        n_stones = self.data[index]
        self.data[index] = 0
        last_drop_index = self._drop(self.board.next_hole[index], n_stones)
        return last_drop_index

    def moveable(self, move, color):
        index = move.index
        if color == Color.black:
            return (index in self.board.black_holes) and self.data[index] > 0
        if color == Color.white:
            return (index in self.board.white_holes) and self.data[index] > 0
        return False

    def move(self, move, color, events = []):
        assert self.moveable(move, color)
        index = move.index
        # pick
        last_drop_index = self._pick(index)
        # gather
        if self.data[last_drop_index] == 1:
            if color == Color.black and last_drop_index in self.board.black_holes:
                j = self.board.opposite_holes[last_drop_index]
                if self.data[j] > 0:
                    self.data[self.board.black_kalah] += self.data[j] + self.data[last_drop_index]
                    self.data[j] = 0
                    self.data[last_drop_index] = 0
                    events.append('gather')
            elif color == Color.white and last_drop_index in self.board.white_holes:
                j = self.board.opposite_holes[last_drop_index]
                if self.data[j] > 0:
                    self.data[self.board.white_kalah] += self.data[j] + self.data[last_drop_index]
                    self.data[j] = 0
                    self.data[last_drop_index] = 0
                    events.append('gather')
        # gain-turn
        if color == Color.black and last_drop_index == self.board.black_kalah:
            events.append('gain-turn')
            return Color.black
        if color == Color.white and last_drop_index == self.board.white_kalah:
            events.append('gain-turn')
            return Color.white
        # normal turn end
        return - color

    def tentative_stones_balance(self):
        # (black stones, white stones)
        return self.data[self.board.black_holes_and_kalah].sum(), self.data[self.board.white_holes_and_kalah].sum()

    def pretty(self):
        size = self.board.size
        return u'\n'.join([
            u'   ' + u''.join('#{:<2d} '.format(i) for i in range(size + 1)),
            u'W |{:2d}|'.format(self.data[0]) + u''.join('[{:2d}]'.format(i) for i in self.data[1:size + 1]) + u'|  |',
            u'B |  |' + u''.join('[{:2d}]'.format(i) for i in self.data[size + 2:][::-1]) + u'|{:2d}|'.format(self.data[size + 1]),
            u'      ' + u''.join('#{:<2d} '.format(i) for i in range(size * 2 + 2 - 1, size, -1)),
            ])


def evaluate_pos_kalah(pos, color):
    # evaluate the Position using only stones in the kalah.
    b = pos.data[pos.board.black_kalah]
    w = pos.data[pos.board.white_kalah]

    if color == Color.black:
        return b - w
    else:
        return w - b

def evaluate_pos_all(pos, color):
    # evaluate the Position using all stones.
    b, w = pos.tentative_stones_balance()

    if color == Color.black:
        return b - w
    else:
        return w - b

evaluate_pos = evaluate_pos_kalah

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

def generate_legal_moves(pos, color):
    moves = [Move(i) for i in pos.board.holes(color)]
    return [m for m in moves if pos.moveable(m, color)]

class SearchInfo(object):
    def __init__(self):
        self.nodes = 0
        self.start_time = time.time()
    def stop(self):
        self.stop_time = time.time()
        self.elapsed_s = self.stop_time - self.start_time
        self.nps = self.nodes / self.elapsed_s if self.elapsed_s else 0.0

def search_dfs(stack, root_color, turn_color, max_depth, depth, search_info):
    # return (move, score, PV) where PV := [(leaf status, color, move, score)]
    search_info.nodes += 1

    pos = stack[-1]
    moves = generate_legal_moves(pos, turn_color)
    if len(moves) == 0:
        # no more move possible. game end.
        p = pos.clone()
        p.finalize_game()
        score = evaluate_pos(p, root_color)
        return None, score, [('NL', None, turn_color, score)]

    if depth == 0:
        # static evaluate
        score = evaluate_pos(pos, root_color)
        return None, score, [('LE', None, turn_color, score)]

    # search next depth
    minimax_score = None
    minimax_move = None
    minimax_color = None
    minimax_pv = []
    for move in moves:
        stack.append(pos.clone())
        next_color = stack[-1].move(move, turn_color)
        _, score, pv = search_dfs(stack, root_color, next_color, max_depth, depth - 1, search_info)
        stack.pop(-1) # unmove.

        if turn_color == root_color:
            do_update = minimax_score is None or score > minimax_score
        else:
            do_update = minimax_score is None or score < minimax_score

        if do_update:
            minimax_score = score
            minimax_move = move
            minimax_color = next_color
            minimax_pv = pv

    minimax_pv.append(('NO', minimax_move, turn_color, minimax_score))
    return minimax_move, minimax_score, minimax_pv


def test():
    b = Board()
    p = Position(b)
    print('-'*80)
    print(p.pretty())
    p._pick(10)
    print('-'*80)
    print(p.pretty())
    color = Color.black
    print(generate_legal_moves(p, color))
    print('-'*80)
    print(p.move(Move(4), Color.white))
    print(p.pretty())
    print('eval = ', evaluate_pos(p, Color.black))
    print('-'*80)
    p.finalize_game()
    print('-'*80)
    print(p.pretty())

def test_search():
    stack = [Position(Board())]
    info = SearchInfo()
    move, score, pv = search_dfs(stack, Color.black, Color.black, 4, 4, info)
    print(pv)

def test_play(args, white_is_human = False):
    board = Board()
    pos = Position(board)
    color = Color.black
    max_depth = args.depth
    max_turn = 100
    for i in range(max_turn):
        print('-- # {:2d} --------------------------------'.format(i))
        print('')
        print(pos.pretty())
        print('')
        b = pos.data[board.black_kalah]
        w = pos.data[board.white_kalah]
        allb, allw = pos.tentative_stones_balance()
        print('BLACK = {}/{} WHITE = {}/{}'.format(b, allb, w, allw))

        moves = generate_legal_moves(pos, color)
        if len(moves) == 0:
            break

        if white_is_human and color == Color.white:
            print('Your turn ({}).'.format(color))
            moves = generate_legal_moves(pos, color)
            if len(moves) == 0:
                break
            move = None
            while move is None or not pos.moveable(move, color):
                ch = six.moves.input('Enter move:').strip()
                try:
                    move = Move(int(ch))
                except ValueError:
                    continue
            print('move = {}'.format(move))
        else:
            print('TURN = {} thinking..'.format(color))
            info = SearchInfo()
            stack = [pos]
            move, score, pv = search_dfs(stack, color, color, max_depth, max_depth, info)
            info.stop()
            if move is None:
                break
            print('move = {} score = {} (examined {} nodes, {:.2f}s, NPS={:.2f})'.format(move, score, info.nodes, info.elapsed_s, info.nps))

        next_pos = pos.clone()
        events = []
        next_color = next_pos.move(move, color, events)
        if 'gain-turn' in events:
            print('{} gained another turn!!'.format(color))
        if 'gather' in events:
            print('{} gathered stones!!'.format(color))

        color = next_color
        pos = next_pos

    print('='*80)
    print('Game end.')
    pos.finalize_game()
    b, w = pos.tentative_stones_balance()
    print(pos.pretty())
    if b > w:
        print('Black won.')
    elif b < w:
        print('White won.')
    else:
        print('draw.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # mode selection
    parser.add_argument('--test', dest='mode',
            action='store_const', const='test',
            help=u'set test mode')
    parser.add_argument('--search', dest='mode',
            action='store_const', const='search',
            help=u'set search mode')
    parser.add_argument('--self-play', dest='mode',
            action='store_const', const='self', default='human',
            help=u'set self play mode')
    # common options
    parser.add_argument('--depth', type=int, default=4,
            help=u'max search depth')
    parser.add_argument('--eval-kalah', dest='eval',
            action='store_const', const='kalah',
            help=u'evaluate the position using only stones in the kalah')
    parser.add_argument('--eval-all', dest='eval',
            action='store_const', const='all', default='kalah',
            help=u'evaluate the position using all stones in the player\'s side')
    args = parser.parse_args()

    # select evaluation method.
    global evaluate_pos
    if args.eval == 'kalah':
        evaluate_pos = evaluate_pos_kalah
    elif args.eval == 'all':
        evaluate_pos = evaluate_pos_all

    if args.mode == 'test':
        test()
    elif args.mode == 'search':
        test_search()
    elif args.mode == 'self':
        test_play(args)
    else:
        test_play(args, white_is_human=True)


