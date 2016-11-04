# mancala_test

this is a simple computer-mancala software written in Python.

## Features

- CUI computer vs. human play
- computer self play
- full Python
- search algorithms:
 - minimax
 - negamax
 - alpha-beta

## Mancala game

### Kalah

- 2 players.
- **6** holes and 1 kalah on each side. use --holes to change the number of holes.
- initially each hole has **4** stones. use --stones to change the number of stones.
- stowing: in each turn, the current player selects a non-empty hole in the player's side. pick all stones in it and drop one by one to the counter clockwise hole (or kalah)
- dropping the last stone in the current player's empty hole, all stones in the opposite hole is gathered to the current player's kalah.
- dropping the last stone in the current player's kalah, the player gains another turn.
- if all holes of the current player's side are empty, the game ends.
- on the game end, remaining stones in each players' holes are put into their own kalah.
- finally, the player who earned the larger number of stones in her kalah wins the game.

## Prerequisites

- Python 3
- numpy


# Usage

## Computer Self Play

basic:
> python mancala.py --self

with additional options:
> python mancala.py --self --depth 5 --search-algorithm alpha-beta --holes 7 --stones 5

## CUI Computer vs. Human Play

basic:
> python mancala.py

with additional options:
> python mancala.py --depth 5 --search-algorithm alpha-beta

# Self Play Demo
```
> python mancala.py --depth 5 --self  --search-algorithm alpha-beta
-- #  0 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W | 0|[ 4][ 4][ 4][ 4][ 4][ 4]|  |
B |  |[ 4][ 4][ 4][ 4][ 4][ 4]| 0|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 0/24 WHITE = 0/24
TURN = B thinking..
move = m11 score = 0 (examined 2752 nodes, 0.11s, NPS=25702.30)
B gained another turn!!
-- #  1 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W | 0|[ 4][ 4][ 4][ 4][ 4][ 4]|  |
B |  |[ 4][ 4][ 0][ 5][ 5][ 5]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/24 WHITE = 0/24
TURN = B thinking..
move = m12 score = 0 (examined 2352 nodes, 0.09s, NPS=25828.67)
-- #  2 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W | 0|[ 4][ 4][ 4][ 4][ 4][ 4]|  |
B |  |[ 4][ 0][ 1][ 6][ 6][ 6]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/24 WHITE = 0/24
TURN = W thinking..
move = m4 score = 0 (examined 3154 nodes, 0.12s, NPS=25625.36)
W gained another turn!!
-- #  3 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W | 1|[ 5][ 5][ 5][ 0][ 4][ 4]|  |
B |  |[ 4][ 0][ 1][ 6][ 6][ 6]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/24 WHITE = 1/24
TURN = W thinking..
move = m5 score = 0 (examined 2407 nodes, 0.09s, NPS=25591.49)
-- #  4 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W | 1|[ 6][ 6][ 6][ 1][ 0][ 4]|  |
B |  |[ 4][ 0][ 1][ 6][ 6][ 6]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/24 WHITE = 1/24
TURN = B thinking..
move = m13 score = 8 (examined 2463 nodes, 0.10s, NPS=25640.85)
-- #  5 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W | 1|[ 6][ 6][ 6][ 1][ 0][ 4]|  |
B |  |[ 0][ 1][ 2][ 7][ 7][ 6]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/24 WHITE = 1/24
TURN = W thinking..
move = m6 score = 0 (examined 2418 nodes, 0.09s, NPS=26265.18)
-- #  6 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W | 1|[ 6][ 7][ 7][ 2][ 1][ 0]|  |
B |  |[ 0][ 1][ 2][ 7][ 7][ 6]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/24 WHITE = 1/24
TURN = B thinking..
move = m11 score = 0 (examined 1857 nodes, 0.07s, NPS=25774.59)
-- #  7 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W | 1|[ 6][ 7][ 7][ 2][ 1][ 0]|  |
B |  |[ 0][ 1][ 0][ 8][ 8][ 6]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/24 WHITE = 1/24
TURN = W thinking..
move = m2 score = 0 (examined 1667 nodes, 0.06s, NPS=26236.63)
-- #  8 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W | 2|[ 7][ 0][ 7][ 2][ 1][ 0]|  |
B |  |[ 1][ 2][ 1][ 9][ 9][ 6]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/29 WHITE = 2/19
TURN = B thinking..
move = m11 score = 2 (examined 1801 nodes, 0.07s, NPS=25529.13)
-- #  9 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W | 2|[ 7][ 0][ 7][ 2][ 1][ 0]|  |
B |  |[ 1][ 2][ 0][10][ 9][ 6]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/29 WHITE = 2/19
TURN = W thinking..
move = m4 score = -6 (examined 1323 nodes, 0.05s, NPS=25927.42)
W gathered stones!!
-- # 10 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W | 5|[ 7][ 0][ 8][ 0][ 1][ 0]|  |
B |  |[ 1][ 0][ 0][10][ 9][ 6]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/27 WHITE = 5/21
TURN = B thinking..
move = m13 score = 2 (examined 1003 nodes, 0.04s, NPS=25700.33)
-- # 11 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W | 5|[ 7][ 0][ 8][ 0][ 1][ 0]|  |
B |  |[ 0][ 1][ 0][10][ 9][ 6]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/27 WHITE = 5/21
TURN = W thinking..
move = m1 score = -6 (examined 842 nodes, 0.03s, NPS=26295.79)
-- # 12 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W | 6|[ 0][ 0][ 8][ 0][ 1][ 0]|  |
B |  |[ 1][ 2][ 1][11][10][ 7]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/33 WHITE = 6/15
TURN = B thinking..
move = m13 score = 6 (examined 738 nodes, 0.03s, NPS=26819.94)
-- # 13 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W | 6|[ 0][ 0][ 8][ 0][ 1][ 0]|  |
B |  |[ 0][ 3][ 1][11][10][ 7]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/33 WHITE = 6/15
TURN = W thinking..
move = m5 score = -6 (examined 387 nodes, 0.01s, NPS=26669.66)
W gathered stones!!
-- # 14 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W |18|[ 0][ 0][ 8][ 0][ 0][ 0]|  |
B |  |[ 0][ 3][ 1][ 0][10][ 7]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/22 WHITE = 18/26
TURN = B thinking..
move = m11 score = 6 (examined 338 nodes, 0.01s, NPS=27012.08)
-- # 15 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W |18|[ 0][ 0][ 8][ 0][ 0][ 0]|  |
B |  |[ 0][ 3][ 0][ 1][10][ 7]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/22 WHITE = 18/26
TURN = W thinking..
move = m3 score = -6 (examined 171 nodes, 0.01s, NPS=26295.13)
-- # 16 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W |19|[ 1][ 1][ 0][ 0][ 0][ 0]|  |
B |  |[ 1][ 4][ 1][ 2][11][ 7]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/27 WHITE = 19/21
TURN = B thinking..
move = m13 score = 6 (examined 456 nodes, 0.02s, NPS=27617.22)
-- # 17 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W |19|[ 1][ 1][ 0][ 0][ 0][ 0]|  |
B |  |[ 0][ 5][ 1][ 2][11][ 7]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/27 WHITE = 19/21
TURN = W thinking..
move = m1 score = -6 (examined 184 nodes, 0.01s, NPS=26266.15)
W gained another turn!!
-- # 18 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W |20|[ 0][ 1][ 0][ 0][ 0][ 0]|  |
B |  |[ 0][ 5][ 1][ 2][11][ 7]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/27 WHITE = 20/21
TURN = W thinking..
move = m2 score = -6 (examined 139 nodes, 0.01s, NPS=27780.82)
-- # 19 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W |20|[ 1][ 0][ 0][ 0][ 0][ 0]|  |
B |  |[ 0][ 5][ 1][ 2][11][ 7]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/27 WHITE = 20/21
TURN = B thinking..
move = m10 score = 6 (examined 359 nodes, 0.01s, NPS=27595.62)
-- # 20 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W |20|[ 1][ 0][ 0][ 0][ 0][ 0]|  |
B |  |[ 0][ 5][ 1][ 0][12][ 8]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/27 WHITE = 20/21
TURN = W thinking..
move = m1 score = -6 (examined 2 nodes, 0.00s, NPS=0.00)
W gained another turn!!
-- # 21 --------------------------------

   #0  #1  #2  #3  #4  #5  #6
W |21|[ 0][ 0][ 0][ 0][ 0][ 0]|  |
B |  |[ 0][ 5][ 1][ 0][12][ 8]| 1|
      #13 #12 #11 #10 #9  #8  #7

BLACK = 1/27 WHITE = 21/21
================================================================================
Game end. (total nodes = 26813)
   #0  #1  #2  #3  #4  #5  #6
W |21|[ 0][ 0][ 0][ 0][ 0][ 0]|  |
B |  |[ 0][ 0][ 0][ 0][ 0][ 0]|27|
      #13 #12 #11 #10 #9  #8  #7
Black won.
Search algorithm used: alpha-beta
```