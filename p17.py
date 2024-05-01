samp = """2413432311323
3215453535623
3255245654254
3446585845452
4546657867536
1438598798454
4457876987766
3637877979653
4654967986887
4564679986453
1224686865563
2546548887735
4322674655533"""

samp2 = """111111111111
999999999991
999999999991
999999999991
999999999991"""

import numpy as np
import heapq
from dataclasses import dataclass
from helpers import *


parser = IntMatrix()
real = open('p17inp.txt').read()
grid = parser.parse(real)

def showgrid(grid):
    return '\n'.join([''.join(str(x) for x in row) for row in grid])

print(showgrid(grid))


MAX_SEQ_STEPS = 3

def weighted_shortest_path(g, unconstrained_sps):
    start = (0,0)
    end = (g.shape[0]-1, g.shape[1]-1)

    # node class
    @dataclass
    class Node:
        pos: tuple
        path: list[tuple]
        sequential_steps_origin: tuple
        cost: int

        def heuristic(self):
            # use unconstrained sps
            return unconstrained_sps[self.pos]
            #return abs(self.pos[0] - end[0]) + abs(self.pos[1] - end[1])

        def __lt__(self, other):
            return self.cost + self.heuristic() < other.cost + other.heuristic()

    # we'll use a priority queue to keep track of the next node to visit
    pq = []
    heapq.heappush(pq, Node(start, [], start, 0))

    steps = 0

    while pq:
        current = heapq.heappop(pq)
        if current.pos == end:
            return current.cost
        
        steps += 1
        if steps % 10000 == 0:
            print(f"examining {current.pos}, steps={len(current.path)}, cost={current.cost} after { steps } steps")
            g2 = g.copy().astype(str)
            for p in current.path:
                g2[p] = 'X'

            #print(showgrid(g2))
            #break
        
        # try all neighbors
        for d in [(0,1), (0,-1), (1,0), (-1,0)]:
            newpos = (current.pos[0] + d[0], current.pos[1] + d[1])

            # can't go backwards
            if newpos in current.path:
                continue

            # check bounds
            if not (0 <= newpos[0] < g.shape[0] and 0 <= newpos[1] < g.shape[1]):
                continue

            # check sequential steps in one direction
            seqr = abs(newpos[0] - current.sequential_steps_origin[0])
            seqc = abs(newpos[1] - current.sequential_steps_origin[1])
            if seqr != 0 and seqc != 0:
                # new direction
                seq_origin = current.pos
            elif seqr == 0 and seqc == 0:
                # stepping back onto origin square, skip neighbor
                continue
            elif (seqr == 0 and seqc <= MAX_SEQ_STEPS) or (seqc == 0 and seqr <= MAX_SEQ_STEPS):
                # same row/col, fewer than maximum steps in one direction
                seq_origin = current.sequential_steps_origin
            else:
                # too many steps in one direction, skip neighbor
                continue

            newcost = current.cost + g[newpos]
            heapq.heappush(pq, Node(newpos, current.path + [current.pos], seq_origin, newcost))


# ok let's start with DP to compute the shortest path to the end from every cell
# (disregarding the sequential steps constraint).
# we'll start at the end and work backwards
def compute_sp(g):
    sp = np.zeros_like(g)
    sp[-1,-1] = g[-1,-1]
    for i in range(g.shape[0]-2, -1, -1):
        sp[i, -1] = sp[i+1, -1] + g[i, -1]
    for j in range(g.shape[1]-2, -1, -1):
        sp[-1, j] = sp[-1, j+1] + g[-1, j]

    for i in range(g.shape[0]-2, -1, -1):
        for j in range(g.shape[1]-2, -1, -1):
            sp[i,j] = g[i,j] + min(sp[i+1,j], sp[i,j+1])

    return sp

def turn(dir):
    """Return both 90 degree turn directions."""
    if dir[0] == 0:
        return [(1,0), (-1,0)]
    else:
        return [(0,1), (0,-1)]

# ok now similar to above, but compute using the sequential steps constraint
# The way we implement this is that we "always turn": the neighbors are 1-3 steps in
# a direction, then you always turn left or right.
def compute_sps_constrained(g):

    # Dynamic programming to compute four 'dir arrays': if you land on this cell
    # and are facing this direction, the cell holds the length of the shortest 
    # path to the end cell.
    dirs = [(0,1), (0,-1), (1,0), (-1,0)]
    arr = np.zeros_like(g)
    arr.fill(1_000)

    # No matter which way you are facing, when you land at the end cell, the shortest
    # path is just the value of the end cell. (AFAIK there is no way to land on the 
    # end cell facing N or W, but whatever)
    arr[-1,-1] = g[-1,-1]

    dirarrays = {d: arr.copy() for d in dirs}
    
    prev_dirarrays = {d: dirarrays[d].copy() for d in dirs}

    iters = 0
    while True:

        # compute the whole grid
        for i in range(g.shape[0]-1, -1, -1):
            for j in range(g.shape[1]-1, -1, -1):
                # "When you land on this cell..."

                for dir in dirs:
                    # "...facing this direction..."
                    ourcost = dirarrays[dir][i,j]

                    # 'dir' indicates the way you are facing when you land on the cell
                    # which means we know you walked in that direction.

                    # Discover a new route to each neighboring cell in direction.
                    total_neighbor_cost = 0
                    for offset in range(1, 11):
                        neighbor = i - dir[0] * offset, j - dir[1] * offset
                        if not (0 <= neighbor[0] < g.shape[0] and 0 <= neighbor[1] < g.shape[1]):
                            # skip out of bounds
                            continue
                        total_neighbor_cost += g[neighbor]

                        if offset < 4:
                            # we haven't reached the minimum number of steps in one direction
                            continue

                        # cost via this path of getting from this neighbor cell to end
                        # will be our cost so far
                        # plus the cost of the cells between us and the neighbor
                        cost_this_path = ourcost + total_neighbor_cost

                        # fill it in if it's better into both turndirs' arrays
                        for turndir in turn(dir):
                            dirarrays[turndir][neighbor] = min(dirarrays[turndir][neighbor], cost_this_path)

        if all((dirarrays[d] == prev_dirarrays[d]).all() for d in dirs):
            print("converged")
            break
        else:
            iters += 1
            print(f"doing another iter: {iters}")
            prev_dirarrays = {d: dirarrays[d].copy() for d in dirs}

    
    return dirarrays
                

uncon = compute_sp(grid)
print(uncon)

constrained = compute_sps_constrained(grid)
best = 999_999
for d in constrained:
    res = constrained[d][0,0] - grid[0,0]
    if res < best:
        best = res
    
print(f"best is {best}")

#print(weighted_shortest_path(grid, uncon))

# 851 or 852 is what the system outputs