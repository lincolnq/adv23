samp = """O....#....
O.OO#....#
.....##...
OO.#O....O
.O.....O#.
O.#..O.#.#
..O..#O..O
.......O..
#....###..
#OO..#...."""

import numpy as np
from helpers import *

parser = Matrix()
real = open('p14x.txt').read()
grid = parser.parse(real)


def showgrid(grid):
    return '\n'.join([''.join(row) for row in grid])

def slide1(grid):
    """Slides all 'O' rocks in the grid upwards one step 
    unless it hits a #, another rock, or the edge of the grid."""

    grid = grid.copy()

    # we'll do it one row at a time, one step at a time
    for i in range(0, grid.shape[0]):
        for j in range(0, grid.shape[1]):
            if grid[i,j] == 'O':
                # if we can slide up, do so
                if i > 0 and grid[i-1,j] == '.':
                    grid[i-1,j] = 'O'
                    grid[i,j] = '.'
    return grid

def slide(grid):
    while True:
        newgrid = slide1(grid)
        if (newgrid == grid).all():
            return newgrid
        grid = newgrid

def spin(grid):
    """Slide north, then west, then south, then east."""
    # 1 2
    # 3 4
    # transpose, slide up
    # 1 3
    # 2 4
    # [[transpose back
    # 1 2
    # 3 4]]

    # now invert rows, slide up
    # 3 4
    # 1 2

    # now transpose and invert rows, slide up
    # 4 2 
    # 3 1

    # and to recover original grid: invert rows, invert cols, transpose
    # 3 1
    # 4 2

    # 1 3
    # 2 4

    # 1 2
    # 3 4

    spins = 0
    seen_grids = {}


    while True:
        newgrid = slide(grid)
        newgrid = slide(newgrid.T).T
        newgrid = slide(newgrid[::-1])
        newgrid = slide(newgrid.T[::-1])[::-1, ::-1].T

        spins += 1

        gridstr = showgrid(newgrid)
        if gridstr in seen_grids:
            prevspins = seen_grids[gridstr]
            print(f"seen grid @ {spins} before @ {prevspins}!")
            return seen_grids
        else:
            seen_grids[gridstr] = spins

        #print(f"-- after {spins} spins: --")
        #print(showgrid(newgrid))
        #if (newgrid == grid).all():
        #    return newgrid
        
        grid = newgrid

def load(grid):
    return np.sum((grid == 'O') * np.arange(grid.shape[0], 0, -1).reshape(-1, 1))

print(showgrid(grid))
print("--")
# i computed this manually, after noticing that the grid was repeating after 115 spins
# every 22 spins
mygridid = (1000000000 - 115) % 22 + 115

# ok spin it
gridz = spin(grid)

mygridstr = [x for (x,c) in gridz.items() if c == mygridid][0]

print(mygridstr)

print(load(parser.parse(mygridstr)))