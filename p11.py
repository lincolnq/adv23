samp = """...#......
.......#..
#.........
..........
......#...
.#........
.........#
..........
.......#..
#...#....."""

input = open('p11inp.txt').read()

from helpers import *
import numpy as np

expansion_factor = 1_000_000

parsed = Matrix().parse(input)

galaxies_locs = np.argwhere(parsed == '#')
all_rows = set(range(parsed.shape[0]))
occupied_rows = set(galaxies_locs[:,0])
expansion_rows = all_rows - occupied_rows

all_cols = set(range(parsed.shape[1]))
occupied_cols = set(galaxies_locs[:,1])
expansion_cols = all_cols - occupied_cols

def dist(loca, locb):
    # Manhattan distance, except that we count expansion rows and cols double
    rowmin, rowmax = min(loca[0], locb[0]), max(loca[0], locb[0])
    colmin, colmax = min(loca[1], locb[1]), max(loca[1], locb[1])
    
    rowdist = rowmax - rowmin
    for erow in expansion_rows:
        if rowmin <= erow < rowmax:
            rowdist += expansion_factor - 1

    coldist = colmax - colmin
    for ecol in expansion_cols:
        if colmin <= ecol < colmax:
            coldist += expansion_factor - 1
    
    return rowdist + coldist

def all_pairs_dist():
    gcount = len(galaxies_locs)
    result = 0
    for i in range(gcount):
        for j in range(i+1, gcount):
            result += dist(tuple(galaxies_locs[i]), tuple(galaxies_locs[j]))

    return result

print(f"dist is {all_pairs_dist()}")