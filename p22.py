samp = """1,0,1~1,2,1
0,0,2~2,0,2
0,2,3~2,2,3
0,0,4~0,2,4
2,0,5~2,2,5
0,1,6~2,1,6
1,1,8~1,1,9"""

samp2 = """1,0,2~1,0,3
2,0,2~2,0,2
3,0,2~3,0,2
4,0,2~4,0,2
2,0,3~4,0,3
3,0,4~3,0,4
2,0,5~4,0,5
2,0,6~2,0,8
1,0,4~2,0,4
"""


samp3 = """1,0,1~1,0,1
2,0,1~2,0,1
3,0,1~3,0,1
4,0,1~4,0,1
5,0,1~5,0,1
6,0,1~6,0,1
1,0,2~2,0,2
3,0,2~4,0,2
5,0,2~6,0,2
2,0,3~2,0,3
4,0,3~5,0,3
1,0,4~2,0,4
3,0,4~4,0,4
5,0,4~6,0,4
"""

samp4 = """1,0,1~1,0,1
1,0,2~1,0,2
1,0,3~1,0,3
3,0,1~3,0,1
3,0,2~3,0,2
3,0,3~3,0,3
1,0,4~3,0,4
"""

from helpers import *
import numpy as np

parser = Lines() ** Split('~') ** Split(',') ** int

#input = parser.parse(samp4)
#GRIDHEIGHT, GRIDSIZE = 10, 7

input = parser.parse(open('p22inp.txt').read())
GRIDHEIGHT, GRIDSIZE = 1000, 10

BRICKS = len(input)

#print(input)

# step 1 is to cause bricks to fall. they come in a random order.
# step 2 is to make a graph of which bricks are dependent on which others.

# we'll fill in 'brick id' (line number) as the int value of the brick
grid = np.zeros((GRIDHEIGHT, GRIDSIZE, GRIDSIZE), dtype=int)

def load_bricks(grid, input):
    for lineno, ((x0, y0, z0), (x1, y1, z1)) in enumerate(input):
        print(f"loading brick {lineno+1} from {x0, y0, z0} to {x1, y1, z1}")
        assert np.all(grid[z0:z1+1, y0:y1+1, x0:x1+1] == 0), "Overlapping bricks!"
        grid[z0:z1+1, y0:y1+1, x0:x1+1] = lineno+1

def settle_map(grid) -> np.ndarray:
    # create an adjacency matrix (above, below): which bricks are directly atop 
    # which other bricks by ID?
    # Having 0 as a 'below' means the given brick is 'atop' layer 0 and can't fall.
    
    # This is used both for falling bricks, and figuring out which
    # bricks to disintegrate.
    
    settle_map = np.zeros((BRICKS+1, BRICKS+1), dtype=bool)

    lastlayer = np.zeros((GRIDSIZE, GRIDSIZE), dtype=int)
    firstlayer = grid[1]
    firstlayeradj = set(firstlayer[(firstlayer != 0)])

    for ab in firstlayeradj:
        settle_map[ab, 0] = True

    lastlayer = firstlayer

    for z in range(2, GRIDHEIGHT):
        # adjacencies are any bricks in this layer who are different
        # from the bricks in the prior layer
        layer = grid[z]
        adj = (layer != lastlayer) & (layer != 0) & (lastlayer != 0)
        ixs = adj.nonzero()
        settle_map[layer[ixs], lastlayer[ixs]] = True
        # for ix in np.argwhere(adj):
        #     ab = layer[ix]
        #     bb = lastlayer[ix]
        #     breakpoint()
        #     settle_map[ab, bb] = True
        #     assert not settle_map[bb, ab], "Found a reciprocal adjacency!"
        #         #print(f"adjacency on layer {z}: {ab} atop {bb}")
        lastlayer = layer

    return settle_map


load_bricks(grid, input)
# we never use grid level 0, so render it starting from 1, 
# and also render it upside-down so as to match the example
print(grid[-1:0:-1])

#print(sm)

# now search up all connected components in the grid
def findcc(adjmat, start) -> set:
    """Use adjacency matrix to find a connected component (set of all block indices),
    starting at block index 'start'."""
    
    cc = np.zeros(adjmat.shape[0], dtype=bool)
    cc[start] = True
    prev = cc

    while True:
        # get all blocks atop any block in 'prev'
        #breakpoint()
        next = adjmat[:, prev].T.any(axis=0)
        if not next.any():
            break
        cc = (cc | next)
        prev = next
    return set(cc.nonzero()[0])

def drop1(grid, brickids):
    """Drop all elements in the set of brickids by one layer."""

    gridbools = numpy.isin(grid, list(brickids))
    gridixs = np.where(gridbools)
    gridvals = grid[gridixs]
    
    grid[gridixs] = 0
    gridixs_new = (gridixs[0] - 1, gridixs[1], gridixs[2])
    if grid[gridixs_new].sum() != 0:
        breakpoint()
        #assert  == 0, "There's a brick in the way!"
    grid[gridixs_new] = gridvals


def dropall(grid):
    """Drop all non-grounded bricks until they stop"""
    allbricks = set(range(1, BRICKS+1))

    sm = settle_map(grid)
    grounded = findcc(sm, 0)

    nongrounded = allbricks - grounded
    while nongrounded:
        print(f"dropping {len(nongrounded)}, gridsum is now {grid.sum()}")
        drop1(grid, nongrounded)

        sm = settle_map(grid)
        grounded = findcc(sm, 0)
        nongrounded = allbricks - grounded

print("After dropall:")
dropall(grid)
print(grid)

EXCLUDERS = ~np.eye(BRICKS + 1, dtype=bool)

def sibs(sm, brick):
    """Returns sibling bricks: other bricks that are holding up at least some of the 
    same stuff as the given brick id.
    
    Returns 2d boolean array: one row for each brick atop us; each column is a brick id.
    """
    atop = sm[:, brick]
    sibs = sm[atop] & EXCLUDERS[brick]
    return sibs

# safe to disintegrate bricks are the ones who have no bricks atop them,
# or if they do have any bricks, each of those bricks have other bricks that they are atop.
def safe_bricks(sm):
    safe = set()
    for i in range(1, BRICKS+1):
        if not sm[:, i].any():
            #print(f"{i} is safe because nothing is atop it")
            safe.add(i)
        else:
            sibixs = sibs(sm, i)
            #print(f"brick {i}: {adj}")
            # We need a true in every row of 'adj' to be ok
            if sibixs.any(axis=1).all():
                #print(f"{i} is safe because all bricks atop it have other bricks they are atop")
                safe.add(i)
            
    return safe

sm = settle_map(grid)
sb = safe_bricks(sm)
print(len(sb))

allbricks = set(range(1, BRICKS+1))

def unsafe_above(sm, ccs_ex_self, brick):
    """For a given unsafe brick, figure out how many bricks will fall if this one
    goes away. This is equivalent to findcc(sm, brick) minus itself, minus
    findcc of this brick's 'siblings'. (Siblings are bricks that co-support
    something alongside us.)"""

    sibbricks = sibs(sm, brick).any(axis=0).nonzero()[0]
    sibbrickccs = set()
    #breakpoint()
    for sib in sibbricks:
        sibbrickccs |= ccs_ex_self[sib]
    return ccs_ex_self[brick] - {brick} - sibbrickccs


def bruteforce_unsafe(grid):
    # brute-force method: try disintegrating each brick in turn,
    # then regenerate the settle map

    result = {}
    
    for i in range(1, BRICKS+1):
        gridcopy = grid.copy()
        gridcopy[gridcopy == i] = 0
        adj2 = settle_map(gridcopy)
        sb2 = findcc(adj2, 0)
        moved = allbricks - sb2 - {i}
        result[i] = moved

        #print(f"changed after disintegrate {i}: {moved}")
    return result


unsafe = allbricks - sb
ccs_ex_self = {x: findcc(sm, x) - {x} for x in allbricks}
#ccs_counts = {x:len(y) for (x,y) in ccs_ex_self.items()}

unsafe_aboves = {x: unsafe_above(sm, ccs_ex_self, x) for x in allbricks}
answer = sum(len(y) for y in unsafe_aboves.values())

# for sbi in sb:
#     assert sbi not in ccs_counts
#     ccs_counts[sbi] = 0

# result = sum(ccs_counts.values())
# print(result)
#print(ccs_ex_self)
print(unsafe_aboves)
print(answer)

brute = bruteforce_unsafe(grid)
brute_answer = sum(len(y) for y in brute.values())
print(brute_answer)

print(f"the answers are equal? {brute_answer} {answer}")

#breakpoint()
#bruteforce_unsafe(grid)

# 127620 is too high
# 118472 is too high

