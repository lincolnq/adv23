samp = """...........
.....###.#.
.###.##..#.
..#.#...#..
....#.#....
.##..S####.
.##..#...#.
.......##..
.##.#.####.
.##..##.##.
..........."""


from dataclasses import dataclass
import numpy as np
import functools

from helpers import *

parser = Matrix()

#input = parser.parse(samp)
input = parser.parse(open('p21inp.txt').read())
start = tuple(np.argwhere(input == 'S')[0])
input[input == 'S'] = '.'

def printgrid(g):
    print("\n".join("".join(row) for row in g))

def progress(g):
    # find all '.' neighbors of all 'O' cells
    ocells = g == 'O'
    op = np.pad(g == 'O', (1,), constant_values=(False,))

    neighbors = op[2:, 1:-1] | op[:-2, 1:-1] | op[1:-1, :-2] | op[1:-1, 2:]
    neighbors = neighbors & (g == '.')

    g2 = g.copy()
    g2[ocells] = '.'
    g2[neighbors] = 'O'
    return g2

# printgrid(input)
g = input
# for _ in range(64):
#     g = progress(g)
#     print()
#     printgrid(g)
#     print(np.sum(g == 'O'))


# for part 2 we need a quad tree

# does not include the rank 0 nodes.
ALL_NODES: dict[tuple[int, int, int, int], 'QNode'] = {}

class QNode:
    # for rank 0 we have no children.
    rank: int
    # row then col.
    children: tuple['QNode', 'QNode', 'QNode', 'QNode']


    def __init__(self, rank, children):
        self.rank = rank
        self.children = children

    def __repr__(self) -> str:
        return f"QNode(rank={self.rank}, pop={self.count})"

    @functools.cache
    def centered_subsquare(self):
        """Returns the subsquare of rank-1 that is centered at our center."""
        return QNode.make(*(self.children[i].children[3-i] for i in range(4)))
    
    def subsquare_edge(self, edge):
        """Returns the subsquare of rank-1 that is centered at the given edge of
        our centered_subsquare [...]"""
        # top
        if edge==0:
            return QNode.make(self.children[0].children[1], self.children[1].children[0],
                              self.children[0].children[3], self.children[1].children[2])
        # bottom
        if edge==2:
            return QNode.make(self.children[2].children[1], self.children[3].children[0],
                              self.children[2].children[3], self.children[3].children[2])
        # left
        if edge==1:
            return QNode.make(self.children[0].children[2], self.children[0].children[3],
                              self.children[2].children[0], self.children[2].children[1])
        # right
        if edge==3:
            return QNode.make(self.children[1].children[2], self.children[1].children[3],
                              self.children[3].children[0], self.children[3].children[1])


    @functools.cached_property
    def grid(self):
        if self.rank == 0:
            return np.array([[self.val]])
        return np.vstack([
            np.hstack([self.children[0].grid, self.children[1].grid]),
            np.hstack([self.children[2].grid, self.children[3].grid]),
        ])
    
    @functools.cached_property
    def count(self):
        if self.rank == 0:
            return 1 if self is STEP else 0
        return sum(c.count for c in self.children)
    
    @staticmethod
    def make(*children):
        # if any QNode exists with these exact children, return it,
        # otherwise create a new one.
        #assert len(children) == 4
        #assert all(c.rank == children[0].rank for c in children)

        ids = tuple(id(c) for c in children)
        if ids in ALL_NODES:
            #print("found existing node")
            return ALL_NODES[ids]
        else:
            n = QNode(rank=children[0].rank + 1, children=children)
            ALL_NODES[ids] = n
            if len(ALL_NODES) % 100000 == 0:
                print(f"all nodes size: {len(ALL_NODES)} - just added rank {n.rank}")
            return n

def subcenters(rank, center):
    """Returns offsets of our child centers from `center`."""
    assert rank != 0

    if rank == 1:
        return [
                (center[0] - 1, center[1] - 1),
                (center[0] - 1, center[1]),    
                (center[0], center[1] - 1),    
                (center[0], center[1]),        
        ]
    
    half = 2 ** (rank - 2)
    return [(center[0] - half, center[1] - half),
            (center[0] - half, center[1] + half),
            (center[0] + half, center[1] - half),
            (center[0] + half, center[1] + half),]
    

EMPTY = QNode(rank=0, children=None); EMPTY.val = '.'
ROCK = QNode(rank=0, children=None); ROCK.val = '#'
STEP = QNode(rank=0, children=None); STEP.val = 'O'

LOAD_MEMO = {}

def load_quads(g: np.ndarray, center: tuple[int, int], rank: int) -> QNode:
    # load copies of grid into QNode(s). qnode returned should be of the given rank.

    # memoize only loads of the input array
    memokey = (center[0] % g.shape[0], center[1] % g.shape[1], rank)
    if g is input and memokey in LOAD_MEMO:
        return LOAD_MEMO[memokey]
    
    #if rank==0:
    #    print(f"loading rank {rank} at {center}")
    if rank == 0:
        center = (center[0] % g.shape[0], center[1] % g.shape[1])
        if g[center] == '.':
            return EMPTY
        elif g[center] == '#':
            return ROCK
        else:
            return STEP
    else:
        children = [load_quads(g, subcenter, rank - 1) for subcenter in subcenters(rank, center)]
        result = QNode.make(*children)
        LOAD_MEMO[memokey] = result
        return result

def update_quad(q: QNode, center: tuple[int, int], dest: tuple[int, int], c: QNode) -> QNode:
    """Update the quad tree q, centered at 'center', such that the cell at 'dest' is set to c."""

    if q.rank == 0:
        assert center == dest, f"center={center}, dest={dest}, rank={q.rank}"
        return c
    
    childix = 2 * (dest[0] >= center[0]) + (dest[1] >= center[1])
    
    newchild = update_quad(q.children[childix], subcenters(q.rank, center)[childix], dest, c)
    return QNode.make(*[newchild if i == childix else q.children[i] for i in range(4)])

def print_quad(q: QNode):
    printgrid(q.grid)


def query(q: QNode, center: tuple[int, int], dest: tuple[int, int]) -> QNode:
    """Return the node at dest in the quad tree q, centered at center."""
    if q.rank == 0:
        assert center == dest
        return q

    childix = 2 * (dest[0] >= center[0]) + (dest[1] >= center[1])
    return query(q.children[childix], subcenters(q.rank, center)[childix], dest)


ADVANCE_MEMO = {}

def advance_timestep(q: QNode) -> QNode:
    """Simulate the logic for the 2**(rank - 1) subsquare at our center one step.
    Returns the subsquare.

    To do this, we need to call advance_timestep recursively 4 times, where
    the result is 4 subsquares (of rank-2) that we will then assemble into our
    rank-1 result.

    Thus our main task is to create the arguments for our 4 sub-calls, which
    require rank-1 subsquares, but these are not centered, so we need to assemble
    them out of rank-2 subsquares.
    """

    if q.count == 0:
        return q.centered_subsquare()

    # check the memo
    if id(q) in ADVANCE_MEMO:
        return ADVANCE_MEMO[id(q)]

    # but first - base case.

    # we can't advance a 0-rank (1x1) node or a 1-rank (2x2) node, the smallest we can do
    # is a 2-rank node (4x4)
    assert q.rank >= 2
    if q.rank == 2:
        g2 = progress(q.grid)
        # and return the 1-rank node at the center of resulting grid
        result = load_quads(g2, (2, 2), 1)
        ADVANCE_MEMO[id(q)] = result
        if len(ADVANCE_MEMO) % 100000 == 0:
            print(f"advance memo size: {len(ADVANCE_MEMO)}")
        return result
        
    #breakpoint()
    ss_center = q.centered_subsquare().centered_subsquare()
    ss_corners = [q.children[i].centered_subsquare() for i in range(4)]
    ss_edges = [q.subsquare_edge(i).centered_subsquare() for i in range(4)]

    result = QNode.make(
        advance_timestep(QNode.make(ss_corners[0], ss_edges[0], ss_edges[1], ss_center)),
        advance_timestep(QNode.make(ss_edges[0], ss_corners[1], ss_center, ss_edges[3])),
        advance_timestep(QNode.make(ss_edges[1], ss_center, ss_corners[2], ss_edges[2])),
        advance_timestep(QNode.make(ss_center, ss_edges[3], ss_edges[2], ss_corners[3])),
    )
    ADVANCE_MEMO[id(q)] = result
    if len(ADVANCE_MEMO) % 100000 == 0:
        print(f"advance memo size: {len(ADVANCE_MEMO)}")
    return result

ADVANCE_FAST_MEMO = {}

PEAK_RANK = 0

def advance_timestep_fast(q: QNode, steps: int) -> QNode:
    """Advance by (up to) powers of 2 instead of one step at a time.

    Same args as advance_timestep, same constraints (advances the center subsquare).

    A rank 3 node can be advanced at most 1 step, rank 4 2 steps:
        max_steps = 2 ** (rank - 3)
    """

    if q.count == 0:
        return q.centered_subsquare()

    #print(f"ADVANCE FAST({q}, steps={steps})")
    maxsteps = max_steps(q.rank)
    assert 0 <= steps <= maxsteps

    if steps == 0:
        return q.centered_subsquare()
    
    if steps == 1 or q.rank <= 3:
        # don't memoize here or we duplicate the non-fast advance memo
        return advance_timestep(q)

    midpoint_steps = min(steps, maxsteps // 2)
    final_steps = steps - midpoint_steps

    if (id(q), steps) in ADVANCE_FAST_MEMO:
        return ADVANCE_FAST_MEMO[(id(q), steps)]

    ss_center = advance_timestep_fast(q.centered_subsquare(), midpoint_steps)
    ss_corners = [advance_timestep_fast(q.children[i], midpoint_steps) for i in range(4)]
    ss_edges = [advance_timestep_fast(q.subsquare_edge(i), midpoint_steps) for i in range(4)]

    # PAUSE POINT: we have 9 subsquares into half-result. Point to be able to render it
    
    result = QNode.make(
        advance_timestep_fast(QNode.make(ss_corners[0], ss_edges[0], ss_edges[1], ss_center), final_steps),
        advance_timestep_fast(QNode.make(ss_edges[0], ss_corners[1], ss_center, ss_edges[3]), final_steps),
        advance_timestep_fast(QNode.make(ss_edges[1], ss_center, ss_corners[2], ss_edges[2]), final_steps),
        advance_timestep_fast(QNode.make(ss_center, ss_edges[3], ss_edges[2], ss_corners[3]), final_steps),
    )

    ## SLOW STUFF
    if False:
        midpoint9 = np.vstack([
            np.hstack([ss_corners[0].grid, ss_edges[0].grid, ss_corners[1].grid]),
            np.hstack([ss_edges[1].grid, ss_center.grid, ss_edges[3].grid]),
            np.hstack([ss_corners[2].grid, ss_edges[2].grid, ss_corners[3].grid]),
        ])
        full = 2 ** q.rank
        half = 2 ** (q.rank - 1)
        quarter = 2 ** (q.rank - 2)
        eighth = 2 ** (q.rank - 3)
        assert midpoint9.shape[0] == quarter + half

        finalgrid = result.grid
        assert finalgrid.shape[0] == half

        #if q.count > 0 and q.rank >= 5:
        #    print(f"midpoint9 for {q}: midpop={np.sum(midpoint9 == 'O')}, finalpop={result.count}")
        #    printgrid(midpoint9)

        # slow check: advance the original grid, then compare to midpoint and final
        steps_to_midpoint = 2 ** (q.rank - 4)
        slowgrid = q.grid
        for _ in range(steps_to_midpoint):
            slowgrid = progress(slowgrid)

        slowgrid2 = slowgrid.copy()
        # trim the slowgrid to the same size as midpoint9 and compare
        assert np.all(midpoint9 == slowgrid[eighth:full - eighth, eighth:full - eighth])

        # now compare the slowgrid to the final result
        for _ in range(steps_to_midpoint):
            slowgrid = progress(slowgrid)
        
        assert np.all(result.grid == slowgrid[quarter:full - quarter, quarter:full - quarter])
        #if np.sum(midpoint9 == 'O') == 30:
        #    breakpoint()

        if q.rank >= 4 and q.count > 0:
            print(f"ADVANCE FAST({q}, centerpop={q.centered_subsquare().count}) -> midpoint(count={np.sum(midpoint9 == 'O')}) -> {result}.  (midpoint slowgrid pop={np.sum(slowgrid2 == 'O')}, endpoint slowgrid pop={np.sum(slowgrid == 'O')})")

    ADVANCE_FAST_MEMO[(id(q), steps)] = result
    if len(ADVANCE_FAST_MEMO) % 100000 == 0:
        print(f"advance fast memo size: {len(ADVANCE_FAST_MEMO)}, key")

    global PEAK_RANK
    if result.rank > PEAK_RANK:
        PEAK_RANK = result.rank
        print(f"Peak rank is now {PEAK_RANK}")

    return result


# we'll cache the quads loaded at 0,0 since we need to reuse them every generation.
LOADED_QUADS = {}


# expand the universe around a given quad tree node - return a q+1-rank node
def expand(q: QNode, center: tuple) -> QNode:
    # in lieu of actually being able to expand anywhere, we will just always expand
    # from the origin. That allows us to simply cache the loaded quads from center.
    assert center == (0, 0)

    if q.rank+1 in LOADED_QUADS:
        base = LOADED_QUADS[q.rank+1]
    else:        
        base = load_quads(g, center, q.rank + 1)
        LOADED_QUADS[q.rank+1] = base

    c0, c1, c2, c3 = base.children

    # now place q at the center of base
    return QNode.make(
        QNode.make(c0.children[0], c0.children[1], c0.children[2],  q.children[0]),
        QNode.make(c1.children[0], c1.children[1],  q.children[1], c1.children[3]),
        QNode.make(c2.children[0],  q.children[2], c2.children[2], c2.children[3]),
        QNode.make( q.children[3], c3.children[1], c3.children[2], c3.children[3]),
    )

# test expand
def run_tests():
    # testing load and update
    q = load_quads(input, (0, 0), 2)
    q = update_quad(q, (0, 0), (0, 0), STEP)
    assert np.all(q.grid == q.grid)

    q2 = load_quads(input, (0, 0), 3)
    q2 = update_quad(q2, (0, 0), (0, 0), STEP)
    q3 = load_quads(input, (0, 0), 4)
    q3 = update_quad(q3, (0, 0), (0, 0), STEP)
    

    # testing expand and centered_subsquare
    q2e = expand(q, (0, 0))
    assert np.all(q2e.grid == q2.grid)

    q3e = expand(q2, (0, 0))
    assert np.all(q3e.grid == q3.grid)

    assert np.all(q3e.centered_subsquare() == q2)
    assert np.all(q2e.centered_subsquare() == q)


    # testing simulate
    q = update_quad(load_quads(input, (0,0), 4), (0, 0), start, STEP)
    printgrid(q.grid)
    expected = progress(q.grid)
    printgrid(expected)
    print(np.sum(expected == 'O'))
    q2 = simulate1(q).centered_subsquare()
    printgrid(q2.grid)
    print(q2.count)
    assert np.all(expected == q2.grid)

def max_steps(rank: int) -> int:
    if rank < 3:
        return 1
    return 2 ** (rank - 3)
    

def expand_for_advance(q: QNode, desired_steps: int) -> QNode:
    # before we take our first step, we may need to expand the universe
    # Since the step could fill to expand q.centered_subsquare and then we throw 
    # away everything outside that when we advance, we aren't ready to simulate
    # until we know the doubly centered subsquare is the only populated region
    while ((max_steps(q.rank) < desired_steps)
            or (q.centered_subsquare().centered_subsquare().count != q.count)):
        q = expand(q, (0, 0))

    return q

def shrink(q: QNode) -> QNode:
    while q.rank > 2 and q.centered_subsquare().count == q.count:
        q = q.centered_subsquare()
    return q

def simulate(q: QNode, steps: int) -> tuple[QNode, int]:
    print(f"SIMULATE - starting with {q}")
    q = expand_for_advance(q, steps)
    print(f"SIMULATE - expanded to {q}")
    return advance_timestep_fast(q, steps)

def simulate1(q: QNode) -> tuple[QNode, int]:
    q = expand_for_advance(q, 1)
    return advance_timestep(q)


q = load_quads(input, (0, 0), 10)
q = update_quad(q, (0, 0), start, STEP)

#import cProfile
#cProfile.run('simulate(q, 26501365)')

def go():
    q = load_quads(input, (0, 0), 10)
    q = update_quad(q, (0, 0), start, STEP)
    #print_quad(q)
    print(f"Starting pop is {q.count}\n")
    
    target_steps = 26501365

    qfast = simulate(q, target_steps)
    print(f"Population after {target_steps} steps is {qfast.count}")


#    for target_steps in [26501365]:
        

    # SIM_STEPS = 100

    # #print(f"shrink qfast: {qfast.count} and {shrinkfast.count}")
    # qslow = q

    # for step in range(SIM_STEPS):
    #     qfast = simulate(q, step+1)
    #     qslow = simulate1(qslow)
        
    #     print(f"on step {step + 1}: Fast Population is now {qfast.count}, slow is {qslow.count}\n")
    #     assert qfast.count == qslow.count

    # shrinkslow = shrink(q)
    # # now compare
    # if np.all(shrinkfast.grid == shrinkslow.grid):
    #     print(f"fast and slow agree after {steps} steps.")
    # else:
    #     print(f"fast and slow disagree after {steps} steps.")
    #     print("fast:")
    #     print_quad(shrinkfast)
    #     print("slow:")
    #     print_quad(shrinkslow)

#run_tests()

go()