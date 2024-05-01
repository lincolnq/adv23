# blah blah blah

samp = """.|...\\....
|.-.\\.....
.....|-...
........|.
..........
.........\\
..../.\\\\..
.-.-/..|..
.|....-|.\\
..//.|...."""

import numpy as np
from helpers import *

parser = Matrix()
real = open('p16inp.txt').read()
grid = parser.parse(real)

def vec2str(v):
    return {(-1,0): '^', (1,0): 'v', (0,-1): '<', (0,1): '>'}[v]

def simulate(m, startpos, startvec):
    """Simulate the beam entering at 'startpos' in the given direction.
    """
    beams = [(startpos, startvec)]
    beam_history = set()
    
    energized = numpy.zeros_like(m)
    energized.fill('.')

    tracker = m.copy()

    steps = 0

    # now advance the beams
    while len(beams):
        beam = beams.pop(0)
        # if we've already seen this beam at this position and direction, ignore it
        if beam in beam_history:
            #print(f"beam @ {beam} seen, terminating")
            continue

        # ok, we haven't seen this beam before
        beam_history.add(beam)
        beams.extend(advance_beam(m, beam))
        energized[beam[0]] = '1'
        if m[beam[0]] == '.':
            if tracker[beam[0]] == '.':
                tracker[beam[0]] = vec2str(beam[1])
            elif tracker[beam[0]] == '2':
                tracker[beam[0]] = '3'
            else:
                tracker[beam[0]] = '2'
        steps += 1

        #print()
        #printgrid(tracker)

        if steps % 1000 == 0:
            print(f"{steps}")
    
    return energized

def next_beam_vecs(vec, c):
    """
    If the beam encounters empty space (.), it continues in the same direction.
    If the beam encounters a mirror (/ or \), the beam is reflected 90 degrees depending on the angle of the mirror. For instance, a rightward-moving beam that encounters a / mirror would continue upward in the mirror's column, while a rightward-moving beam that encounters a \ mirror would continue downward from the mirror's column.
    If the beam encounters the pointy end of a splitter (| or -), the beam passes through the splitter as if the splitter were empty space. For instance, a rightward-moving beam that encounters a - splitter would continue in the same direction.
    If the beam encounters the flat side of a splitter (| or -), the beam is split into two beams going in each of the two directions the splitter's pointy ends are pointing. For instance, a rightward-moving beam that encounters a | splitter would split into two beams: one that continues upward from the splitter's column and one that continues downward from the splitter's column.
    """

    if c == '.':
        return [vec]
    elif c == '/':
        return [(-vec[1], -vec[0])]
    elif c == '\\':
        return [(vec[1], vec[0])]
    elif c == '|':
        # vertical splitter
        if vec[1] == 0:
            # column axis zero, meaning pointy end, ignore
            return [vec]
        else:
            # split into two vertical beams
            return [(-1,0), (1,0)]
    elif c == '-':
        # horizontal splitter
        if vec[0] == 0:
            # row axis zero, meaning pointy end, ignore
            return [vec]
        else:
            # split into two horizontal beams
            return [(0,-1), (0,1)]



def advance_beam(m, beam):
    """Returns a list of new beams that result from advancing the given beam."""
    pos, vec = beam
    
    result = []
    for newvec in next_beam_vecs(vec, m[pos]):

        newpos = (pos[0] + newvec[0], pos[1] + newvec[1])

        # check that newpos is in bounds. if not, this beam dies
        if newpos[0] < 0 or newpos[0] >= m.shape[0] or newpos[1] < 0 or newpos[1] >= m.shape[1]:
            continue

        result.append((newpos, newvec))
    return result

def printgrid(g):
    print("\n".join(["".join(row) for row in g]))

def sim_and_count(m, startpos, startvec):
    energized = simulate(m, startpos, startvec)
    return np.count_nonzero(energized == '1')

def tryall(g):
    best = 0
    for i in range(g.shape[0]):
        # try entering the beam along left edge
        cells = sim_and_count(grid, (i,0), (0,1))
        print(f"entering at {i},0 rw: {cells} energized cells")
        best = max(best, cells)

        # right edge
        col = g.shape[1] - 1
        cells = sim_and_count(grid, (i,col), (0,-1))
        print(f"entering at {i},{col} lw: {cells} energized cells")
        best = max(best, cells)

    for j in range(g.shape[1]):
        # top edge
        cells = sim_and_count(grid, (0,j), (1,0))
        print(f"entering at 0,{j} dn: {cells} energized cells")
        best = max(best, cells)

        # bottom edge
        row = g.shape[0] - 1
        cells = sim_and_count(grid, (row,j), (-1,0))
        print(f"entering at {row},{j} up: {cells} energized cells")
        best = max(best, cells)

    print(f"best: {best}")

printgrid(grid)
energized = simulate(grid, (0,0), (0,1))
printgrid(energized)

cells = np.count_nonzero(energized == '1')
print(f"{cells} energized cells")

tryall(grid)