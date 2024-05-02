samp = """R 6 (#70c710)
D 5 (#0dc571)
L 2 (#5713f0)
D 2 (#d2c081)
R 2 (#59c680)
D 2 (#411b91)
L 5 (#8ceee2)
U 2 (#caa173)
L 1 (#1b58a2)
U 2 (#caa171)
R 2 (#7807d2)
U 3 (#a77fa3)
L 2 (#015232)
U 2 (#7a21e3)"""

minisamp = """R 1 (blah)
D 1 (blah)
L 1 (blah)
U 1 (blah)"""

import numpy as np
import ranges
from helpers import *

parser = Lines() ** Sections(" ", str, int, str)
real = open('p18inp.txt').read()

lines = parser.parse(real)
newlines = []
for l in lines:
    hexcode = l[2].strip('()#')
    count = int(hexcode[:5], 16)
    dir = {'0': 'R', '1': 'D', '2': 'L', '3': 'U'}[hexcode[5]]
    newlines.append((dir, count, ""))
lines = newlines
#print(lines)
SIZE_START = ((20,30), (5, 10))
#lines = parser.parse(real)
#SIZE_START = ((500,500), (250, 0))

# Assume right handedness
#RIGHTHAND = True

#print(lines)

def carve_border(lines):
    size, start = SIZE_START
    grid = np.zeros(size, dtype=numpy.unicode_)
    grid.fill('.')

    pos = start
    dirs = {'D': (1,0), 'U': (-1,0), 'L': (0,-1), 'R': (0,1)}
    corners = {"DL": 'J', "DR": 'L', "UL": '7', "UR": 'F',
               "LD": 'F', "LU": 'L', "RD": '7', "RU": 'J'
               }

    for (i, (dir, c, _)) in enumerate(lines):
        # draw prior corner
        if i > 0:
            prevdir = lines[i-1][0]
            corner = corners[prevdir + dir]
            grid[pos] = corner

        for i in range(c):
            vec = dirs[dir]
            pos = (pos[0] + vec[0], pos[1] + vec[1])
            grid[pos] = '|' if dir in 'UD' else '-'
    
    # draw final corner
    prevdir, dir = lines[-1][0], lines[0][0]
    corner = corners[prevdir + dir]
    grid[start] = corner

    return grid

def fill(grid):
    for row in range(len(grid)):
        for col in range(len(grid[row])):

            if grid[row, col] != '.':
                continue
            
            intersections = 0
            # go up and to the left
            for i in range(1, min(row, col)+1):

                if grid[row-i, col-i] in '|-JF':
                    intersections += 1
            
            if intersections % 2 == 1:
                grid[row, col] = '~'

def showgrid(g):
    return '\n'.join([''.join(row) for row in g])




def carve_border_with_scanlines(lines):
    pos = (0,0)
    dirs = {'D': (1,0), 'U': (-1,0), 'L': (0,-1), 'R': (0,1)}
    corners = {"DL": 'J', "DR": 'L', "UL": '7', "UR": 'F',
               "LD": 'F', "LU": 'L', "RD": '7', "RU": 'J'
               }

    # Accumulate pos as we carve the border, tracking all vertical lines start and endpoints
    # (x coord, direction, min y, max y)
    verticals = []
    # Only tracking the hidden horizontals - and only on which vertical they are
    # (y coord, min x, max x)
    horizontals = []
    lastdir = ''

    for (i, (dir, c, _)) in enumerate(lines):
        vec = dirs[dir]
        newpos = (pos[0] + vec[0] * c, pos[1] + vec[1] * c)
        if dir in 'UD':
            verticals.append((pos[1], dir, min(pos[0], newpos[0]), max(pos[0], newpos[0])))
        if (lastdir+dir) in ('DR', 'UL'):
            # hidden horizontal scanline.
            startend = min(pos[1], newpos[1]), max(pos[1], newpos[1])
            # only include if length >= 2, since 1 or less won't need a horizontal fix-up
            if startend[1] - startend[0] >= 2:
                horizontals.append((pos[0], *startend))
        pos = newpos
        lastdir = dir
    
    return verticals, horizontals


def query(verticals, y):
    """Returns all verticals that touch 'y'"""
    return sorted((x for x in verticals if x[2] <= y <= x[3]), key=lambda x: x[0])

def findnext(verticals, y):
    """Returns the next (min or max) y-coordinate after 'y' that has any scanlines, or
    None"""
    allys = sorted(set(x[2] for x in verticals) | set(x[3] for x in verticals))
    for y2 in allys:
        if y2 > y:
            return y2
    return None

def merge_scanline(line, startx, endx):
    # Line is a RangeSet. Startx and endx represent an inclusive range.
    # mutates (and return) line
    line.add(ranges.Range(startx, endx, include_end=True))
    return line

def get_scanline(verticals, horizontals, y):
    touching = query(verticals, y)
    # accumulate a scan line with the delta
    line = ranges.RangeSet()

    # three states: beginning of a section, middle of a section no end in sight, 
    # and "pending end".
    # When beginning, only U is valid. When no end in sight, U and D are valid and
    # only D indicates "pending end". When "pending end", D extends, and U ends the
    # prior section to start a new one.
    firstu = None
    lastd = None


    for (x, dir, *_) in touching:
        if firstu is None:
            assert dir == 'U', "started a section with a D"
            # beginning
            firstu = x
        elif firstu is not None and lastd is None:
            # we're in the middle of a section no end in sight
            if dir == 'U':
                # consecutive "U", just ignore
                pass
            elif dir == 'D':
                # move to pending end
                lastd = x

        elif firstu is not None and lastd is not None:
            # we're in a pending end
            if dir == 'U':
                # end the prior section and start a new one
                merge_scanline(line, firstu, lastd)
                firstu = x
                lastd = None
            elif dir == 'D':
                # extend the pending end
                lastd = x

    # if we're in a pending end, end it
    if lastd is not None:
        merge_scanline(line, firstu, lastd)

    # add horizontals
    for (hy, startx, endx) in horizontals:
        if hy == y:
            merge_scanline(line, startx + 1, endx - 1)

    return [r.end - r.start + 1 for r in line]


def fill_with_scanlines(verticals, horizontals):
    # starting row is the minimum of verticals
    curr = min(x[2] for x in verticals)

    # there should be at least two points touching
    scanlines = []
    while curr is not None:
        scanlines.append(get_scanline(verticals, horizontals, curr))

        next = findnext(verticals, curr)
        if next is not None and next - curr > 1:
            # everything in the middle is repeated (next - curr - 1) times:
            line = get_scanline(verticals, horizontals, curr + 1)
            count = next - curr - 1
            scanlines.append([x * count for x in line])
        
        curr = next
    
    return scanlines


def test_scan(x, expected):
    inp = parser.parse(x)
    verts,horiz = carve_border_with_scanlines(inp)
    print(verts)
    lines = fill_with_scanlines(verts, horiz)
    count = sum(sum(l) for l in lines)
    if expected != count:

        print(f"expected {expected}, got {count}: {lines}")
        # also trace it using the old method and print it
        grid = carve_border(inp)
        fill(grid)
        print(showgrid(grid))
        nz = np.count_nonzero(grid != '.')
        print(nz)


def run_tests():
    test_scan("R 1 x\nD 1 x\nL 1 x\nU 1 x", 4)
    test_scan("R 2 x\nD 2 x\nL 2 x\nU 2 x", 9)
    test_scan("R 2 x\nD 10 x\nL 2 x\nU 10 x", 33)
    test_scan("R 2 x\nD 3 x\nR 2 x\nD 4 x\nL 4 x\nU 7 x", 34)
    test_scan("R 2 x\nD 3 x\nR 3 x\nU 3 x\nR 2 x\nD 7 x\nL 7 x\nU 7 x", 58)
    test_scan(samp, 62)


# grid = carve_border(lines)
# fill(grid)
# print(showgrid(grid))
# nz = np.count_nonzero(grid != '.')
# print(nz)


#run_tests()

verts,horiz = carve_border_with_scanlines(lines)
lines = fill_with_scanlines(verts, horiz)
count = sum(sum(l) for l in lines)
print(count)