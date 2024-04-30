samp = """.F----7F7F7F7F-7....
.|F--7||||||||FJ....
.||.FJ||||||||L7....
FJL7L7LJLJ||LJ.L-7..
L--J.L7...LJS7F-7L7.
....F-J..F7FJ|L7L7L7
....L7.F7||L7|.L7L7|
.....|FJLJ|FJ|F7|.LJ
....FJL-7.||.||||...
....L---J.LJ.LJLJ..."""

input = open('p10inp.txt').read()

from helpers import *
import numpy as np
from dataclasses import dataclass

parsed = Matrix().parse(input)

# find the start node
startnode = np.argwhere(parsed == 'S')[0]
print(startnode)

def neighbors(ix):
    """Yields all orthogonal neighbors of a given index within bounds."""
    (r,c) = ix
    if r > 0:
        yield (r-1,c)
    if r < parsed.shape[0]-1:
        yield (r+1,c)
    if c > 0:
        yield (r,c-1)
    if c < parsed.shape[1]-1:
        yield (r,c+1)

PIPENEIGHBORS = {
    '|': np.array([(-1,0),(1,0)]),
    '-': np.array([(0,-1),(0,1)]),
    'L': np.array([(-1,0),(0,1)]),
    'J': np.array([(-1,0),(0,-1)]),
    '7': np.array([(1,0),(0,-1)]),
    'F': np.array([(1,0),(0,1)])
}


def pipeneighbors(c):
    """
    Given a pipe character 'c', returns two 2-vectors indicating the connections
    of that type of pipe, or the empty vector if the character is anything other
    than the six types of pipe squares.

    The pipes are arranged in a two-dimensional grid of tiles:

        | is a vertical pipe connecting north and south.
        - is a horizontal pipe connecting east and west.
        L is a 90-degree bend connecting north and east.
        J is a 90-degree bend connecting north and west.
        7 is a 90-degree bend connecting south and west.
        F is a 90-degree bend connecting south and east.
    """


    return PIPENEIGHBORS.get(c, np.array([]))

def nsstartnode(m):
    """Returns the neighbors of the start node"""
    incoming_ns = []
    for n in neighbors(startnode):
        pns = pipeneighbors(m[n])
        if len(pns) > 0 and any(np.array_equal(startnode, loc) for loc in n + pns):
            incoming_ns.append(n)

    # There should be two of these
    print(incoming_ns)
    assert len(incoming_ns) == 2
    return incoming_ns


def bfs(m):
    # bfs to find the farthest node

    @dataclass
    class Node:
        dist: int
        loc: np.ndarray

    seen = {tuple(startnode)}
    edge = [Node(1, x) for x in nsstartnode(m)]

    while edge:
        next = edge.pop(0)
        if tuple(next.loc) in seen:
            # we must have visited this node before, but we wouldn't've enqueued it as
            # a neighbor, so we are done
            #print(f"Seen {next} before, returning it")
            # return next.dist
            return seen
        seen.add(tuple(next.loc))
        #print(f"Traversing to {next}")
        
        # now add neighbors. one of the neighbors is where we came from, the other is new.
        for nv in pipeneighbors(m[next.loc]):
            newloc = next.loc + nv
            if tuple(newloc) not in seen:
                edge.append(Node(next.dist+1, tuple(newloc)))

def crossings(m, loop, loc):
    """Returns the number of times a ray starting from 'loc' crosses the loop
    of pipe indicated by 'loop'. 

    We count by casting the ray diagonally up and leftwards.
    
    Counts all | and - crossings along the diagonal; corners are counted when 
    they are J and F but not when they are L and 7.

    If the result is even, the location is conceptually 'inside' the loop,
    otherwise it is 'outside'.

    The loop itself is counted as 'inside', in case you care
    """

    crossings = 0
    while True:
        (r,c) = loc
        if r<0 or c<0:
            break

        if loc in loop and m[loc] in '|-JF':
            crossings += 1

        if m[loc] == 'S':
            # it's the start node, we have to fix this up later, but for now count it
            # probabilistically
            print("encountered start node in crossing search, need to figure out crossings, counting it as a YES")
            crossings += 1

        loc = (r-1,c-1)
    
    return crossings

def draw_with_crossings(m, loop):
    output = m.copy()
    for locm in np.indices(m.shape).reshape(2,-1).T:
        loc = tuple(locm)
        if loc not in loop:
            cx = crossings(m, loop, loc)
            if cx % 2 == 0:
                output[loc] = 'O'
            else:
                output[loc] = 'I'


    return output

loop = bfs(parsed)
#result = bfs(parsed)
print(f"Result loop is {loop}")

# let's make a map showing ins and outs
newm = draw_with_crossings(parsed, loop)
#asstr = '\n'.join(''.join(r) for r in newm)

count_inside = newm[newm == 'I'].size

print(f"Result is:\n{count_inside}")

#x = crossings(parsed, loop, (4,4))
