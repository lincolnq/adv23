samp = """#.#####################
#.......#########...###
#######.#########.#.###
###.....#.>.>.###.#.###
###v#####.#v#.###.#.###
###.>...#.#.#.....#...#
###v###.#.#.#########.#
###...#.#.#.......#...#
#####.#.#.#######.#.###
#.....#.#.#.......#...#
#.#####.#.#.#########v#
#.#...#...#...###...>.#
#.#.#v#######v###.###v#
#...#.>.#...>.>.#.###.#
#####v#.#.###v#.#.###.#
#.....#...#...#.#.#...#
#.#########.###.#.#.###
#...###...#...#...#.###
###.###.#.###v#####v###
#...#...#.#.>.>.#.>.###
#.###.###.#.###.#.#v###
#.....###...###...#...#
#####################.#"""

samp2 = """#.###
#...#
##.##"""

from helpers import *
import numpy as np

parser = Matrix()

#grid = parser.parse(samp)
grid = parser.parse(open("p23inp.txt").read())

#print(grid)

startpoint = (0, (grid[0] == '.').nonzero()[0][0])
endpoint = (len(grid) - 1, (grid[-1] == '.').nonzero()[0][0])
print(startpoint, endpoint)

ADJ = [(0, 1), (1, 0), (0, -1), (-1, 0)]
FORCED_MOVES = ">v<^"

def grid_neighbors(grid, point):
    """Assuming 'point' is a walkable grid square, return a list of adjacent
    walkable grid squares."""

    #breakpoint()
    assert grid[point] != '#'
    result = []
    for (dr, dc) in ADJ:
        r,c = point[0] + dr, point[1] + dc
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
            continue

        # if you're on a '>v<^' square, you can move only in the specified direction
        #if grid[point] in FORCED_MOVES and (dr, dc) != ADJ[FORCED_MOVES.index(grid[point])]:
        #    continue
        
        # otherwise can move in any direction that is not '#'
        if grid[r,c] != '#':
            result.append((r,c))
    return result

def flood1(grid, curr, next):
    """Flood neighbors from `curr` in direction of `next` until there is a branchpoint.
    Returns (branchpoint, dist, [next neighbors]).
    """
    ns = grid_neighbors(grid, next)
    dist = 1
    while True:
        if curr in ns:
            ns.remove(curr)

        if ns == endpoint:
            # found end
            return (next, dist, [])
        
        if len(ns) == 0:
            # dead end
            return (next, dist, [])

        if len(ns) > 1:
            # branchpoint
            return (next, dist, ns)
        
        #print(f"flood {next}")
        curr = next
        next = ns[0]
        dist += 1
        ns = grid_neighbors(grid, next)
    

def explore_dfs(grid, curr, nextss, visited, dist):
    """Explore the grid from start to end, returning the maximum distance.
    
    Invariant: curr is the previous node visited and is already in `visited`.
    Next is not in `visited`, and indicates the next viable single step along
    the path
    Dist is the path taken so far from start to curr.
    """

    (branchpoint, ddist, nextsses) = flood1(grid, curr, nextss)
    if branchpoint in visited:
        # already visited this branchpoint
        return []
    
    visited = visited | {branchpoint}
    dist += ddist

    if branchpoint == endpoint:
        #print(f"found end: {dist}")
        return [dist]

    #print(f"explore {curr}")

    paths = []
    for n in nextsses:
        #print(f" next {n}")    

        paths.extend(explore_dfs(grid, branchpoint, n, visited, dist))

    if len(paths) == 0:
        # no path at all
        return []
    
    #if len(paths) > 1:
        #print(f"longest paths: {[x for x in paths if x>0]}")
        #return paths
    
    return paths


def load_nodes(grid):
    # explore the grid, identifying all branchpoints and tracing their connectivity
    # to make a (bi-di) weight matrix


    dists: dict[tuple[tuple, tuple], int] = {}
    visited: set[tuple] = {startpoint}

    # start at 1 step away from startpoint (to match problem description)
    nexts: list[tuple[tuple, tuple]] = [(startpoint, (startpoint[0] + 1, startpoint[1]))]


    while len(nexts) > 0:
        (origin, direction) = nexts.pop(0)
        (destination, dist, nns) = flood1(grid, origin, direction)
        dists[(origin, destination)] = dist
        dists[(destination, origin)] = dist
        if destination in visited:
            continue
        visited.add(destination)
        nexts.extend([(destination, n) for n in nns])

    #NODES = len(visited)
    #adjmat = np.zeros((NODES, NODES), dtype=int)
    

    return visited, dists

#grid_neighbors(grid, (4,3))
#x = flood1(grid, (20,19),(21,19))
#print(x)

#result = explore_dfs(grid, startpoint, startpoint, set(), 0)
#print(list(sorted(x-1 for x in result)))

nodes, edges = load_nodes(grid)

NNODES = len(nodes)
NODE_IDS = list(sorted(nodes))  # 0 -> NNODES

def nodeid(nodetuple):
    return NODE_IDS.index(nodetuple)

MAX_WEIGHT = 1000000
ADJ = np.full((NNODES, NNODES), MAX_WEIGHT, dtype=int)
START = nodeid(startpoint)
END = nodeid(endpoint)

for n in range(NNODES):
    ADJ[n,n] = 0

for ((n1, n2), dist) in edges.items():
    n1id = nodeid(n1)
    n2id = nodeid(n2)
    ADJ[n1id][n2id] = dist

CONNECTED = ADJ != MAX_WEIGHT

# Ok, ADJ is our adjacency matrix.

BEST_KNOWN = 0
# Let's try to do an optimized dfs using the adjacency matrix to find all
# unique paths
def all_paths(lastnodeix, visited, dist):
    # lastnodeix is the index of the last node visited
    # visited is a boolean array of nodes visited
    # dist is the distance travelled so far
    global BEST_KNOWN

    if lastnodeix == END:
        if dist > BEST_KNOWN:
            BEST_KNOWN = dist
            print(f"new best: {dist}")
        return [dist]
    
    result = []
    visited[lastnodeix] = True
    nexts = CONNECTED[lastnodeix] & ~visited
    for nodeix in nexts.nonzero()[0]:
        result.extend(all_paths(nodeix, visited, dist + ADJ[lastnodeix][nodeix]))
    
    visited[lastnodeix] = False
    return result

allpaths = all_paths(START, np.zeros(NNODES, dtype=bool), 0)
print(allpaths)
print(f"best is {max(allpaths)}")

# print(ADJ)
# print("---")
# def floyd_warshall():
#     adj = ADJ.copy()

#     adj = -adj
#     adj[adj == -MAX_WEIGHT] = MAX_WEIGHT

#     print(adj)

#     n = NNODES
#     for k in range(n):
#         for i in range(n):
#             for j in range(n):
#                 adj[i][j] = min(adj[i][j], adj[i][k] + adj[k][j])
#     return adj

# fw = floyd_warshall()
# print(fw)
# print(f"shortest path from start to end = {fw[nodeid(startpoint)][nodeid(endpoint)]}")

def test_load_nodes():
    nodes2 = set().union(*({e1, e2} for (e1, e2) in edges.keys()))
    assert nodes == nodes2
    assert all((e1 in nodes and e2 in nodes for (e1, e2) in edges.keys()))
    assert startpoint in nodes
    assert endpoint in nodes
    print(f"{len(nodes)} nodes and {len(edges)} edges")

test_load_nodes()

#breakpoint()