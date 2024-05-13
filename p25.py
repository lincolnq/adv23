samp = """jqt: rhn xhk nvd
rsh: frs pzl lsr
xhk: hfx
cmg: qnr nvd lhk bvb
rhn: xhk bvb hfx
bvb: xhk hfx
pzl: lsr hfx nvd
qnr: nvd
ntq: jqt hfx bvb xhk
nvd: lhk
lsr: lhk
rzs: qnr cmg lsr rsh
frs: qnr lhk lsr
"""

from helpers import *
from collections import defaultdict
import networkx as nx

parser = Lines() ** Split(":") ** Split()
#lines = parser.parse(samp)
lines = parser.parse(open("p25inp.txt").read())

G = nx.Graph()
#edges = defaultdict(set)

for (k,), vs in lines:
    for v in vs:
        G.add_edge(k, v)

print(f"Number of nodes: {len(G.nodes)}")

mincut = nx.minimum_edge_cut(G)

print(f"Minimum cut: {mincut}")

# now delete them and find components
for edge in mincut:
    G.remove_edge(*edge)

comps = list(nx.connected_components(G))
print(f"Number of components: {len(comps)}")
print(f"Product of size of first 2 components: {len(comps[0]) * len(comps[1])}")