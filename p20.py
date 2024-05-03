samp = """broadcaster -> a, b, c
%a -> b
%b -> c
%c -> inv
&inv -> a"""

samp2 = """broadcaster -> a
%a -> inv, con
&inv -> b
%b -> con
&con -> output"""

from helpers import *
from dataclasses import dataclass

parser = Lines() ** (FancyRe(r"(.*) -> (.*)", 2) % (str, Split(", ")))
#input = parser.parse(samp2)
input = parser.parse(open('p20inp.txt').read())
output_node = 'rx'

print(input)

HIGH = True
LOW = False

@dataclass
class Node:
    outputs: list[str]
    name: str
    inputs: list[str]

    def receive(self, signal, source):
        # FILL ME IN
        return []

    def pulse(self, signal):
        return [(self.name, signal, o) for o in self.outputs]
    
    @staticmethod
    def make(name, outputs):
        if name == 'broadcaster':
            return Broadcaster(outputs=outputs, name=name, inputs=[])
        elif name.startswith('%'):
            return FlipFlop(outputs=outputs, name=name[1:], inputs=[])
        elif name.startswith('&'):
            return Conjunction(outputs=outputs, name=name[1:], inputs=[], last_inputs={})
        else:
            # generic output node
            return Node(outputs=outputs, name=name, inputs=[])
    
    def add_input(self, input):
        self.inputs.append(input)

    def connect_inputs(self, all_nodes):
        # back-connect our outputs to inputs
        for o in self.outputs:
            all_nodes[o].add_input(self.name)

@dataclass
class Broadcaster(Node):
    def receive(self, signal, source):
        return self.pulse(signal)


@dataclass
class FlipFlop(Node):
    last: bool = False

    def receive(self, signal, source):
        if signal == HIGH:
            return []
        self.last = not self.last
        return self.pulse(self.last)

@dataclass
class Conjunction(Node):
    last_inputs: dict[str, bool]

    def add_input(self, input):
        super().add_input(input)
        self.last_inputs[input] = LOW

    def receive(self, signal, source):
        """Conjunction modules (prefix &) remember the type of the most recent pulse 
        received from each of their connected input modules; they initially default 
        to remembering a low pulse for each input. 
        
        When a pulse is received, the conjunction module first updates its memory 
        for that input. Then, if it remembers high pulses for all inputs, it sends 
        a low pulse; otherwise, it sends a high pulse."""
        assert len(self.last_inputs) == len(self.inputs)
        self.last_inputs[source] = signal
        if all(self.last_inputs.values()):
            return self.pulse(LOW)
        else:
            return self.pulse(HIGH)
        
def init_nodes():
    nodes = {}
    for (name, outputs) in input:
        n = Node.make(name, outputs)
        nodes[n.name] = n

    if output_node not in nodes:
        nodes[output_node] = Node(outputs=[], name=output_node, inputs=[])

    for n in nodes.values():
        n.connect_inputs(nodes)

    return nodes

def button(c, nodes, counts):
    pulses = [('button', LOW, 'broadcaster')]

    while pulses:
        (src, sig, dest) = pulses.pop(0)
        counts[sig] += 1
        node = nodes[dest]
        #print(f"{src} -{sig}-> {dest}")
        if src.startswith('conj') and sig == LOW:
            print(f"node {src} output low signal at buttonpress {c}")
        nextpulses = node.receive(sig, src)
        pulses.extend(nextpulses)

    #print(counts)
cluster_4 = "bx fx sk hr ff xc nx fn kn mv fk rv conj4" # 3907 steps
# did the `lcm` for all 4 clusters and it was right!


nodes = init_nodes()
def run(count):
    counts = {True: 0, False: 0}
    for i in range(count):
        button(i+1, nodes, counts)

        # show status of flip-flops
        #flips = [nodes[n] for n in cluster_1.split() if isinstance(nodes[n], FlipFlop)]
        #print(''.join(['#' if f.last else '.' for f in flips]))

run(10000)

#print(counts)
#print(counts[True] * counts[False])

# next step: get a single LOW pulse to the output node
# work backwards

output = nodes[output_node]

def print_inputs(node, maxdepth, depth):
    print(' ' * (maxdepth - depth), node.name, type(node), f"{len(node.inputs)} inputs:")
    if depth == 0:
        return
    for i in node.inputs:
        print_inputs(nodes[i], maxdepth, depth-1)

#print_inputs(output, 4, 4)


# print("digraph G {")
# for node in nodes.values():
#     for outp in node.outputs:
#         print(f"  {node.name} -> {outp};")
# print("}")