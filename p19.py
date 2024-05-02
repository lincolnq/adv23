samp = """px{a<2006:qkq,m>2090:A,rfg}
pv{a>1716:R,A}
lnx{m>1548:A,A}
rfg{s<537:gd,x>2440:R,A}
qs{s>3448:A,lnx}
qkq{x<1416:A,crn}
crn{x>2662:A,R}
in{s<1351:px,qqz}
qqz{s>2770:qs,m<1801:hdj,R}
gd{a>3333:R,R}
hdj{m>838:A,pv}

{x=787,m=2655,a=1222,s=2876}
{x=1679,m=44,a=2067,s=496}
{x=2036,m=264,a=79,s=2244}
{x=2461,m=1339,a=466,s=291}
{x=2127,m=1623,a=2188,s=1013}
"""

simple = """in{m>10:A,x<10:A,R}"""

from helpers import *
from dataclasses import dataclass
import ranges
import json


parser = Sections(
    workflows=Lines() ** (FancyRe(r"(\w+){(.*)}", 2) % (str, Split(","))),
    objects=Lines() ** FancyRe("{(.*)}", 1) ** Split(",") ** (Split("=") % (str, int))
)

#input = parser.parse(samp)
#input = parser.parse(simple)

input = parser.parse(open('p19inp.txt').read())

workflows = dict(input['workflows'])

MAX = 4000
RR = ranges.Range(1, MAX+1)
NONE = {k: ranges.RangeSet() for k in 'xmas'}

Ranges = dict[str, ranges.RangeSet]
Result = Ranges

def union_ranges(ranges1: Ranges, ranges2: Ranges) -> Result:
    #return {x: ranges1[x].union(ranges2[x]) for x in 'xmas'}
    if isinstance(ranges1, list) and isinstance(ranges2, list):
        res = ranges1 + ranges2
    elif isinstance(ranges1, list):
        res = ranges1 + [ranges2]
    elif isinstance(ranges2, list):
        res = [ranges1] + ranges2
    else:
        res = [ranges1, ranges2]

    # try to collapse elements of res - filter out the nulls
    res = [r for r in res if elemcount(r) > 0]
    return res

def elemcount(ranges):
    p = 1
    for r in ranges.values():
        if r.isempty():
            return 0
        for range in r:
            p *= range.end - range.start
    
    return p


def traverse_wf_dest(dest: str, ranges0: Ranges) -> Result:
    # yay, leaf node
    if dest == 'A':
        return ranges0
    elif dest == 'R':
        return NONE
    else:
        return traverse_wf(workflows[dest], ranges0)

def divide_ranges(ranges0: Ranges, cond: str) -> tuple[Ranges, Ranges]:
    """Divide ranges into two, based on the condition"""
    var, comparator, val = cond[0], cond[1], int(cond[2:])

    if comparator == '>':
        target_range = ranges.Range(val+1, MAX+1)
    elif comparator == '<':
        target_range = ranges.Range(1, val)

    # get result from substep
    iftrue = ranges0.copy()
    iftrue[var] = iftrue[var].intersection(target_range)

    iffalse = ranges0.copy()
    iffalse[var] = iffalse[var].difference(target_range)

    #print(f"Dividing {ranges0} by {cond}, true range={target_range}")

    return (iftrue, iffalse)


# recursively traverse the workflow in order
def traverse_wf(wf: list[str], ranges0: Ranges) -> Result:
    stmt = wf[0]
    rest = wf[1:]

    if len(rest) == 0:
        # must be a pure 'dest' statement
        return traverse_wf_dest(stmt, ranges0)

    # otherwise, traverse a conditional
    cond, dest = stmt.split(":")

    # divide ranges and union iftrue with the recursive call
    ranges_iftrue, ranges_iffalse = divide_ranges(ranges0, cond)
    result_true = traverse_wf_dest(dest, ranges_iftrue)
    result_false = traverse_wf(rest, ranges_iffalse)
    return union_ranges(result_true, result_false)

BEGIN = {k: ranges.RangeSet([RR]) for k in 'xmas'}
result = traverse_wf(workflows['in'], BEGIN)
print(result)

cts = [elemcount(x) for x in result]
print(cts)
print(sum(cts))

