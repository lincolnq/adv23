samp = """???.### 1,1,3
.??..??...?##. 1,1,3
?#?#?#?#?#?#?#? 1,3,1,6
????.#...#... 4,1,1
????.######..#####. 1,6,5
?###???????? 3,2,1"""

full = open('p12inp.txt').read()

from helpers import *

parser = Lines() ** Sections(' ', pat=Id, seq=(Split(',') ** int))
result = parser.parse(full)
print(result)

# import z3
# x = z3.Int('x')

# def solver_for_line(line):
#     s = z3.Solver()

#     linelen = len(line['pat'])
#     nvars = len(line['seq'])
#     # create start/end variables for each starting index.
#     # start var = index of first #
#     # end var = 1 + index of last #
#     vars = []
#     prevend = None
#     for i in range(nvars):
#         ilen = line['seq'][i]
#         startvar = z3.Int(f"S{i}")
#         endvar = z3.Int(f"E{i}")
#         s.add(startvar >= 0)
#         s.add(endvar <= linelen)
#         s.add(endvar - startvar == ilen)

#         # constrain end of previous one must be separated from start of this one by >=1
#         if prevend is not None:
#             s.add(startvar - prevend > 0)
#         prevend = endvar
#         vars.append((startvar, endvar))
    
#     # now add constraints derived from the pattern
#     for (i,c) in enumerate(line['pat']):
#         if c == '#':
#             block = []
#             # some (Start,end) range can overlap this index
#             for (start,end) in vars:
#                 block.append(z3.And(start <= i, i < end))
#             s.add(z3.Or(block))
#         elif c == '.':
#             block = []
#             # no (Start,end) range can overlap this index
#             for (start,end) in vars:
#                 block.append(z3.And(start <= i, i < end))
#             s.add(z3.Not(z3.Or(block)))

#     return s

# def count_solutions(solver):
#     solver.simplify()
#     count = 0
#     while solver.check() == z3.sat:
#         count += 1
#         model = solver.model()
#         #print(model)
#         block = []
#         for d in model.decls():
#             block.append(d() != model[d])
#         solver.add(z3.Or(block))
#     return count


# total = 0
# for line in result:
#     sols = count_solutions(solver_for_line(line))
#     total += sols
#     print(f"{line}: {sols}")


SOLCOUNT_MEMO = {}

def solutionCount(pat, seq):
    # base case: there are no more sequence items
    if len(seq) == 0:
        # if no more '#' then success, otherwise fail
        if '#' not in pat:
            return 1        
        else:
            return 0

    # also, speed things up a bit by checking basic total length constraint
    if len(pat) < sum(seq) + len(seq) - 1:
        return 0
    
    # check the memo
    if (pat, tuple(seq)) in SOLCOUNT_MEMO:
        return SOLCOUNT_MEMO[(pat,tuple(seq))]
    
    # skip leading dots since we won't place items there.
    pat = pat.lstrip('.')

    #print(f"Trying to solve {pat} with {seq}")

    solutions = []
    
    # First, we attempt to place the first item in 'seq' in the first block-group-area
    mylen = seq[0]
    firstarea = firstBlockArea(pat)

    if len(firstarea) >= mylen:
        # ok, it fits here but in how many places?
        # it must end before the first '.' defined by 'firstarea'
        spotcount = len(firstarea) - mylen + 1

        # also, it must start at or before the first '#'
        firsthash = firstarea.find('#')
        if firsthash != -1:
            spotcount = min(spotcount, firsthash + 1)

        for offset in range(spotcount):
            pat_remain = pat[offset+mylen:]
            newseq = seq[1:]

            # require a '.' or '?' next (unless this is last seq)
            if len(newseq):
                if len(pat_remain) == 0 or pat_remain[0] not in '.?':
                    # we have no more pattern to match or it's not a valid separator
                    continue
                # trim the pattern one more character to consume the space
                pat_remain = pat_remain[1:]

            # recur on the rest of the pattern and remaining seq items
            solutions.append(solutionCount(pat_remain, newseq))
        
    # We now consider the possibility of skipping the first area. This is only 
    # possible if there are no '#' - in other words, if it is all question marks.
    if '#' not in firstarea:
        solutions.append(solutionCount(pat[len(firstarea):], seq))

    #print(f"in this recurrence of ({pat},{seq}) we had the following solutions: {solutions}")

    # save the result in the memo
    result = sum(solutions)
    SOLCOUNT_MEMO[(pat, tuple(seq))] = result
    return result


# Determine first possible area to place a block
def firstBlockArea(pat):
    # get subpattern from the start until first '.'
    firstdot = pat.find('.')
    # if there is no '.' then we fit in the rest
    if firstdot == -1:
        return pat
    return pat[:firstdot]

total = 0
for line in result:
    line5x = '?'.join(line['pat'] for _ in range(5)), line['seq'] * 5
    sols = solutionCount(*line5x)
    print(f"{line5x}: {sols}")
    total += sols
    

print(f"Total {total}")

"""
?###????????  
.###.##.#... 3,2,1
.###.##..#.. 3,2,1
.###.##...#. 3,2,1
.###.##....# 3,2,1
.###..##.#.. 3,2,1
.###..##..#. 3,2,1
.###..##...# 3,2,1
.###...##.#. 3,2,1
.###...##..# 3,2,1
.###....##.# 3,2,1
"""