samp = """
#.##..##.
..#.##.#.
##......#
##......#
..#.##.#.
..##..##.
#.#.##.#.

#...##..#
#....#..#
..##..###
#####.##.
#####.##.
..##..###
#....#..#"""

from helpers import *
import numpy as np

parser = Split('\n\n') ** Matrix()
realinp = open('p13inp.txt').read()
result = parser.parse(realinp)

def reflect(matrix):
    # returns the 'line of reflection' if matrix is vertically reflected anywhere
    for line in range(2, matrix.shape[0] + 1, 2):
        print(f"comparing matrix of first {line} rows")
        submatrix = matrix[:line]
        print(submatrix)
        #if (submatrix == submatrix[::-1]).all():
        #    print(f"found @ {line//2}")
        #    return line // 2
        
        # otherwise, count the number of non-matching items in the reflection
        nonmatches = np.count_nonzero(submatrix != submatrix[::-1])
        print(f"almost matches = {nonmatches}")
        if nonmatches == 2:
            print(f"found glitch @ {line//2}")
            return line // 2


        
    return None

def updownReflect(matrix):
    rnorm = reflect(matrix)
    if rnorm is not None:
        return rnorm
    
    rupsidedown = reflect(matrix[::-1])
    if rupsidedown is not None:
        #print(f"upsidedown returned {rupsidedown}, size is {matrix.shape[0]}")
        return matrix.shape[0] - rupsidedown

def anyReflect(matrix):
    rupdown = updownReflect(matrix)
    if rupdown is not None:
        return (rupdown, 0)
    
    rleftright = updownReflect(matrix.T)
    if rleftright is not None:
        return (0, rleftright)

result = sum(np.array(anyReflect(x)) for x in result)
print(result[0] * 100 + result[1])


