readInt :: String -> Int
readInt = read


parse = map (map readInt) . map words . lines

loadSamp = do
    txt <- readFile "p9samp.txt"
    return $ parse txt

samp :: [[Int]] = [[0,3,6,9,12,15],[1,3,6,10,15,21],[10,13,16,21,30,45]]

loadReal = do
    txt <- readFile "p9inp.txt"
    return $ parse txt

diffs :: [Int] -> [Int]
diffs xs = zipWith (-) (tail xs) xs

extrapolate :: [Int] -> Int
extrapolate xs | all (==0) xs = 0
               | otherwise = last xs + extrapolate (diffs xs)
    
part1 = sum . map extrapolate

extrapolateBack :: [Int] -> Int
extrapolateBack xs | all (==0) xs = 0
               | otherwise = head xs - extrapolateBack (diffs xs)

part2 = sum . map extrapolateBack