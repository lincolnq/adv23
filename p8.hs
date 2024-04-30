--parse3 :: String -> String -> (String, String, String)
--parse3 fmt = 

import Text.Parsec
import Data.Either
import qualified Data.Map as M
import qualified Data.Set as S
import Debug.Trace

row :: Stream s m Char => ParsecT s u m (String, (String, String))
row = ((,) <$> (many1 alphaNum <* string " = ") <*> 
    ((,) <$> (char '(' *> many1 alphaNum <* string ", ") <*> many1 alphaNum <* char ')'))


loadparse fn = do
    contents <- lines <$> readFile fn
    let rls = contents !! 0
    let map = M.fromList [fromRight (error "parse error") $ parse row "(input)" s | s <- drop 2 contents]

    return (rls, map)

fstsnd 'L' = fst
fstsnd 'R' = snd

advance map hlrs curr = fstsnd hlrs $ map M.! curr

-- old
travel _ _ "ZZZ" count = count
travel (hrls:rest) map curr count = travel rest map (advance map hrls curr) (count + 1)

--findLoop :: Int -> M.Map String (String, String) 
    -- -> String -> String -> Int -> S.Set (String, Int) 
    -- -> Int
findLoop baseLength map (hrls:rest) curr count visited zindexes
    | M.member ccm visited = 
        -- stepcount to reach 1st repeated state, offset of beginning of loop, loop length, zindexes
        (count, visited M.! ccm, count - visited M.! ccm, zindexes)
    | otherwise = findLoop baseLength map 
        rest 
        (advance map hrls curr)
        (count + 1)
        (M.insert ccm count visited)
        (if curr !! 2 == 'Z' then (zindexes ++ [count]) else zindexes)
    where ccm = (curr, count `mod` baseLength)

-- all stepcounts where this ghost steps on a 'z'-node
-- example:
-- ghost_zs x 1 2 [2] = [2, 4, 6, 8...]
-- ghost_zs x 2 2 [3] = [3, 5, 7, 9...]
-- ghost_zs x 2 2 [3,4] = [3, 4, 5, 6, 7...]
-- ghost_zs x 1 6 [3,6] = [3, 6, 9, 12...]
ghost_zs :: (Int, Int, Int, [Int]) -> [Int]
ghost_zs (_, offset, looplength, zindexes) = zipWith (+) replicated_loopadds (cycle zindexes)
    where
    loopadds = fmap (*looplength) [0..]
    replicated_loopadds = concatMap (replicate (length zindexes)) loopadds

allEqual :: Eq a => [a] -> Bool
allEqual (h:rest) = all (== h) rest

sim :: [[Int]] -> [Int]
sim ghosts 
    | allEqual (head <$> ghosts) = (head $ head ghosts):recur
    | otherwise = recur  -- drop lowest ghost head
    where 
    recur = 
        (if min `mod` 100000 == 0 
            then trace (show min) else id) $
        sim (map (dropWhile (== min)) ghosts)
    min = minimum $ head <$> ghosts

loopinfo (rls, map) = let starts = [x | x@(_:_:'A':[]) <- M.keys map] in
    [findLoop (length rls) map (cycle rls) x 0 M.empty [] | x <- starts]

runit x = 
    let ghosts = fmap ghost_zs (loopinfo x) in
    -- Lq note: This did not work, took too long
    sim ghosts
    -- The right answer was something on the order of 16 trillion. I had to
    -- simulate each pair of ghosts at a time (or really, up to 4 at a time).
    -- Then I could find the LCM of each pair.



parseSamp = loadparse "p8samp.txt"
parseReal = loadparse "p8inp.txt"