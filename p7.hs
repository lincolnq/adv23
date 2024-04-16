import Data.List
import Data.Maybe
import Data.Function
--import Control.Exception

data Card = Joker | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Q | K | A
    deriving (Eq, Ord, Enum, Bounded, Show)

printCard :: Card -> String
printCard = pure . ("J23456789TQKA" !!) . fromEnum

readCard :: Char -> Card
readCard = toEnum . fromJust . (`elemIndex` "J23456789TQKA")

data Hand = Hand { cards :: [Card] }
    deriving (Eq, Show)

printHand :: Hand -> String
printHand (Hand h) = concatMap printCard h

readHand :: String -> Hand
readHand = Hand . map readCard


data Rank = HighCard | OnePair | TwoPair | ThreeOfAKind | FullHouse | FourOfAKind | FiveOfAKind
    deriving (Eq, Ord, Show)

rankHand :: Hand -> Rank
rankHand (Hand cs) | length cs == 5 = rankHand5 cs
    | otherwise = error "Hand is not of length 5"

rankHand5 :: [Card] -> Rank
rankHand5 cs 
    | countJokers == 5 = FiveOfAKind
    | countFirst == 5 = FiveOfAKind
    | countFirst == 4 = FourOfAKind
    | countFirst == 3 && countSecond == 2 = FullHouse
    | countFirst == 3 = ThreeOfAKind
    | countFirst == 2 && countSecond == 2 = TwoPair
    | countFirst == 2 = OnePair
    | otherwise = HighCard
    where
        countJokers = length $ filter (== Joker) cs
        countFirst = counts !! 0 + countJokers
        countSecond = counts !! 1
        counts = reverse $ sort $ map length $ group $ sort $ filter (/= Joker) cs

instance Ord Hand where
    compare h1 h2 
        | rh1 == rh2 = compare (cards h1) (cards h2)
        | otherwise = compare rh1 rh2
        where
            rh1 = rankHand h1
            rh2 = rankHand h2


assert :: Bool -> IO ()
assert x = do
    if x then
        return ()
    else
        error "Test failed"

runTests = do
    assert (rankHand (readHand "23456") == HighCard)
    assert (rankHand (readHand "23455") == OnePair)
    assert (rankHand (readHand "66A22") == TwoPair)
    assert (rankHand (readHand "A6222") == ThreeOfAKind)
    assert (rankHand (readHand "66222") == FullHouse)
    assert (rankHand (readHand "66A66") == FourOfAKind) 
    assert (rankHand (readHand "66666") == FiveOfAKind) 
    assert (readHand "AAAAA" > readHand "AAAAK")
    assert (readHand "AAAAK" > readHand "AAAKA")
    assert (readHand "AAAAK" == readHand "AAAAK")
    assert (readHand "33332" > readHand "2AAAA")
    assert (readHand "77888" > readHand "77788")

    -- joker tests
    assert (rankHand (readHand "JJJJJ") == FiveOfAKind) 
    assert (rankHand (readHand "T55J5") == FourOfAKind)
    assert (rankHand (readHand "KTJJT") == FourOfAKind)
    assert (rankHand (readHand "QQQJA") == FourOfAKind)
    assert (sort (map readHand ["T55J5", "KTJJT", "QQQJA"]) == map readHand ["T55J5", "QQQJA", "KTJJT"])


parse fn = do
    rows <- fmap (fmap words . lines) $ readFile fn
    let res = [(readHand h, read bid::Int) | h:bid:_ <- rows]
    --putStrLn $ show $ res
    return res

readSamp = parse "p7samp.txt"
readReal = parse "p7inp.txt"

result hs = sum [ rank * bid | (rank, (_, bid)) <- zip [1..] (sort hs)]