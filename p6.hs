sample = [(71530,940200)]
real = [(63789468,411127420471035)]

bestcharge duration = duration / 2
solveChargetime duration dist = 
    let a=bestcharge duration in 
        let  b=(sqrt $ duration * duration - 4 * dist) * 0.5 in 
            (a-b, a+b)
countChargetime (a,b) = 1 + floor b - ceiling a 
    - (if fromIntegral (floor b) == b then 1 else 0) 
    - (if fromIntegral (ceiling a) == a then 1 else 0)
