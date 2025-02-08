-- | Examples from the 3Blue1Brown video on convolutions
--
-- See "But what is a convolution?", <https://www.youtube.com/watch?v=KuXjwB4LzSA>
module TestSuite.Test.Convolution.Examples3B1B (
    -- * Simple example
    simpleInput
  , simpleKernel
  , simpleResult
    -- * Weighted dice
  , weightedDiceInput
  , weightedDiceKernel
  , weightedDiceResult
    -- * Moving average
  , movingAverageInput
  , movingAverageKernel
  , movingWeightedAverageKernel
  , movingAverageResult
  , movingWeightedAverageResult
  ) where

{-------------------------------------------------------------------------------
  Simple example

  In this example the input/kernel distinction is somewhat artificial.
  We rotate the kernel.
-------------------------------------------------------------------------------}

simpleInput :: Num a => [a]
simpleInput = [1, 2, 3]

simpleKernel :: Num a => [a]
simpleKernel = reverse [4, 5, 6]

simpleResult :: Num a => [a]
simpleResult = [4, 13, 28, 27, 18]

{-------------------------------------------------------------------------------
  Weighted dice

  Same comments as for the simple example apply.
-------------------------------------------------------------------------------}

weightedDiceInput :: Fractional a => [a]
weightedDiceInput = [0.03, 0.11, 0.23, 0.29, 0.23, 0.11]

weightedDiceKernel :: Fractional a => [a]
weightedDiceKernel = reverse [0.46, 0.20, 0.12, 0.09, 0.07, 0.05]

weightedDiceResult :: Fractional a => [a]
weightedDiceResult = [
      0.01 -- 2
    , 0.06 -- 3
    , 0.13 -- 4
    , 0.20 -- 5
    , 0.21 -- 6
    , 0.16 -- 7
    , 0.10 -- 8
    , 0.07 -- 9
    , 0.04 -- 10
    , 0.02 -- 11
    , 0.01 -- 12
    ]

{-------------------------------------------------------------------------------
  Moving average
-------------------------------------------------------------------------------}

movingAverageInput :: Fractional a => [a]
movingAverageInput = concat [
      replicate 5 0.1
    , replicate 5 1.0
    , replicate 5 0.1
    , replicate 5 1.0
    , replicate 5 0.1
    ]

movingAverageKernel :: Fractional a => [a]
movingAverageKernel = [0.2, 0.2, 0.2, 0.2, 0.2]

movingWeightedAverageKernel :: Fractional a => [a]
movingWeightedAverageKernel = [0.1, 0.2, 0.4, 0.2, 0.1]

movingAverageResult :: Fractional a => [a]
movingAverageResult = [
      0.06, 0.08, 0.10, 0.28, 0.46
    , 0.64, 0.82, 1.00, 0.82, 0.64
    , 0.46, 0.28, 0.10, 0.28, 0.46
    , 0.64, 0.82, 1.00, 0.82, 0.64
    , 0.46, 0.28, 0.10, 0.08, 0.06
    ]

movingWeightedAverageResult :: Fractional a => [a]
movingWeightedAverageResult = [
      0.07, 0.09, 0.10, 0.19, 0.37
    , 0.73, 0.91, 1.00, 0.91, 0.73
    , 0.37, 0.19, 0.10, 0.19, 0.37
    , 0.73, 0.91, 1.00, 0.91, 0.73
    , 0.37, 0.19, 0.10, 0.09, 0.07
    ]
