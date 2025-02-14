-- | Test values
--
-- Intended for unqualified import.
module Test.Tensor.TestValue (
    TestValue -- opaque
  ) where

import Data.List (sort)
import Test.QuickCheck
import Text.Printf (printf)

{-------------------------------------------------------------------------------
  Definition
-------------------------------------------------------------------------------}

-- | Test values
--
-- Test values are suitable for use in QuickCheck tests involving floating
-- point numbers, if you want to ignore rounding errors.
newtype TestValue = TestValue Float
  deriving newtype (Num, Fractional, Real)

-- | Test values are equipped with a crude equality
--
-- >               (==)
-- > --------------------
-- > 1.0    1.1    False
-- > 1.00   1.01   True
-- > 10     11     False
-- > 10.0   10.1   True
-- > 100    110    False
-- > 100    101    True
instance Eq TestValue where
  TestValue x == TestValue y = nearlyEqual x y

-- | Show instance
--
-- We have more precision available for smaller values, so we show more
-- decimals. However, larger values the show instance does not reflect the
-- precision: @1000@ and @1001@ are shown as @1000@ and @1001@, even though
-- they are considered to be equal.
--
-- > show @TestValue 0     == "0"     -- True zero
-- > show @TestValue 0.001 == "0.00"
-- > show @TestValue 0.009 == "0.01"
-- > show @TestValue 1.001 == "1.0"
-- > show @TestValue 11    == "11"
instance Show TestValue where
  show (TestValue x)
    | x == 0    = "0"
    | x <  1    = printf "%0.2f" x
    | x <  10   = printf "%0.1f" x
    | otherwise = printf "%0.0f" x

-- | Arbitrary instance
--
-- The definition of 'arbitrary' simply piggy-backs on the definition for
-- 'Float', but in shrinking we avoid generating nearly equal values, and prefer
-- values closer to integral values. Compare:
--
-- >    shrink @TestValue 100.1
-- > == [0,50,75,88,94,97]
--
-- versus
--
-- >    shrink @Float 100.1
-- > == [100.0,0.0,50.0,75.0,88.0,94.0,97.0,99.0,0.0,50.1,75.1,87.6,93.9,97.0,98.6,99.4,99.8,100.0]
instance Arbitrary TestValue where
  arbitrary = TestValue <$> arbitrary

  shrink (TestValue x)
    | x == 0          = []
    | nearlyEqual x 0 = [0]
    | otherwise       = case sort (shrink x) of
                          []   -> []
                          y:ys -> aux y ys
    where
      aux :: Float -> [Float] -> [TestValue]
      aux y []
        | nearlyEqual y x = []
        | otherwise       = [TestValue y]
      aux y (z:zs)
        | nearlyEqual y z = if decimalPart y < decimalPart z
                              then aux y zs
                              else aux z zs
        | otherwise       = TestValue y : aux z zs

instance Ord TestValue where
  compare (TestValue x) (TestValue y)
    | nearlyEqual x y = EQ
    | x < y           = LT
    | otherwise       = GT

{-------------------------------------------------------------------------------
  Internal auxiliary
-------------------------------------------------------------------------------}

-- | Compare for near equality
--
-- Adapted from <https://stackoverflow.com/a/32334103/742991>
nearlyEqual :: Float -> Float -> Bool
nearlyEqual a b
  | a == b    = True
  | otherwise = diff < max abs_th (epsilon * norm)
  where
    diff, norm :: Float
    diff = abs (a - b)
    norm = abs a + abs b

    -- Define precision
    abs_th, epsilon :: Float
    epsilon = 0.01
    abs_th  = 0.01

decimalPart :: Float -> Float
decimalPart x = x - fromIntegral (floor x :: Int)