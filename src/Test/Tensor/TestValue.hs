-- | Test values
--
-- Intended for unqualified import.
module Test.Tensor.TestValue (
    TestValue -- opaque
  ) where

import Test.QuickCheck
import Text.Printf (printf)

{-------------------------------------------------------------------------------
  Definition
-------------------------------------------------------------------------------}

-- | Test values
--
-- Test values are 'Float' values with a crude equality:
--
-- >               (==)
-- > --------------------
-- > 1.0    1.1    False
-- > 1.00   1.01   True
-- > 10     11     False
-- > 10.0   10.1   True
-- > 100    110    False
-- > 100    101    True
newtype TestValue = TestValue Float
  deriving newtype (Num, Fractional, Real)
  deriving newtype (Arbitrary)

instance Show TestValue where
  show (TestValue x) = printf "%0.2f" x

instance Eq TestValue where
  TestValue x == TestValue y = nearlyEqual x y

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
