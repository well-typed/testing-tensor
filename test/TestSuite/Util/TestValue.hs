-- | Test values
--
-- These are doubles with modified 'Eq' and 'Show' instances.
--
-- Intended for unqualified import.
module TestSuite.Util.TestValue (
    TestValue -- opaque
  ) where

import Test.QuickCheck
import Text.Printf (printf)

newtype TestValue = TestValue Double
  deriving newtype (
      Fractional
    , Num
    , Real
    )

instance Arbitrary TestValue where
  -- we restrict the range so that 'epsilon' is relevant
  arbitrary = TestValue <$> choose (0, 1)
  shrink (TestValue x) = TestValue <$> shrink x

instance Show TestValue where
  show (TestValue x) = printf "%0.2f" x

instance Eq TestValue where
  TestValue x == TestValue y = abs (x - y) < epsilon

instance Ord TestValue where
  compare (TestValue x) (TestValue y)
    | abs (x - y) < epsilon = EQ
    | x < y                 = LT
    | otherwise             = GT

-- | Threshold for considering values to be equal
--
-- This is a rather crude way of comparing floating point numbers, but it's
-- good enough for our purposes here. This value of epsilon is also reflected
-- in how we print values ('Show' instance).
epsilon :: Double
epsilon = 0.01

