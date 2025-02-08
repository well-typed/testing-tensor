-- | Test kernels
--
-- For testing purposes it's very useful to be able to specify at the type level
-- the exact size of kernel we want (not just its dimension).
--
-- Notes:
--
-- * Use sites will always pick a specific size, so we're not worried here about
--   stuck type families etc.
-- * Size we specify the size of the kernel at the type level, the size of the
--   kernel does not shrink (in the 'Arbitrary' instance).
--
-- Intended for unqualified import.
module TestSuite.Util.TestKernel (
    TestKernel -- opaque
  , testKernel
  ) where

import Data.Kind
import Data.Type.Nat
import Data.Vec.Lazy (Vec(..))
import Data.Vec.Lazy qualified as Vec
import Test.QuickCheck

import Test.Tensor (Tensor(..))

{-------------------------------------------------------------------------------
  Definition
-------------------------------------------------------------------------------}

data TestKernel :: [Nat] -> Type -> Type where
  TKZ :: a -> TestKernel '[] a
  TKS :: Vec n (TestKernel ns a) -> TestKernel (n : ns) a

instance Show a => Show (TestKernel ns a) where
  show = show . testKernel

{-------------------------------------------------------------------------------
  Conversion
-------------------------------------------------------------------------------}

type family Length (as :: [k]) where
  Length '[]    = Z
  Length (x:xs) = S (Length xs)

testKernel :: TestKernel ns a -> Tensor (Length ns) a
testKernel (TKZ x)  = Scalar x
testKernel (TKS xs) = Tensor $ map testKernel (Vec.toList xs)

{-------------------------------------------------------------------------------
  Arbitrary instance
-------------------------------------------------------------------------------}

instance Arbitrary a => Arbitrary (TestKernel '[] a) where
  arbitrary      = TKZ <$> arbitrary
  shrink (TKZ x) = TKZ <$> shrink x

instance (SNatI n, Arbitrary (TestKernel ns a))
      => Arbitrary (TestKernel (n : ns) a) where
  arbitrary       = TKS <$> liftArbitrary arbitrary
  shrink (TKS xs) = TKS <$> shrink xs

