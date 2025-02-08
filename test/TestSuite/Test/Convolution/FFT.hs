-- | Test against a reference implementation using fast fourier transforms
--
-- We do this only for 1D tensors.
module TestSuite.Test.Convolution.FFT (tests) where

import Data.Array.CArray (CArray)
import Data.Array.IArray (IArray)
import Data.Array.IArray qualified as IA
import Data.Complex (Complex)
import Data.Ix (Ix)
import Data.Type.Nat
import Math.FFT qualified as FFT
import Test.Tasty
import Test.Tasty.HUnit
import Test.Tasty.QuickCheck

import Test.Tensor qualified as Tensor

import TestSuite.Test.Convolution.Examples3B1B
import TestSuite.Util.TestKernel
import TestSuite.Util.TestValue

{-------------------------------------------------------------------------------
  List of testse
-------------------------------------------------------------------------------}

tests :: TestTree
tests = testGroup "Test.Convolution.FFT" [
      testGroup "Examples" [
          testCase "weightedMovingAverage" example_weightedMovingAverage
        ]
    , testGroup "Properties" [
          testGroup "matchesModel" [
              testProperty "kernelSize3" $ prop_matchesModel @Nat3
            , testProperty "kernelSize4" $ prop_matchesModel @Nat4
            , testProperty "kernelSize5" $ prop_matchesModel @Nat5
            ]
        ]
    ]

{-------------------------------------------------------------------------------
  Examples
-------------------------------------------------------------------------------}

example_weightedMovingAverage :: Assertion
example_weightedMovingAverage =
    assertEqual "" (movingWeightedAverageResult @TestValue) $
      removePadding 2 $
        convolveFFT
          movingWeightedAverageKernel
          (movingAverageInput @TestValue)

{-------------------------------------------------------------------------------
  Properties
-------------------------------------------------------------------------------}

-- | Compare our implementation against FFT implementation
prop_matchesModel :: forall n.
     TestKernel '[n] TestValue  -- ^ Kernel
  -> NonEmptyList TestValue     -- ^ Input
  -> Property
prop_matchesModel (testKernel -> kernel) (getNonEmpty -> input) =
        convolveFFT (reverse $ Tensor.toLists kernel) input
    === ( Tensor.toLists $
            Tensor.convolve
              kernel
              (Tensor.padWith 0 (length kernel - 1) $ Tensor.dim1 input)
        )

{-------------------------------------------------------------------------------
  Convolution implementation using FFT

  FFT requires an input of even length, so if the input has odd length, we add
  an additional zero padding byte.
-------------------------------------------------------------------------------}

-- | Compute convolution using FFT
convolveFFT :: forall a. (Fractional a, Real a) => [a] -> [a] -> [a]
convolveFFT kernel input_ =
    adjustOutput needOddAdjustment $ map realToFrac $ IA.elems inv
  where
    needOddAdjustment :: Bool
    needOddAdjustment = odd (length input_ + length kernel - 1)

    input :: [a]
    input = adjustInput needOddAdjustment input_

    n, m :: Int
    n = length input
    m = length kernel

    arrInput, arrKernel :: CArray Int Double
    arrInput  = paddedArrayFromList (m + n - 1) (map realToFrac input)
    arrKernel = paddedArrayFromList (m + n - 1) (map realToFrac kernel)

    dftInput, dftKernel, dftMult :: CArray Int (Complex Double)
    dftInput  = FFT.dftRC arrInput
    dftKernel = FFT.dftRC arrKernel
    dftMult   = zipArraySameBounds (*) dftInput dftKernel

    inv :: CArray Int Double
    inv = FFT.dftCR dftMult

adjustInput :: Num a => Bool -> [a] -> [a]
adjustInput True  = (:) 0
adjustInput False = id

adjustOutput :: Bool -> [a] -> [a]
adjustOutput True  = drop 1
adjustOutput False = id

removePadding :: Int -> [a] -> [a]
removePadding n xs = take (length xs - 2 * n) (drop n xs)

{-------------------------------------------------------------------------------
  Internal auxiliary: arrays
-------------------------------------------------------------------------------}

paddedArrayFromList :: forall a e. (IArray a e, Num e)
  => Int  -- ^ Decided length of the array
  -> [e]  -- ^ List to initialize the array from
  -> a Int e
paddedArrayFromList len xs = IA.listArray (0, len - 1) (xs ++ repeat 0)

zipArraySameBounds ::
     (IArray a x, IArray a y, IArray a z, Ix i)
  => (x -> y -> z)
  -> a i x -> a i y -> a i z
zipArraySameBounds f xs ys =
    IA.listArray (IA.bounds xs) [
        f (xs IA.! i) (ys IA.! i)
      | i <- IA.indices xs
      ]



