{-# LANGUAGE CPP #-}

module Main (main) where

import Test.Tasty

import TestSuite.Test.Convolution qualified as Convolution
import TestSuite.Test.QuickCheck  qualified as QuickCheck
import TestSuite.Test.StdOps      qualified as StdOps

#ifdef TEST_FFT
import TestSuite.Test.Convolution.FFT qualified as Convolution.FFT
#endif

#ifdef TEST_CUDNN
import TestSuite.Test.Convolution.CUDNN qualified as Convolution.CUDNN
#endif

main :: IO ()
main = defaultMain $ testGroup "testing-tensor" [
      testGroup "Convolutions" [
          QuickCheck.tests
        , StdOps.tests
        , Convolution.tests
#ifdef TEST_FFT
        , Convolution.FFT.tests
#endif
#ifdef TEST_CUDNN
        , Convolution.CUDNN.tests
#endif
      ]
    ]
