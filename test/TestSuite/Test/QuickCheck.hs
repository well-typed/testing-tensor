-- | Meta-tests: test the Tensor QuickCheck infrastructure
module TestSuite.Test.QuickCheck (tests) where

import Data.Type.Nat
import Test.Tasty
import Test.Tasty.HUnit

import Test.Tensor qualified as Tensor

{-------------------------------------------------------------------------------
  List of tests
-------------------------------------------------------------------------------}

tests :: TestTree
tests = testGroup "TestSuite.Test.QuickCheck" [
      testGroup "Examples" [
          testCase "shrinkWith" example_shrinkWith
        ]
    ]

{-------------------------------------------------------------------------------
  Examples
-------------------------------------------------------------------------------}

example_shrinkWith :: Assertion
example_shrinkWith =
    assertEqual "" expected $
      Tensor.shrinkWith (const [0]) (Tensor.dim2 [[1,2,3], [4,5,6]])
  where
    expected :: [Tensor.Tensor Nat2 Int]
    expected = [
        -- Shrink outer dimension
          Tensor.dim2 [[4,5,6]]
        , Tensor.dim2 [[1,2,3]]
         -- Shrink inner dimension
        , Tensor.dim2 [[2,3],[5,6]]
        , Tensor.dim2 [[1,3],[4,6]]
        , Tensor.dim2 [[1,2],[4,5]]
          -- Shrink one of the elements
        , Tensor.dim2 [[0,2,3],[4,5,6]]
        , Tensor.dim2 [[1,0,3],[4,5,6]]
        , Tensor.dim2 [[1,2,0],[4,5,6]]
        , Tensor.dim2 [[1,2,3],[0,5,6]]
        , Tensor.dim2 [[1,2,3],[4,0,6]]
        , Tensor.dim2 [[1,2,3],[4,5,0]]
        ]
