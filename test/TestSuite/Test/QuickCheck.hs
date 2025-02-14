-- | Meta-tests: test the Tensor QuickCheck infrastructure
module TestSuite.Test.QuickCheck (tests) where

import Data.Foldable qualified as Foldable
import Data.Type.Nat
import Data.Vec.Lazy (Vec(..))
import Test.Tasty
import Test.Tasty.HUnit
import Test.Tasty.QuickCheck

import Test.Tensor (Tensor)
import Test.Tensor qualified as Tensor

{-------------------------------------------------------------------------------
  List of tests
-------------------------------------------------------------------------------}

tests :: TestTree
tests = testGroup "TestSuite.Test.QuickCheck" [
      testGroup "Examples" [
          testCase "shrinkWith" example_shrinkWith
        ]
    , testGroup "Properties" [
          testProperty "allAxes_shrinkList" prop_allAxes_shrinkList
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

{-------------------------------------------------------------------------------
  Properties
-------------------------------------------------------------------------------}

-- | 'allAxes' essentially reifies the decisions made by 'shrinkList'
prop_allAxes_shrinkList :: NonEmptyList Int -> Property
prop_allAxes_shrinkList (getNonEmpty -> xs) =
    counterexample ("tensor: " ++ show tensor) $
    counterexample ("size: " ++ show size) $
          filter (not . null) (shrinkList (const []) xs)
      === [ Foldable.toList $ Tensor.axeWith axe tensor
          | axe <- Tensor.allAxes size
          ]
  where
    tensor :: Tensor Nat1 Int
    tensor = Tensor.fromList (length xs ::: VNil) xs

    size :: Tensor.Size Nat1
    size = Tensor.size tensor
