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
        , testProperty "axeSize" prop_axeSize
        , testProperty "length_zeroWith" prop_length_zeroWith
        ]
    ]

{-------------------------------------------------------------------------------
  Examples
-------------------------------------------------------------------------------}

example_shrinkWith :: Assertion
example_shrinkWith =
    assertEqual "" expected $
      Tensor.shrinkWith
        (Just $ Tensor.Zero (-1))
        (const [0])
        (Tensor.dim2 [[1,2,3], [4,5,6]])
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
          -- Zero outer dimension
        , Tensor.dim2 [[-1,-1,-1],[4,5,6]]
        , Tensor.dim2 [[1,2,3],[-1,-1,-1]]
          -- Zero inner dimension
        , Tensor.dim2 [[-1,2,3],[-1,5,6]]
        , Tensor.dim2 [[1,-1,3],[4,-1,6]]
        , Tensor.dim2 [[1,2,-1],[4,5,-1]]
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

prop_axeSize :: Tensor Nat2 Int -> Property
prop_axeSize tensor = conjoin [
      counterexample ("axe: " ++ show axe) $
            length (Tensor.axeWith axe tensor)
        === length tensor - Tensor.axeSize size axe
    | axe <- Tensor.allAxes size
    ]
  where
    size :: Tensor.Size Nat2
    size = Tensor.size tensor

prop_length_zeroWith :: Tensor Nat2 Int -> Property
prop_length_zeroWith tensor = conjoin [
      counterexample ("axe: " ++ show axe) $
        case Tensor.zeroWith Tensor.zero axe tensor of
          Nothing      -> property True
          Just tensor' -> length tensor' === length tensor
    | axe <- Tensor.allAxes size
    ]
  where
    size :: Tensor.Size Nat2
    size = Tensor.size tensor
