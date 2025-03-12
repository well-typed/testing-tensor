module TestSuite.Test.Convolution (tests) where

import Data.List qualified as L
import Data.Type.Nat
import Data.Vec.Lazy (Vec(..))
import Test.Tasty
import Test.Tasty.HUnit
import Test.Tasty.QuickCheck

import Test.Tensor (Tensor)
import Test.Tensor qualified as Tensor
import Test.Tensor.TestValue

import TestSuite.Test.Convolution.Examples3B1B

{-------------------------------------------------------------------------------
  List of tests
-------------------------------------------------------------------------------}

tests :: TestTree
tests = testGroup "TestSuite.Test.Convolution.Prop" [
      testGroup "Examples" [
          testCase "rotate"       example_rotate
        , testCase "distrib_dim2" example_distrib_dim2
        , testCase "subs_dim1"    example_subs_dim1
        , testCase "subs_dim2"    example_subs_dim2
        , testCase "subs_dim3"    example_subs_dim3
        , testCase "padWith"      example_padWith
        , testCase "padWith'"     example_padWith'
        ]
    , testGroup "3B1B" [
          testCase "simple"                example_3b1b_simple
        , testCase "movingAverage"         example_3b1b_movingAverage
        , testCase "movingWeightedAverage" example_3b1b_movingWeightedAverage
        , testCase "weightedDice"          example_3b1b_weightedDice
        ]
    , testGroup "Properties" [
          testProperty "distrib_dim0" prop_distrib_dim0
        , testProperty "distrib_dim1" prop_distrib_dim1
        ]
    ]

{-------------------------------------------------------------------------------
  Examples
-------------------------------------------------------------------------------}

example_rotate :: Assertion
example_rotate =
    assertEqual "" expected $
      Tensor.rotate (Tensor.dim2 [ [1,2,3], [4,5,6] ])
  where
    expected :: Tensor Nat2 Integer
    expected = Tensor.dim2 [ [6,5,4], [3,2,1] ]

example_distrib_dim2 :: Assertion
example_distrib_dim2 =
    assertEqual "" expected $
      Tensor.distrib (Tensor.size expected) input
  where
    input :: [Tensor Nat2 Int]
    input = [
          Tensor.dim2 [[111, 112, 113, 114], [121, 122, 123, 124], [131, 132, 133, 134]]
        , Tensor.dim2 [[211, 212, 213, 214], [221, 222, 223, 224], [231, 232, 233, 234]]
        , Tensor.dim2 [[311, 312, 313, 314], [321, 322, 323, 324], [331, 332, 333, 334]]
        , Tensor.dim2 [[411, 412, 413, 414], [421, 422, 423, 424], [431, 432, 433, 434]]
        , Tensor.dim2 [[511, 512, 513, 514], [521, 522, 523, 524], [531, 532, 533, 534]]
        ]

    expected :: Tensor Nat2 [Int]
    expected = Tensor.dim2 [
          [ [111,211,311,411,511]
          , [112,212,312,412,512]
          , [113,213,313,413,513]
          , [114,214,314,414,514]
          ]
        , [ [121,221,321,421,521]
          , [122,222,322,422,522]
          , [123,223,323,423,523]
          , [124,224,324,424,524]
          ]
        , [ [131,231,331,431,531]
          , [132,232,332,432,532]
          , [133,233,333,433,533]
          , [134,234,334,434,534]
          ]
        ]

example_subs_dim1 :: Assertion
example_subs_dim1 =
    assertEqual "" expected $
      Tensor.subs (2 ::: VNil) $
        Tensor.dim1 [1,2,3]
  where
    expected :: Tensor Nat1 (Tensor Nat1 Int)
    expected = Tensor.dim1 [ Tensor.dim1 [1,2], Tensor.dim1 [2,3] ]

example_subs_dim2 :: Assertion
example_subs_dim2 =
    assertEqual "" expected $
      Tensor.subs (2 ::: 2 ::: VNil) $
        Tensor.dim2 [[11,12,13],[21,22,23],[31,32,33]]
  where
    expected :: Tensor Nat2 (Tensor Nat2 Int)
    expected = Tensor.dim2 [
          [ Tensor.dim2 [[11,12],[21,22]], Tensor.dim2 [[12,13],[22,23]] ]
        , [ Tensor.dim2 [[21,22],[31,32]], Tensor.dim2 [[22,23],[32,33]] ]
        ]

example_subs_dim3 :: Assertion
example_subs_dim3 =
    assertEqual "" expected $
      Tensor.subs (2 ::: 2 ::: 2 ::: VNil) $
        Tensor.dim3 [
            [[111,112,113],[121,122,123],[131,132,133]]
          , [[211,212,213],[221,222,223],[231,232,233]]
          , [[311,312,313],[321,322,323],[331,332,333]]
          ]
  where
    expected :: Tensor Nat3 (Tensor Nat3 Int)
    expected = Tensor.dim3 [
            [ [ Tensor.dim3 [[[111,112],[121,122]],[[211,212],[221,222]]]
              , Tensor.dim3 [[[112,113],[122,123]],[[212,213],[222,223]]]
              ]
            , [ Tensor.dim3 [[[121,122],[131,132]],[[221,222],[231,232]]]
              , Tensor.dim3 [[[122,123],[132,133]],[[222,223],[232,233]]]
              ]
            ]
          , [ [ Tensor.dim3 [[[211,212],[221,222]],[[311,312],[321,322]]]
              , Tensor.dim3 [[[212,213],[222,223]],[[312,313],[322,323]]]
              ]
            , [ Tensor.dim3 [[[221,222],[231,232]],[[321,322],[331,332]]]
              , Tensor.dim3 [[[222,223],[232,233]],[[322,323],[332,333]]]
              ]
            ]
        ]

example_padWith :: Assertion
example_padWith =
    assertEqual "" expected $
      Tensor.padWith 0 2 $ Tensor.dim2 [ [1, 2, 3], [4, 5, 6] ]
  where
    expected :: Tensor Nat2 Int
    expected = Tensor.dim2 [
          [ 0, 0, 0, 0, 0, 0, 0 ]
        , [ 0, 0, 0, 0, 0, 0, 0 ]
        , [ 0, 0, 1, 2, 3, 0, 0 ]
        , [ 0, 0, 4, 5, 6, 0, 0 ]
        , [ 0, 0, 0, 0, 0, 0, 0 ]
        , [ 0, 0, 0, 0, 0, 0, 0 ]
        ]

example_padWith' :: Assertion
example_padWith' =
    assertEqual "" expected $
      Tensor.padWith' 0 ((1, 1) ::: (2, 3) ::: VNil) (Tensor.dim2 [[1]])
  where
    expected :: Tensor Nat2 Int
    expected = Tensor.dim2 [
          [0,0,0,0,0,0]
        , [0,0,1,0,0,0]
        , [0,0,0,0,0,0]
        ]

{-------------------------------------------------------------------------------
  Examples from the 3B1B video
-------------------------------------------------------------------------------}

example_3b1b ::
     Tensor Nat1 TestValue -- ^ Input (padded)
  -> Tensor Nat1 TestValue -- ^ Kernel
  -> Tensor Nat1 TestValue -- ^ Expected result
  -> Assertion
example_3b1b input kernel result =
    assertEqual "" result $
      Tensor.convolve kernel input

example_3b1b_simple :: Assertion
example_3b1b_simple =
    example_3b1b
      (Tensor.padWith 0 2 $ Tensor.dim1 simpleInput)
      (Tensor.dim1 simpleKernel)
      (Tensor.dim1 simpleResult)

example_3b1b_weightedDice :: Assertion
example_3b1b_weightedDice =
    example_3b1b
      (Tensor.padWith 0 5 $ Tensor.dim1 weightedDiceInput)
      (Tensor.dim1 weightedDiceKernel)
      (Tensor.dim1 weightedDiceResult)

example_3b1b_movingAverage :: Assertion
example_3b1b_movingAverage =
    example_3b1b
      (Tensor.padWith 0 2 $ Tensor.dim1 movingAverageInput)
      (Tensor.dim1 movingAverageKernel)
      (Tensor.dim1 movingAverageResult)

example_3b1b_movingWeightedAverage :: Assertion
example_3b1b_movingWeightedAverage =
    example_3b1b
      (Tensor.padWith 0 2 $ Tensor.dim1 movingAverageInput)
      (Tensor.dim1 movingWeightedAverageKernel)
      (Tensor.dim1 movingWeightedAverageResult)

{-------------------------------------------------------------------------------
  Properties
-------------------------------------------------------------------------------}

-- | Distribute over a list of 0-D tensor is the identity
prop_distrib_dim0 :: NonEmptyList Int -> Property
prop_distrib_dim0 (getNonEmpty -> xs) =
        Tensor.toLists (Tensor.distrib size (map Tensor.scalar xs))
    === xs
  where
    size :: Tensor.Size Nat0
    size = VNil

-- | Distribute over a list of 1-D tensor is 'transpose'
--
-- This is true only for rectangular input.
prop_distrib_dim1 :: NonEmptyList (NonEmptyList Int) -> Property
prop_distrib_dim1 (getSameLength -> xss) =
    counterexample ("input: " ++ show xss) $
          Tensor.toLists (Tensor.distrib size (map Tensor.dim1 xss))
      === L.transpose xss
  where
    size :: Tensor.Size Nat1
    size = length (L.head xss) ::: VNil

{-------------------------------------------------------------------------------
  Auxiliary
-------------------------------------------------------------------------------}

getNonEmpty2 :: NonEmptyList (NonEmptyList a) -> [[a]]
getNonEmpty2 = map getNonEmpty . getNonEmpty

getSameLength :: NonEmptyList (NonEmptyList a) -> [[a]]
getSameLength = aux . getNonEmpty2
  where
    aux :: [[a]] -> [[a]]
    aux xss = map (take (minimum $ map length xss)) xss