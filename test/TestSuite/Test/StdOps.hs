module TestSuite.Test.StdOps (tests) where

import Data.Foldable qualified as Foldable
import Data.Type.Nat
import Test.Tasty
import Test.Tasty.QuickCheck

import Test.Tensor (Tensor)
import Test.Tensor qualified as Tensor

{-------------------------------------------------------------------------------
  List of tests
-------------------------------------------------------------------------------}

tests :: TestTree
tests = testGroup "TestSuite.Test.StdOps" [
      testGroup "properties" [
            testGroup "fromList_toList" [
                testProperty "dim0" $ prop_fromList_toList @Nat0
              , testProperty "dim1" $ prop_fromList_toList @Nat1
              , testProperty "dim2" $ prop_fromList_toList @Nat2
              , testProperty "dim3" $
                  withMaxSuccess 100 $ -- random 3D tensors get large quick
                    prop_fromList_toList @Nat3
              ]
          , testProperty "distrib_transpose" $ prop_distrib_transpose
        ]
    ]

{-------------------------------------------------------------------------------
  Properties
-------------------------------------------------------------------------------}

prop_fromList_toList :: SNatI n => Tensor n Int -> Property
prop_fromList_toList tensor =
        Tensor.fromList (Tensor.size tensor) (Foldable.toList tensor)
    === tensor

prop_distrib_transpose :: Tensor Nat2 Int -> Property
prop_distrib_transpose tensor =
        (restructure . Tensor.distrib . Tensor.getTensor $ tensor)
    === (Tensor.transpose $ tensor)
  where
    restructure :: Tensor Nat1 [Int] -> Tensor Nat2 Int
    restructure = Tensor.fromLists . Tensor.toLists
