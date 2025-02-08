-- | Tensors (n-dimensional arrays)
--
-- This is an implementation of tensors that emphasizes simplicify above all; it
-- is meant for use in QuickCheck tests.
--
-- Intended for qualified import.
--
-- > import Test.Tensor (Tensor)
-- > import Test.Tensor qualified as Tensor
module Test.Tensor (
    -- * Definition
    Tensor(..)
  , getScalar
  , getTensor
    -- ** Convenience constructors
  , scalar
  , dim1
  , dim2
  , dim3
  , dim4
  , dim5
  , dim6
  , dim7
  , dim8
  , dim9
    -- * Size
  , Size
  , size
  , sizeAtLeast
    -- * Standard operations
  , zipWith
  , replicate
  , rotate
  , distrib
  , foreach
    -- * Subtensors
  , subs
  , subsWithStride
  , convolve
  , convolveWithStride
  , padWith
  , padWith'
    -- * Conversions
  , Lists
  , toLists
  , fromLists
  , fromList
    -- * QuickCheck support
    -- ** Generation
  , arbitraryOfSize
    -- ** Shrinking
  , Axe(..)
  , allAxes
  , axeWith
  , shrinkWith
  , shrinkWith'
  , shrinkElem
    -- * FFI
  , toStorable
  , fromStorable
  ) where

import Prelude hiding (zipWith, replicate)

import Control.Monad.Trans.State (StateT(..), evalStateT)
import Data.Foldable (foldl')
import Data.Foldable qualified as Foldable
import Data.List qualified as L
import Data.Proxy
import Data.Type.Nat
import Data.Vec.Lazy (Vec(..))
import Data.Vec.Lazy qualified as Vec
import Data.Vector.Storable qualified as Storable (Vector)
import Data.Vector.Storable qualified as Vector
import Foreign (Storable)
import GHC.Show (appPrec1, showSpace)
import GHC.Stack
import Numeric.Natural
import Test.QuickCheck (Arbitrary(..), Gen)
import Test.QuickCheck qualified as QC

{-------------------------------------------------------------------------------
  Definition
-------------------------------------------------------------------------------}

data Tensor n a where
  Scalar :: a -> Tensor Z a
  Tensor :: [Tensor n a] -> Tensor (S n) a

deriving stock instance Eq a => Eq (Tensor n a)

deriving stock instance Functor     (Tensor n)
deriving stock instance Traversable (Tensor n)
deriving stock instance Foldable    (Tensor n)

getScalar :: Tensor Z a -> a
getScalar (Scalar x) = x

getTensor :: Tensor (S n) a -> [Tensor n a]
getTensor (Tensor xs) = xs

{-------------------------------------------------------------------------------
  Size
-------------------------------------------------------------------------------}

type Size n = Vec n Int

-- | Analogue of 'List.length'
size :: Tensor n a -> Size n
size (Scalar _)  = VNil
size (Tensor xs) = L.length xs ::: size (L.head xs)

-- | Check that each dimension has at least the specified size
sizeAtLeast :: Size n -> Tensor n a -> Bool
sizeAtLeast sz = and . Foldable.toList . Vec.zipWith (<=) sz . size

{-------------------------------------------------------------------------------
  Standard operations
-------------------------------------------------------------------------------}

-- | Analogue of 'List.zipWith'
zipWith :: (a -> b -> c) -> Tensor n a -> Tensor n b -> Tensor n c
zipWith f (Scalar a)  (Scalar b)  = Scalar (f a b)
zipWith f (Tensor as) (Tensor bs) = Tensor $ L.zipWith (zipWith f) as bs

-- | Analogue of 'List.replicate'
replicate :: Size n -> a -> Tensor n a
replicate VNil       x = Scalar x
replicate (n ::: ns) x = Tensor $ L.replicate n (replicate ns x)

-- | Analogue of 'List.reverse'
--
-- This amounts to a 180 degrees rotation of the tensor.
rotate :: Tensor n a -> Tensor n a
rotate (Scalar x)  = Scalar x
rotate (Tensor xs) = Tensor $ map rotate (L.reverse xs)

-- | Distribute '[]' over 'Tensor'
--
-- Collects values in corresponding in all tensors.
distrib :: [Tensor n a] -> Tensor n [a]
distrib = \case
    []   -> error "distrib: empty list"
    t:ts -> go ((:[]) <$> t) ts
  where
    go :: Tensor n [a] -> [Tensor n a] -> Tensor n [a]
    go acc []     = reverse <$> acc
    go acc (t:ts) = go (zipWith (:) t acc) ts

-- | Map element over the first dimension of the tensor
foreach :: Tensor (S n) a -> (Tensor n a -> Tensor m b) -> Tensor (S m) b
foreach (Tensor as) f = Tensor (Prelude.map f as)

{-------------------------------------------------------------------------------
  Subtensors
-------------------------------------------------------------------------------}

-- | Subtensors of the specified size
subs :: SNatI n => Size n -> Tensor n a -> Tensor n (Tensor n a)
subs = subsWithStride (pure 1)

-- | Generalization of 'subs' with non-default stride
subsWithStride :: Vec n Int -> Size n -> Tensor n a -> Tensor n (Tensor n a)
subsWithStride VNil       VNil       (Scalar x)  = Scalar (Scalar x)
subsWithStride (s ::: ss) (n ::: ns) (Tensor xs) = Tensor [
      Tensor <$> distrib selected
    | selected <- everyNth s $ consecutive n (map (subsWithStride ss ns) xs)
    ]

-- | Convolution
--
-- See 'padWith' for adjusting boundary conditions.
convolve ::
     (SNatI n, Num a)
  => Tensor n a  -- ^ Kernel
  -> Tensor n a  -- ^ Input
  -> Tensor n a
convolve = convolveWithStride (pure 1)

-- | Generalization of 'convolve' when using a non-default stride
convolveWithStride :: forall n a.
     Num a
  => Vec n Int   -- ^ Stride
  -> Tensor n a  -- ^ Kernel
  -> Tensor n a  -- ^ Input
  -> Tensor n a
convolveWithStride stride kernel input =
    aux <$> subsWithStride stride (size kernel) input
  where
    aux :: Tensor n a -> a
    aux = foldl' (+) 0 . zipWith (*) kernel

{-------------------------------------------------------------------------------
  Padding
-------------------------------------------------------------------------------}

-- | Add uniform padding
padWith :: SNatI n => a -> Int -> Tensor n a -> Tensor n a
padWith padding n = padWith' padding (pure (n, n))

-- | Generalization of 'padWith' with different padding per dimension
padWith' :: forall n a. a -> Vec n (Int, Int) -> Tensor n a -> Tensor n a
padWith' padding paddingSize tensor =
    go paddingSize newSize tensor
  where
    newSize :: Size n
    newSize = Vec.zipWith (\(b, a) n -> n + b + a) paddingSize (size tensor)

    go :: forall m. Vec m (Int, Int) -> Size m -> Tensor m a -> Tensor m a
    go VNil                     VNil       (Scalar x)  = Scalar x
    go ((before, after) ::: ps) (_ ::: ns) (Tensor xs) = Tensor $ concat [
          L.replicate before $ replicate ns padding
        , map (go ps ns) xs
        , L.replicate after $ replicate ns padding
        ]

{-------------------------------------------------------------------------------
  QuickCheck support
-------------------------------------------------------------------------------}

arbitraryOfSize :: Size n -> Gen a -> Gen (Tensor n a)
arbitraryOfSize sz = sequence . replicate sz

data Axe (n :: Nat) where
  -- | Axe some elements from the current dimension
  --
  -- We record which elements to drop as an @(offset, length)@ pair.
  AxeHere :: (Int, Int) -> Axe (S n)

  -- | Axe some elements from a nested dimension
  --
  -- In order to keep the tensor square, we must apply the same axe for every
  -- element of the /current/ dimension
  AxeNested :: Axe n -> Axe (S n)

deriving instance Show (Axe n)

-- | All possible ways to axe some elements
--
-- This is adopted from the implementation of 'shrinkList' (in a way, an 'Axe'
-- is an explanation of the decisions made by 'shrinkList', generalized to
-- multiple dimensions).
allAxes :: Size n -> [Axe n]
allAxes sz =
    case sz of
      VNil     -> []
      n ::: ns -> concat [
            -- Drop in this dimension
            concat [
                L.map AxeHere (removes 0 k n)
              | k <- takeWhile (> 0) (iterate (`div` 2) (n `div` 2))
              ]

            -- Drop in a nested dimension
          , L.map AxeNested (allAxes ns)
          ]
  where
    removes :: Int -> Int -> Int -> [(Int, Int)]
    removes offset k n
      | k > n     = []
      | otherwise = (offset, k) : removes (offset + k) k (n - k)

axeWith :: Axe n -> Tensor n a -> Tensor n a
axeWith (AxeHere (offset, len)) (Tensor xss) = Tensor $
    let (keep, dropFrom) = L.splitAt offset xss
    in keep <> drop len dropFrom
axeWith (AxeNested axe) (Tensor xss) = Tensor $
    L.map (axeWith axe) xss

-- | Shrink tensor
shrinkWith :: (a -> [a]) -> Tensor n a -> [Tensor n a]
shrinkWith f xs = shrinkWith' (allAxes (size xs)) f xs

-- | Generalization of 'shrinkWith'
shrinkWith' :: forall n a.
     [Axe n]     -- ^ Shrink the size of the tensor (see 'allAxes')
  -> (a -> [a])  -- ^ Shrink elements of the tensor
  -> Tensor n a -> [Tensor n a]
shrinkWith' axes f xss = concat [
      [axeWith axe xss | axe <- axes]
    , shrinkElem f xss
    ]

-- | Shrink an element of the tensor, leaving the size of the tensor unchanged
--
-- This is an internal function.
shrinkElem :: (a -> [a]) -> Tensor n a -> [Tensor n a]
shrinkElem f (Scalar x)   = Scalar <$> f x
shrinkElem f (Tensor xss) = [
      Tensor $ before ++ [xs'] ++ after
    | (before, xs, after) <- pickOne xss
    , xs' <- shrinkElem f xs
    ]

instance (SNatI n, Arbitrary a) => Arbitrary (Tensor n a) where
  arbitrary = QC.sized $ \n -> do
      sz :: Size n <- QC.liftArbitrary $ QC.choose (1, 1 + n)
      arbitraryOfSize sz arbitrary

  shrink = shrinkWith shrink

{-------------------------------------------------------------------------------
  FFI
-------------------------------------------------------------------------------}

-- | Translate to storable vector
--
-- The tensor is laid out in order specified (outer dimensions before inner).
toStorable :: Storable a => Tensor n a -> Storable.Vector a
toStorable = Vector.fromList . Foldable.toList

-- | Translate from storable vector
--
-- Throws an exception if the vector does not contain enough elements.
fromStorable ::
     (HasCallStack, Storable a)
  => Size n -> Storable.Vector a -> Tensor n a
fromStorable sz = aux . fromList sz . Vector.toList
  where
    aux :: Maybe (Tensor n a) -> Tensor n a
    aux Nothing       = error "fromStorable: not enough elements"
    aux (Just tensor) = tensor

{-------------------------------------------------------------------------------
  Convenience constructors
-------------------------------------------------------------------------------}

scalar :: a -> Tensor Nat0 a
scalar = fromLists

dim1 :: [a] -> Tensor Nat1 a
dim1 = fromLists

dim2 :: [[a]] -> Tensor Nat2 a
dim2 = fromLists

dim3 :: [[[a]]] -> Tensor Nat3 a
dim3 = fromLists

dim4 :: [[[[a]]]] -> Tensor Nat4 a
dim4 = fromLists

dim5 :: [[[[[a]]]]] -> Tensor Nat5 a
dim5 = fromLists

dim6 :: [[[[[[a]]]]]] -> Tensor Nat6 a
dim6 = fromLists

dim7 :: [[[[[[[a]]]]]]] -> Tensor Nat7 a
dim7 = fromLists

dim8 :: [[[[[[[[a]]]]]]]] -> Tensor Nat8 a
dim8 = fromLists

dim9 :: [[[[[[[[[a]]]]]]]]] -> Tensor Nat9 a
dim9 = fromLists

{-------------------------------------------------------------------------------
  Conversions

  This is primarily useful for specify tensor constants.
-------------------------------------------------------------------------------}

type family Lists n a where
  Lists Z     a = a
  Lists (S n) a = [Lists n a]

toLists :: Tensor n a -> Lists n a
toLists (Scalar x)  = x
toLists (Tensor xs) = map toLists xs

fromLists :: SNatI n => Lists n a -> Tensor n a
fromLists = go snat
  where
    go :: SNat n -> Lists n a -> Tensor n a
    go SZ = Scalar
    go SS = Tensor . map (go snat)

-- | Inverse to 'Foldable.toList'
--
-- Returns 'Nothing' if the list does not have enough elements.
fromList :: forall n a. Size n -> [a] -> Maybe (Tensor n a)
fromList sz xs =
    flip evalStateT xs $ sequenceA (replicate sz genElem)
  where
    genElem :: StateT [a] Maybe a
    genElem = StateT L.uncons

{-------------------------------------------------------------------------------
  Show instance
-------------------------------------------------------------------------------}

showLists :: Show a => Proxy a -> SNat n -> (Show (Lists n a) => r) -> r
showLists _ SZ      k = k
showLists p (SS' n) k = showLists p n k

showConstructor :: Int -> SNat n -> ShowS
showConstructor p sn
  | n' == 0            = showString "scalar"
  | 1 <= n' && n' <= 9 = showString "dim" . shows n'
  | otherwise          = showString "fromLists @"
                       . explicitShowsPrec p (snatToNat sn)
  where
    n' :: Natural
    n' = snatToNatural sn

instance Show a => Show (Tensor n a) where
  showsPrec p tensor = showLists (Proxy @a) (tensorSNat tensor) $
      showParen (p >= appPrec1) $
          showConstructor appPrec1 (tensorSNat tensor)
        . showSpace
        . showsPrec appPrec1 (toLists tensor)

{-------------------------------------------------------------------------------
  Internal auxiliary: SNat
-------------------------------------------------------------------------------}

tensorSNatI :: Tensor n a -> (SNatI n => r) -> r
tensorSNatI (Scalar _)  k = k
tensorSNatI (Tensor xs) k = tensorSNatI (L.head xs) k

tensorSNat :: Tensor n a -> SNat n
tensorSNat tensor = tensorSNatI tensor snat

{-------------------------------------------------------------------------------
  Internal auxiliary: lists
-------------------------------------------------------------------------------}

-- | Consecutive elements
--
-- >    consecutive 3 [1..5]
-- > == [[1,2,3],[2,3,4],[3,4,5]]
consecutive :: Int -> [a] -> [[a]]
consecutive n = L.takeWhile ((== n) . length) . fmap (L.take n) . L.tails

-- | Every nth element of the list
--
-- Examples
--
-- > everyNth 1 [0..9] == [0,2,3,4,5,6,7,8,9]
-- > everyNth 2 [0..9] == [0,2,4,6,8]
-- > everyNth 3 [0..9] == [0,3,6,9]
everyNth :: forall a. Int -> [a] -> [a]
everyNth n = \xs ->
    if n > 0
      then go xs
      else error "everyNth: n should be strictly positive"
  where
    go :: [a] -> [a]
    go []     = []
    go (x:xs) = x : go (drop (n - 1) xs)

-- | Single out an element from the list
--
-- >    pickOne [1..4]
-- > == [ ( []      , 1 , [2,3,4] )
-- >    , ( [1]     , 2 , [3,4]   )
-- >    , ( [1,2]   , 3 , [4]     )
-- >    , ( [1,2,3] , 4 , []      )
-- >    ]
pickOne :: forall a. [a] -> [([a], a, [a])]
pickOne = \case
    []   -> error "pickOne: empty list"
    x:xs -> go [] x xs
  where
    go :: [a] -> a -> [a] -> [([a], a, [a])]
    go acc x []     = [(reverse acc, x, [])]
    go acc x (y:ys) = (reverse acc, x, (y:ys)) : go (x:acc) y ys
